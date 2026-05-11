"""batchrunner for running a factorial experiment design over a model.

To take advantage of parallel execution of experiments, `batch_run` uses
multiprocessing if ``number_processes`` is larger than 1. It is strongly advised
to only run in parallel using a normal python file (so don't try to do it in a
jupyter notebook). This is because Jupyter notebooks have a different execution
model that can cause issues with Python's multiprocessing module, especially on
Windows. The main problems include the lack of a traditional __main__ entry
point, serialization issues, and potential deadlocks.

Moreover, best practice when using multiprocessing is to
put the code inside an ``if __name__ == '__main__':`` code black as shown below::

    from mesa.batchrunner import batch_run

    params = {"width": 10, "height": 10, "N": range(10, 500, 10)}

    if __name__ == '__main__':
        results = batch_run(
            MoneyModel,
            parameters=params,
            iterations=5,
            max_steps=100,
            number_processes=None,
            data_collection_period=1,
            display_progress=True,
        )

"""

import os
import math
import itertools
import multiprocessing
import json
import gc
import random
import numpy as np
import export.runs
import batch.messaging

from collections.abc import Iterable, Mapping
from functools import partial
from multiprocessing import Pool
from typing import Any, Dict
from tqdm.auto import tqdm

from mesa.model import Model

from model.model import ReductionModel
from batch.params import dict_to_params

multiprocessing.set_start_method("spawn", force=True)

last_update_percentage = 0


# idek
# for some reason all my ints are suddenly np.int64
# so I need to define this to get rid of them
class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


def batch_run(
    model_cls: type[ReductionModel],
    sweeps_dir: str,
    current_sweep: str,
    parameters: Mapping[str, Any | Iterable[Any]],
    # We still retain the Optional[int] because users may set it to None (i.e. use all CPUs)
    number_processes: int | None = 1,
    iterations: int = 1,
    data_collection_period: int = -1,
    max_steps: int = 1000,
    datacollector_step_size: float = 1,
    display_progress: bool = True,
    webhook: batch.messaging.Webhook | None = None,
) -> list[dict[str, Any]]:
    """Batch run a mesa model with a set of parameter values.

    Args:
        model_cls (Type[ReductionModel]): The model class to batch-run
        parameters (Mapping[str, Union[Any, Iterable[Any]]]): Dictionary with model parameters over which to run the model. You can either pass single values or iterables.
        number_processes (int, optional): Number of processes used, by default 1. Set this to None if you want to use all CPUs.
        iterations (int, optional): Number of iterations for each parameter combination, by default 1
        data_collection_period (int, optional): Number of steps after which data gets collected, by default -1 (end of episode)
        max_steps (int, optional): Maximum number of model steps after which the model halts, by default 1000
        datacollector_step_size (int, optional): The step interval between runs of the data collector, by default 1
        display_progress (bool, optional): Display batch run process, by default True
        webhook (batch.messaging.Webhook, optional): Webhook info to be used during batch run

    Returns:
        List[Dict[str, Any]]

    Notes:
        batch_run assumes the model has a `datacollector` attribute that has a DataCollector object initialized.

    """

    runs_list = []

    # Keep track of each combination of arguments
    # Then keep track of each iteration for each combination
    # Makes it MUCH easier to find the different parameter combinations later
    run_id = 0
    combination_id = 0
    for kwargs in _make_model_kwargs(parameters):
        for iteration in range(iterations):
            # We have to make a new kwargs object in order to prevent the seed from sticking into this specific combinatino
            if "seed" not in kwargs:
                kwargs_pass = {**kwargs, "seed": random.randint(0, 99999999)}
            else:
                kwargs_pass = kwargs

            # Set the datacollector step size dynamically based on the max steps
            kwargs_pass["datacollector_step_size"] = datacollector_step_size

            runs_list.append(
                (
                    run_id,
                    combination_id,
                    iteration,
                    sweeps_dir,
                    current_sweep,
                    kwargs_pass,
                )
            )
            run_id += 1
        combination_id += 1

    process_func = partial(
        _model_run_func,
        model_cls,
        iterations=iterations,
        max_steps=max_steps,
        data_collection_period=data_collection_period,
    )

    results: list[dict[str, Any]] = []

    with tqdm(total=len(runs_list), disable=not display_progress) as pbar:
        pbar.update()
        if number_processes == 1:
            for run in runs_list:
                data = process_func(run)
                results.extend(data)
                pbar.update()
                if webhook is not None:
                    check_webhook_update(pbar, current_sweep, webhook)
        else:
            with Pool(number_processes, maxtasksperchild=1) as p:
                for data in p.imap_unordered(process_func, runs_list):
                    results.extend(data)
                    pbar.update()
                    if webhook is not None:
                        check_webhook_update(pbar, current_sweep, webhook)

    return results


def _make_model_kwargs(
    parameters: Mapping[str, Any | Iterable[Any]],
) -> list[dict[str, Any]]:
    """Create model kwargs from parameters dictionary.

    Parameters
    ----------
    parameters : Mapping[str, Union[Any, Iterable[Any]]]
        Single or multiple values for each model parameter name.

        Allowed values for each parameter:
        - A single value (e.g., `32`, `"relu"`).
        - A non-empty iterable (e.g., `[0.01, 0.1]`, `["relu", "sigmoid"]`).

        Not allowed:
        - Empty lists or empty iterables (e.g., `[]`, `()`, etc.). These should be removed manually.

    Returns:
    -------
    List[Dict[str, Any]]
        A list of all kwargs combinations.
    """
    parameter_list = []
    for param, values in parameters.items():
        if isinstance(values, str):
            # The values is a single string, so we shouldn't iterate over it.
            all_values = [(param, values)]
        elif isinstance(values, list | tuple | set) and len(values) == 0:
            # If it's an empty iterable, raise an error
            raise ValueError(
                f"Parameter '{param}' contains an empty iterable, which is not allowed."
            )

        else:
            try:
                all_values = [(param, value) for value in values]
            except TypeError:
                all_values = [(param, values)]
        parameter_list.append(all_values)
    all_kwargs = itertools.product(*parameter_list)
    kwargs_list = [dict(kwargs) for kwargs in all_kwargs]
    return kwargs_list


def serialise_data_collector(model: ReductionModel) -> Dict[str, Any]:
    """Serialised a datacollector by turning numpy arrays into lists recursively

    Args:
        model (model.ReductionModel): the model of which to serialise the datacollector

    Returns:
        Dict: the serialised data collector, as a dict
    """

    output_dict = {}
    df = model.datacollector.get_model_vars_dataframe()
    for column in df.columns:
        # Convert each column to list, also for INNER values !!
        output_dict[column] = (
            df[column]
            .apply(lambda x: x.tolist() if hasattr(x, "tolist") else x)
            .tolist()
        )

    return output_dict


def check_webhook_update(
    pbar: tqdm, selected_sweep: str, webhook: batch.messaging.Webhook
):
    """Sends a webhook update during the batch run process.

    Args:
        pbar (tqdm): the tqdm progress bar
        selected_sweep (str): the name of the sweep currently in progress
        webhook (batch.messaging.Webhook): the webhook configuration
    """

    global last_update_percentage

    progress = int(pbar.n / pbar.total * 100)

    # Not enough models to really be a long batch run
    if pbar.total < 100:
        return

    if progress >= last_update_percentage + 10 and progress < 100:
        last_update_percentage = math.floor(progress / 10) * 10

        webhook.handle_event(
            batch.messaging.Event(
                selected_sweep,
                batch.messaging.EventState.PROGRESS,
                batch.messaging.EventType.BATCH_RUN,
                progress=progress,
            )
        )


def _model_run_func(
    model_cls: type[ReductionModel],
    run: tuple[int, int, int, str, str, dict[str, Any]],
    max_steps: int,
    iterations: int,
    data_collection_period: int,
) -> list[dict[str, Any]]:
    """Run a single model run and collect model and agent data.

    Parameters
    ----------
    model_cls : Type[ReductionModel]
        The model class to batch-run
    run: Tuple[int, int, int, str, str, Dict[str, Any]]
        The run id, combination id, iteration number, sweeps directory, current sweep and kwargs for this run
    max_steps : int
        Maximum number of model steps after which the model halts, by default 1000
    iterations : int
        How many times a parameter combination is run
    data_collection_period : int
        Number of steps after which data gets collected

    Returns:
    -------
    List[Dict[str, Any]]
        Return model_data, agent_data from the reporters
    """
    run_id, combination_id, iteration, sweeps_dir, current_sweep, kwargs = run
    model_params = dict_to_params(kwargs)
    model = model_cls(model_params)
    while model.running and model.steps <= max_steps:
        model.step()

    run_data_path = export.runs.make_run_data_path(
        sweeps_dir, current_sweep, run_id, create=True
    )

    with open(run_data_path, "wt") as model_file:
        output_data = serialise_data_collector(model)

        model_file.write(json.dumps(output_data, cls=NpEncoder))
        # pickle.dump(model, model_file)

    data = [
        {
            "run_id": run_id,
            "combination_id": combination_id,
            "iteration": iteration,
            "max_steps": max_steps,
            "steps": model.steps,
            **kwargs,
        }
    ]

    del model
    gc.collect()

    return data

    steps = list(range(0, (model.steps // model.datacollector_step_size) + 1, 1))
    print(steps)
    # if not steps or steps[-1] != model.steps - 1:
    #     steps.append(model.steps - 1)

    for step in steps:
        model_data, all_agents_data = _collect_data(model, step)

        # If there are agent_reporters, then create an entry for each agent
        if all_agents_data:
            stepdata = [
                {
                    "RunId": run_id,
                    "iteration": iteration,
                    "Step": step,
                    **kwargs,
                    **model_data,
                    **agent_data,
                }
                for agent_data in all_agents_data
            ]
        # If there is only model data, then create a single entry for the step
        else:
            stepdata = [
                {
                    "RunId": run_id,
                    "iteration": iteration,
                    "Step": step,
                    **kwargs,
                    **model_data,
                }
            ]
        data.extend(stepdata)

    return data


def _collect_data(
    model: ReductionModel,
    step: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Collect model and agent data from a model using mesas datacollector."""
    if not hasattr(model, "datacollector"):
        raise AttributeError(
            "The model does not have a datacollector attribute. Please add a DataCollector to your model."
        )
    dc = model.datacollector

    model_data = {param: values[step] for param, values in dc.model_vars.items()}

    all_agents_data = []
    raw_agent_data = dc._agent_records.get(step, [])
    for data in raw_agent_data:
        agent_dict = {"AgentID": data[1]}
        agent_dict.update(zip(dc.agent_reporters, data[2:]))
        all_agents_data.append(agent_dict)
    return model_data, all_agents_data
