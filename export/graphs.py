from dataclasses import dataclass, field
from typing import Callable, Dict, Any, Optional, List, Dict, Sequence, Union, Tuple

import copy
import matplotlib.axes
import matplotlib.figure
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import model.reporters_agent
import model.reporters_model

import export.sweeps
import export.runs
import export.combinations

import batch.aggregate
import visualisation.multiplot
import visualisation.energy
import visualisation.communication


class GraphContext:
    """Describes the context in which a graph can appear. This can be either be: (1) export for paper output (2) dashboard for analysis and exploration."""

    EXPORT = 0
    DASHBOARD = 1


@dataclass
class GraphConfig:
    """The configuration for a single graph. It defines what column the graph data comes from, what function needs to be called to create the graph, how extra parameters can be retrieved, etc."""

    reporter_name: (
        str | List[str]
    )  # What model reporter does the required data come from?
    plot_func: Callable  # How can the figure be made?
    model_reporter: bool = (
        False  # Is this a model reporter (True) or an agent reporter? (False)
    )
    reporter_type: int = model.reporters_agent.ReporterType.MEDIAN  # MEAN, MEDIAN, NONE
    data_columns: List[str] = field(default_factory=lambda: [])
    agent_types: List[int] = field(
        default_factory=lambda: [model.reporters_agent.AgentType.ALL]
    )
    # Shorthands for common operations that are given to plot functions.
    # can be: x_scale_factor, min_data or max_data
    common_args: List[str] = field(default_factory=lambda: [])
    extra_args: Optional[Dict[str, Any]] = (
        None  # What extra arguments are needed to plot this figure?  # These can be either Callables or constants
    )
    # Interactive arguments that can be received from the interface
    interactive_args: List[str] | None = None
    action_column: str = "median"  # Aggregate operation column to use data from
    action_column_inner: str = "median"  # Combination operation column to use data from
    aggregate: bool = False  # Aggregate graph or not?
    aggregate_extension: bool = (
        False  # Can this graph be extended to become an overlayed aggregate graph?
    )
    is_mosaic: bool = False  # Is this a mosaic graph?
    single_run_sensible: bool = (
        True  # Does it make sense to show this graph for a single run?
    )
    context: int = GraphContext.EXPORT  # In what context should this graph be shown?

    # Allow setting data columns directly if needed
    disable_autogenerate_columns: bool = False

    def __post_init__(self):
        if isinstance(self.reporter_name, list):
            reporter_names = self.reporter_name
        elif isinstance(self.reporter_name, str):
            reporter_names = [self.reporter_name]
        else:
            raise ValueError("Unrecognised reporter name type")

        if self.disable_autogenerate_columns:
            return

        if self.aggregate:
            self.data_columns = reporter_names
            return

        if self.model_reporter:
            for reporter_name in reporter_names:
                self.data_columns += [
                    model.reporters_model.get_model_reporter_key(
                        reporter_name, self.reporter_type
                    )
                ]
            return

        for reporter_name in reporter_names:
            self.data_columns += [
                model.reporters_agent.get_model_reporter_key(
                    reporter_name, self.reporter_type, agent_type
                )
                for agent_type in self.agent_types
            ]


@dataclass
class MosaicConfig:
    """The configuration for a mosaic graph. It defines what other graphs are part of the mosaic, and in what order they need to be arranged on the mosaic."""

    layout: List[List[str]]  # Names of other graphs
    size: Tuple[int, int] = (10, 16)  # Size of the mosaic
    is_mosaic: bool = True  # Is this a mosaic graph?
    context: int = GraphContext.DASHBOARD  # In what context should this graph be shown?
    single_run_sensible: bool = (
        True  # Does it make sense to show this graph for a single run?
    )
    aggregate: bool = False  # Aggregate graph or not?
    aggregate_extension: bool = (
        False  # Mosaic graph can pass the extension to its children
    )
    extra_args: List[List[Dict[str, Any]]] | None = (
        None  # What extra arguments are needed to plot this figure? These can be either Callables or constants
        # These will be passed 1:1 to the child graphs, so be careful!
    )
    # Interactive arguments that can be received from the interface
    interactive_args: List[str] | None = None


@dataclass
class AggregateSettings:
    """Configuration in aggregate situations, i.e. when model output is abstracted over several parameter combinations."""

    combination_ids: List[int]
    parameter: str
    parameter_values: List[Any]
    combination_data: List[Dict[str, Any]] | None

    def __init__(
        self,
        sweeps_dir: str,
        selected_sweep: str,
        combination_ids: List[int],
        parameter: str,
    ):
        """Initialise an aggregate configuration.

        Args:
            sweeps_dir (str): The path to the directory where all sweeps are stored
            selected_sweep (str): The name of the sweep of interest
            combination_ids (List[int]): Unique IDs for the selected parameter combinations
            parameter (str): The parameter of which the permutations are currently under scrutiny
        """

        self.combination_ids = combination_ids
        self.parameter = parameter

        # We want to know what possible values are they for the parameter that is being expanded
        # So then for each parameter value, we will check what the outcomes are from that combination
        run_infos = export.sweeps.get_run_infos(
            sweeps_dir, selected_sweep, hashable_safe=True
        )
        self.parameter_values = [
            str(item)
            for item in sorted(
                run_infos[run_infos["combination_id"].isin(combination_ids)][parameter]
                .unique()
                .tolist()
            )
        ]

        self.combination_data = (
            None  # by default, we do not send along combination data
        )


def get_num_constructions(
    data: Dict[str, Any] | List[Dict[str, Any]],
    is_single_run: bool = False,
    aggregate_extension: bool = False,
) -> int:
    """Get the number of constructions from a model run evolution of combination of run evolutions.

    Args:
        data (Dict[str, Any]): Run evolution or series of run evolutions.
        is_single_run (bool, optional): Whether the input is a single run. Defaults to False.
        aggregate_extension (bool, optional): Whether the input is data from an aggregate extension graph. Defaults to False.

    Returns:
        int: The number of constructions
    """

    # the additional type checks (list, dict) are to satisfy the type checker

    if not is_single_run:
        if not aggregate_extension and type(data) == dict:
            return len(data["ctx_energy_mean_mean"]["mean"][0])
        elif aggregate_extension and type(data) == list:
            return len(data[0]["ctx_energy_mean_mean"]["mean"][0])
    elif is_single_run and type(data) == dict:
        return len(data["ctx_energy_mean_mean"][0])

    return -1  # to satisfy the type checker


# These are all definitions of graphs
graph_configs: Dict[str, GraphConfig | MosaicConfig] = {
    "total_l1_mean": GraphConfig(
        reporter_name="energy_mean",
        plot_func=visualisation.energy.plot_energy,
        common_args=["x_scale_factor", "min_data", "max_data", "y_max"],
        aggregate_extension=True,
        interactive_args=["step"],
    ),
    "communicative_success": GraphConfig(
        reporter_name="communication_results_go",
        model_reporter=True,
        reporter_type=model.reporters_model.ReporterType.PERCENT,
        plot_func=visualisation.communication.plot_communication,
        common_args=["x_scale_factor", "min_data", "max_data"],
        extra_args={
            "filter_dimension": 1,
        },
        interactive_args=["step"],
        aggregate_extension=True,
    ),
    "reduction_outcomes": GraphConfig(
        reporter_name="reduction_outcomes_go",
        model_reporter=True,
        reporter_type=model.reporters_model.ReporterType.PERCENT,
        plot_func=visualisation.communication.plot_communication,
        common_args=["x_scale_factor", "min_data", "max_data"],
        extra_args={
            "filter_dimension": 1,
            "title_override": "Communication outcome when reducing, across agents",
        },
        interactive_args=["step"],
        aggregate_extension=True,
    ),
    "reentrance_outcomes": GraphConfig(
        reporter_name="reentrance_outcomes_go",
        model_reporter=True,
        reporter_type=model.reporters_model.ReporterType.PERCENT,
        plot_func=visualisation.communication.plot_communication,
        common_args=["x_scale_factor", "min_data", "max_data"],
        extra_args={
            "filter_dimension": 1,
            "title_override": "Communication outcome after re-entrance used, across agents",
        },
        interactive_args=["step"],
        aggregate_extension=True,
    ),
    "reentrance_usage": GraphConfig(
        reporter_name="reentrance_usage_go",
        model_reporter=True,
        reporter_type=model.reporters_model.ReporterType.PERCENT,
        plot_func=visualisation.communication.plot_reentrance,
        common_args=["x_scale_factor", "min_data", "max_data"],
        extra_args={
            "filter_dimension": 1,
        },
        interactive_args=["step"],
        aggregate_extension=True,
    ),
    "decision_entropy": GraphConfig(
        reporter_name="decision_entropy_go",
        model_reporter=True,
        reporter_type=model.reporters_model.ReporterType.MEDIAN,
        plot_func=visualisation.communication.plot_decision_entropy,
        common_args=["x_scale_factor", "min_data", "max_data"],
        extra_args={"num_constructions": get_num_constructions},
        interactive_args=["step"],
        aggregate_extension=True,
    ),
    "decision_entropy_reentrance": GraphConfig(
        reporter_name="decision_entropy_reentrance_go",
        model_reporter=True,
        reporter_type=model.reporters_model.ReporterType.MEDIAN,
        plot_func=visualisation.communication.plot_decision_entropy,
        common_args=["x_scale_factor", "min_data", "max_data"],
        extra_args={"num_constructions": get_num_constructions},
        interactive_args=["step"],
        aggregate_extension=True,
    ),
    "ctx_energy_mean": GraphConfig(
        reporter_name="ctx_energy_mean",
        plot_func=visualisation.energy.plot_energy_per_ctx,
        # TODO: re-introduce min_data and max_data once the graph type has been changed
        common_args=["min_data", "max_data", "y_max"],
        interactive_args=["step"],
        aggregate_extension=True,
    ),
    "communicative_confusion": GraphConfig(
        reporter_name="confusion_matrix_go",
        model_reporter=True,
        reporter_type=model.reporters_model.ReporterType.AS_IS,
        plot_func=visualisation.communication.plot_confusion,
        # TODO: re-introduce min_data and max_data once the graph type has been changed
        # common_args=["x_scale_factor"],
        extra_args={"n": 35},
        interactive_args=["step"],
        aggregate_extension=True,
    ),
    "ctx_energy_per_dim": GraphConfig(
        reporter_name="means",
        plot_func=visualisation.energy.plot_energy_per_ctx_per_dim,
        common_args=["min_data", "max_data", "y_max"],
        extra_args={"n": 5},
        interactive_args=["step"],
        aggregate_extension=False,
    ),
    "ctx_energy_deviation_per_dim": GraphConfig(
        reporter_name="means",
        reporter_type=model.reporters_agent.ReporterType.STD,
        plot_func=visualisation.energy.plot_energy_std_per_ctx_per_dim,
        common_args=["min_data", "max_data", "y_max"],
        extra_args={"n": 5},
        interactive_args=["step"],
        aggregate_extension=False,
    ),
    "comms_mosaic": MosaicConfig(
        layout=[
            ["total_l1_mean", "communicative_success", "decision_entropy"],
            [
                "reentrance_usage",
                "reduction_outcomes",
                "decision_entropy_reentrance",
                "reentrance_outcomes",
            ],
        ],
        size=(24, 12),
        aggregate_extension=True,
    ),
    "energy_mosaic": MosaicConfig(
        layout=[
            ["ctx_energy_mean", "ctx_energy_per_dim"],
            ["ctx_energy_deviation_per_dim"],
        ],
        size=(12, 12),
    ),
    "confusion_mosaic": MosaicConfig(
        layout=[
            ["communicative_confusion", "communicative_confusion"],
            ["communicative_confusion", "communicative_confusion"],
        ],
        size=(12, 12),
        extra_args=[[{"step": 0.25}, {"step": 0.5}], [{"step": 0.75}, {"step": -1}]],
    ),
    "plot_energy_per_ctx_per_dim_norm": GraphConfig(
        reporter_name=["means", "sigmas"],
        plot_func=visualisation.energy.plot_energy_per_ctx_per_dim_norm,
        common_args=["y_max", "y_min"],
        extra_args={"n": 5},
        interactive_args=["step"],
        aggregate_extension=False,
        context=GraphContext.DASHBOARD,
    ),
    "vector_difference": GraphConfig(
        reporter_name="vector_differences_go",
        plot_func=visualisation.energy.plot_energy_differences,
        common_args=["x_scale_factor", "min_data", "max_data"],
        aggregate_extension=True,
        interactive_args=["step"],
        context=GraphContext.DASHBOARD,
        extra_args={
            "plot_mean": True,
        },
    ),
}


def get_graph_names(context: int, is_single_run: bool = False) -> List[str]:
    """Returns a list of the names of all available graphs

    Args:
        context (int): Context where the graphs will be used
        is_single_run (bool): Whether the graphs are meant for a single run display

    Returns:
        List[str]: A list of the names of all available graphs
    """

    return [
        graph_config
        for graph_config in list(graph_configs.keys())
        if graph_configs[graph_config].context == context
        and not graph_configs[graph_config].aggregate
        and (not is_single_run or graph_configs[graph_config].single_run_sensible)
    ]


def get_aggregate_graph_names(context: int) -> List[str]:
    """Returns a list of the names of all available aggregate graphs

    Args:
        context (int): Context where the graphs will be used

    Returns:
        List[str]: A list of the names of all available graphs
    """

    return [
        graph_config
        for graph_config in list(graph_configs.keys())
        if graph_configs[graph_config].context == context
        and (
            graph_configs[graph_config].aggregate
            or graph_configs[graph_config].aggregate_extension
        )
    ]


def get_graph_config(graph_name: str) -> Union[GraphConfig, MosaicConfig]:
    """Retrieve the configuration for a graph or mosaic graph

    Args:
        graph_name (str): Name of the graph

    Raises:
        ValueError: Raised if name of the graph does not reference an existing config

    Returns:
        Union[GraphConfig, MosaicConfig]: Configuration associated with the specified graph name
    """

    # First, retrieve the config for this graph (see above)
    if not graph_name in graph_configs:
        raise ValueError(f"'{graph_name}' is not a valid graph")

    return graph_configs[graph_name]


def generate_graphs(
    sweeps_dir: str,
    selected_sweep: str,
    combination_ids: Union[int, List[int]],
    graphs: List[str],
    aggregate: Optional[AggregateSettings] = None,
    single_run: Optional[int] = None,
    selected_step: int | None = None,
    disable_title=False,
) -> Dict[str, matplotlib.figure.Figure]:
    """Generate the specified graphs depending on the given sweep

    Args:
        sweeps_dir (str): Path to the directory where all sweeps are stored
        selected_sweep (str): Name of the sweep of interest
        combination_id (int): ID of the unique parameter combination
        graphs (List[str]): List of names of the graphs to be generated
        aggregate (AggregateSettings, optional): Configuration for aggregate graphs. Defaults to None.
        single_run (int, optional): ID of the single run to generate a graph for. Defaults to None.
        selected_step (int). Number of the step being inspected. Defaults to None for the default step.
        disable_title (bool, optional): Whether to show a title for this graph. Defaults to False.

    Raises:
        ValueError: Raised if a supplied graph name does not have an associated graph
        ValueError: Raised if multiple combination IDs appear without an aggregate configuration

    Returns:
        Dict[str, matplotlib.figure.Figure]: Dictionary with graph names as keys and generated graphs as values
    """

    scale_factor: int = int(
        export.sweeps.get_sweep_info(sweeps_dir, selected_sweep)[
            "datacollector_step_size"
        ]
    )
    # Find ymax by looking at the run infos
    if isinstance(combination_ids, int):
        _combination_ids = [combination_ids]
    else:
        _combination_ids = combination_ids
    # Filter for the required combination ids
    run_infos = export.sweeps.get_run_infos(sweeps_dir, selected_sweep)
    run_infos = run_infos[run_infos["combination_id"].isin(_combination_ids)]
    y_max = np.max(np.array(run_infos["vector_bounds"].tolist())[:, 1])
    y_min = 1  # TODO ??

    data: Union[dict[str, Any], pd.DataFrame]
    # If only a single combination_id is given, this is a single graph
    if isinstance(combination_ids, int) and aggregate is None and single_run is None:
        # Retrieve the data for the single combination
        combination_id = combination_ids
        data = export.combinations.get_combination_data(
            sweeps_dir, selected_sweep, combination_id
        )
    elif (
        isinstance(combination_ids, int)
        and aggregate is None
        and single_run is not None
    ):
        data = export.runs.get_run_data(sweeps_dir, selected_sweep, single_run)
    elif isinstance(combination_ids, list) and aggregate is not None:
        # Get the combination infos dataframe
        combination_infos = export.sweeps.get_combination_infos(
            sweeps_dir, selected_sweep
        )
        # Filter for the required combinations
        data = combination_infos[
            combination_infos["combination_id"].isin(combination_ids)
        ]

        needs_combination_data = False
        # Now, check if we need combination data
        # With this I meant the data that is needed to overlay multiple regular graphs
        # over each other in an aggregate context
        for graph in graphs:
            print(graph)
            graph_config = graph_configs[graph]
            if graph_config.aggregate_extension:
                needs_combination_data = True

        if needs_combination_data:
            # Get the combination data for each combination_id that is involved in this aggregate
            combination_data: List[Dict[str, Any]] = []
            for combination_id in aggregate.combination_ids:
                combination_data_single = export.combinations.get_combination_data(
                    sweeps_dir, selected_sweep, combination_id
                )
                combination_data.append(combination_data_single)
            # Attach to aggregate settings
            aggregate.combination_data = combination_data
    else:
        raise ValueError(
            "Unrecognised combination of combination IDs and aggregate settings"
        )

    return generate_graphs_inner(
        data,
        graphs,
        aggregate,
        single_run,
        selected_step,
        scale_factor,
        y_max=y_max,
        y_min=y_min,
    )


def generate_graphs_inner(
    data: Union[dict[str, Any], pd.DataFrame],
    graphs: List[str],
    aggregate: Optional[AggregateSettings] = None,
    single_run: Optional[int] = None,
    selected_step: int | None = None,
    scale_factor: int = 1,
    y_max: int = 100,
    y_min: int = 0,
) -> Dict[str, matplotlib.figure.Figure]:

    # Now, we can build the desired graphs and save them
    graphs_output = {}

    # Scale selected step by scale factor
    if selected_step is not None:
        selected_step = selected_step // scale_factor

    # We go over all requested graphs and generate them
    for graph_name in graphs:
        config = get_graph_config(graph_name)

        # Check if mosaic plot
        if isinstance(config, MosaicConfig):
            # One by one, we replace the names of the graphs with the actual functions that build them
            plot_functions = []
            for row_index, row in enumerate(config.layout):
                inner_functions = []
                for column_index, references_graph_name in enumerate(row):
                    # Skip graphs that do not make sense in single run view
                    if (
                        single_run is not None
                        and not get_graph_config(
                            references_graph_name
                        ).single_run_sensible
                    ):
                        continue

                    parent_extra_args = None
                    if config.extra_args is not None:
                        parent_extra_args = config.extra_args[row_index][column_index]

                    graph_function = generate_inner_lambda(
                        data,
                        references_graph_name,
                        scale_factor=scale_factor,
                        aggregate_config=aggregate,
                        single_run=single_run,
                        selected_step=selected_step,
                        parent_extra_args=parent_extra_args,
                        y_max=y_max,
                        y_min=y_min,
                    )
                    inner_functions.append(graph_function)

                # Because we filter graphs, it can be that the row is empty
                # So check first
                if len(inner_functions) > 0:
                    plot_functions.append(inner_functions)

            # Make the plot based on the functions
            figure = visualisation.multiplot.combine(plot_functions, config.size)
        else:
            # Make a single plot. We pass ax=None because there is no existing axis to hook into
            figure, ax = generate_inner_lambda(
                data,
                graph_name,
                scale_factor=scale_factor,
                aggregate_config=aggregate,
                single_run=single_run,
                selected_step=selected_step,
                y_max=y_max,
                y_min=y_min,
            )(ax=None)

        graphs_output[graph_name] = figure

    return graphs_output


def generate_inner_lambda(
    data: Union[Dict[str, Any], pd.DataFrame],
    graph_name: str,
    scale_factor: int = 1,
    y_max: int = 100,
    y_min: int = 0,
    single_run: Optional[int] = None,
    selected_step: int | None = None,
    aggregate_config: Optional[AggregateSettings] = None,
    parent_extra_args: Dict[str, Any] | None = None,
) -> Callable:
    """Generate the function which builds the graph specified by the graph name

    Args:
        data (Union[Dict[str, Any], pd.DataFrame]): Data dump of a specific parameter combination, or combinations
        graph_name (str): Name of the graph to generate the function for
        single_run (int, optional): ID of the single run to plot. Defaults to None.
        selected_step (int). Number of the step being inspected. Defaults to None for the default step.
        aggregate_config (AggregateSettings, optional): Configuration for aggregate graphs. Defaults to None.
        extra_args (Dict[str, Any] | None): Extra arguments supplied by the parent mosaic. Defaults to None.

    Raises:
        TypeError: Raised if the graph name is associated with a mosaic function

    Returns:
        Callable: Function which generates the graph specified by the graph name
    """

    config = get_graph_config(graph_name)

    if isinstance(config, MosaicConfig):
        raise TypeError("Inner plot function cannot be of mosaic type")

    # Check if there are other arguments to be supplied, based on data argument
    kwargs = {}
    extra_args = {}
    if config.extra_args:
        if parent_extra_args is not None:
            extra_args = {**config.extra_args, **parent_extra_args}
        elif config.extra_args is not None:
            extra_args = copy.deepcopy(config.extra_args)

    if config.interactive_args is not None:
        # Overwrite default arguments with interactive arguments
        for interactive_arg_name in config.interactive_args:
            if interactive_arg_name == "step" and selected_step is not None:
                extra_args["step"] = selected_step

    if len(extra_args) > 0:
        for arg_name, arg_func in extra_args.items():
            # extra_arg is a lambda function
            if isinstance(arg_func, Callable):
                # Data source changes depending on whether this is an aggregate extension graph
                # or just a regular extension graph
                arg_func_args: List[Any] = [data]
                arg_func_kwargs: Dict[str, Any] = {}

                if aggregate_config is not None and config.aggregate_extension:
                    if aggregate_config.combination_data is None:
                        raise ValueError(
                            "Cannot apply argument function Callable if combination data is None"
                        )

                    arg_func_args = [aggregate_config.combination_data]

                if single_run is not None:
                    arg_func_kwargs["is_single_run"] = True

                if config.aggregate_extension and aggregate_config is not None:
                    arg_func_kwargs["aggregate_extension"] = True

                kwargs[arg_name] = arg_func(*arg_func_args, **arg_func_kwargs)
            # extra_arg is a constant
            else:
                kwargs[arg_name] = arg_func

    # If aggregate config is None, this is always a simple graph
    # If this is an aggregate extension graph, this is also a simple graph
    # DESPITE the aggregate configuration being defined
    is_regular_graph = aggregate_config is None or config.aggregate_extension

    if is_regular_graph:
        # You cannot have both innovators/conservators and an aggregate extension
        if config.aggregate_extension and len(config.data_columns) > 1:
            raise ValueError(
                "Cannot build aggregate extension graph if innovators_share > 0. This is an architectural decision, and no mistake on your behalf."
            )

        central_data = []

        # Since the aggregate extension graphs complicate things even further,
        # allow me to explain ...

        # Either there is a single data source (one combination ID), and then there can be
        # multiple data columns (innovator, conservator)
        # Or, there are multiple data sources (multiple combination IDs)
        # then there can only be ONE data column
        # So I'm adding one more layer of abstraction where we loop over data sources
        # so then I can switch out the data sources in case fo an aggregate extension graph
        data_sources: Sequence[Union[Dict[str, Any], pd.DataFrame]] = []
        aggregate_extension_x: List[str] | None = (
            None  # x values for aggregate extension graph
        )
        if not config.aggregate_extension or (
            config.aggregate_extension and aggregate_config is None
        ):
            data_sources = [data]
        elif config.aggregate_extension and aggregate_config is not None:
            if aggregate_config.combination_data is None:
                raise ValueError(
                    "Combination data stored in aggregate config cannot be None"
                )

            data_sources = aggregate_config.combination_data
            aggregate_extension_x = aggregate_config.parameter_values
        else:
            raise ValueError("Invalid aggregate config argument")

        min_data: List[List[float]] = []
        max_data: List[List[float]] = []

        for data in data_sources:
            for data_column in config.data_columns:
                for common_arg in config.common_args:
                    value = None
                    if common_arg == "x_scale_factor":
                        value = scale_factor
                    elif common_arg == "y_max":
                        value = y_max
                    elif common_arg == "y_min":
                        value = y_min
                    elif common_arg == "min_data" and single_run is None:
                        min_data.append(data[data_column]["q1"])
                    elif common_arg == "max_data" and single_run is None:
                        max_data.append(data[data_column]["q3"])

                    kwargs[common_arg] = value

                # Combination graph
                if single_run is None:
                    central_data.append(data[data_column][config.action_column])
                else:
                    # No need for aggregation
                    central_data.append(data[data_column])

        if len(min_data) > 0:
            kwargs["min_data"] = min_data
        if len(max_data) > 0:
            kwargs["max_data"] = max_data
        if aggregate_extension_x is not None:
            kwargs["aggregate_extension_x"] = aggregate_extension_x

        kwargs["attributes"] = config.data_columns

        # Make the plot function
        return lambda ax: config.plot_func(central_data, **kwargs, ax=ax)
    # Aggregate graph
    else:
        # To satisfy the type checker
        if aggregate_config is None:
            raise ValueError(
                "Aggregate config cannot be None when an aggregate graph is requested"
            )

        data_column = config.data_columns[
            0
        ]  # temporary workaround for aggregate graphs
        for common_arg in config.common_args:
            value = None
            if common_arg == "min_data":
                value = data[
                    batch.aggregate.make_aggregate_output_name(
                        data_column, config.action_column_inner, "q1"
                    )
                ]
            elif common_arg == "max_data":
                value = kwargs["max_data"] = data[
                    batch.aggregate.make_aggregate_output_name(
                        data_column, config.action_column_inner, "q3"
                    )
                ]
            kwargs[common_arg] = value

        kwargs["attributes"] = data_column

        return lambda ax: config.plot_func(
            [
                data[
                    batch.aggregate.make_aggregate_output_name(
                        data_column, config.action_column_inner, config.action_column
                    )
                ].tolist()
            ],  # I "temporarily" wrap this in brackets until I fix the dimensionality issue
            aggregate_config.parameter_values,
            parameter=aggregate_config.parameter,
            **kwargs,
            ax=ax,
        )
