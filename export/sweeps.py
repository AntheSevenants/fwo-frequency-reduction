import os
import json
import pandas as pd

from typing import List, Union


def get_sweeps(sweeps_dir: str) -> List[str]:
    """Returns a list of directory names associated with different model sweeps

    Args:
        sweeps_dir (str): The path to the directory where all sweeps are stored

    Returns:
        List[str]: List of directory names associated with model sweeps
    """

    sweep_dirs = next(os.walk(sweeps_dir))[1]
    sweep_dirs = sorted(sweep_dirs)

    return sweep_dirs


def make_sweep_info_path(sweeps_dir: str, selected_sweep: str) -> str:
    selected_sweep_dir = make_selected_sweep_dir(sweeps_dir, selected_sweep)
    return os.path.join(selected_sweep_dir, "sweep_info.json")


def get_sweep_info(
    sweep_dir: str, selected_sweep: str
) -> dict[str, Union[float, int, str]]:
    sweep_info_path = make_sweep_info_path(sweep_dir, selected_sweep)
    if not os.path.exists(sweep_info_path):
        raise FileNotFoundError("Sweep info JSON does not exist")

    with open(sweep_info_path, "rt") as reader:
        return json.loads(reader.read())


def get_run_infos(
    sweeps_dir: str, selected_sweep: str, hashable_safe=False
) -> pd.DataFrame:
    """Returns a dataframe containing the run info associated with different sweeps

    Args:
        sweeps_dir (str): The path to the directory where all sweeps are stored
        selected_sweep (str): The name of the sweep of interest
        hashable_safe (bool): Whether to stringify non-hashable types. Defaults to False.

    Raises:
        FileNotFoundError: Raised if the run infos file cannot be found in the sweep directory

    Returns:
        pd.DataFrame: Dataframe containing the run info associated with different sweeps
    """

    # I am not a French speaker, I just like using the word "infos" because it is goofy
    model_infos_path = make_run_infos_path(sweeps_dir, selected_sweep)
    if not os.path.exists(model_infos_path):
        raise FileNotFoundError("Run infos CSV does nost exist")

    run_infos = pd.read_json(model_infos_path)
    if hashable_safe:
        for column in run_infos.columns:
            run_infos[column] = run_infos[column].apply(
                lambda x: str(x) if isinstance(x, (list, dict, set)) else x
            )

    return run_infos


def make_run_infos_path(sweeps_dir: str, selected_sweep: str) -> str:
    """Make the path for where run infos CSV is stored

    Args:
        sweeps_dir (str): The path to the directory where all sweeps are stored
        selected_sweep (str): The name of the sweep fo interest

    Returns:
        str: Path where run infos CSV is stored
    """

    selected_sweep_dir = make_selected_sweep_dir(sweeps_dir, selected_sweep)
    return os.path.join(selected_sweep_dir, "run_infos.json")


def get_combination_infos(sweeps_dir: str, selected_sweep: str) -> pd.DataFrame:
    """Returns a dataframe containing high-order info associated with different parameter combinations

    Args:
        sweeps_dir (str): The path to the directory where all sweeps are stored
        selected_sweep (str): The name of the sweep of interest

    Raises:
        FileNotFoundError: Raised if the combination infos file cannot be found in the sweep directory

    Returns:
        pd.DataFrame: Dataframe containing the combination info associated with different parameter combinatinos
    """

    combination_infos_path = make_combination_infos_path(sweeps_dir, selected_sweep)
    if not os.path.exists(combination_infos_path):
        raise FileNotFoundError("Combination infos JSON does nost exist")

    return pd.read_json(combination_infos_path, orient="records")


def make_combination_infos_path(sweeps_dir: str, selected_sweep: str) -> str:
    """Make the path for where combination infos JSON is stored

    Args:
        sweeps_dir (str): The path to the directory where all sweeps are stored
        selected_sweep (str): The name of the sweep fo interest

    Returns:
        str: Path where combination infos JSON is stored
    """

    selected_sweep_dir = make_selected_sweep_dir(sweeps_dir, selected_sweep)
    return os.path.join(selected_sweep_dir, "combination_infos.json")


def make_selected_sweep_dir(
    sweeps_dir: str, selected_sweep: str, create: bool = False
) -> str:
    """Make the path for the directory of the sweep of interest in the specified sweeps directory

    Args:
        sweeps_dir (str): The path to the directory where all sweeps are stored
        selected_sweep (str): The name of the sweep of interest
        create (bool): Whether to create the directory

    Raises:
        FileNotFoundError: Raised if the selected sweep cannot be found in the sweep directory

    Returns:
        str: Path to directory for sweep of interest
    """

    selected_sweep_dir = os.path.join(sweeps_dir, selected_sweep)
    if not os.path.exists(selected_sweep_dir):
        if create:
            os.makedirs(selected_sweep_dir, exist_ok=True)
        else:
            raise FileNotFoundError("Sweep directory does not exist")

    return selected_sweep_dir
