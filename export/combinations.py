import json
import os.path
import export.sweeps

from typing import Dict, Any


def make_combination_data_path(
    sweeps_dir: str, selected_sweep: str, combination_id: int
) -> str:
    """Make the path for where combnation data JSON is stored

    Args:
        sweeps_dir (str): The path to the directory where all sweeps are stored
        selected_sweep (str): The name of the sweep of interest
        combination_id (int): Unique ID for the parameter selection

    Returns:
        str: Path where combination data is stored
    """

    sweep_dir = export.sweeps.make_selected_sweep_dir(sweeps_dir, selected_sweep)
    return os.path.join(sweep_dir, f"combination_{combination_id}.json")


def get_combination_data(
    sweeps_dir: str, selected_sweep: str, combination_id: int
) -> Dict[str, Any]:
    """Load a data dump for a specific combination

    Args:
        sweeps_dir (str): The path to the directory where all sweeps are stored
        selected_sweep (str): The name of the sweep of interest
        combination_id (int): Unique ID for the parameter selection

    Returns:
        Dict[str, Any]: Unserialised data dump of the specified combination
    """

    combination_data_path = make_combination_data_path(
        sweeps_dir, selected_sweep, combination_id
    )

    with open(combination_data_path, "rt") as reader:
        data = json.loads(reader.read())

    return data
