import os
import json
import pandas as pd
import numpy as np

import export.sweeps

from typing import Dict, Any


def make_run_data_path(
    sweeps_dir: str, selected_sweep: str, run_id: int, create: bool = False
) -> str:
    """Make the path for where run data JSON is stored

    Args:
        sweeps_dir (str): The path to the directory where all sweeps are stored
        selected_sweep (str): The name of the sweep of interest
        run_id (int): ID of the run of interest
        create (bool): Whether to create any missing directories

    Returns:
        str: Path where run data is stored
    """

    sweep_dir = export.sweeps.make_selected_sweep_dir(
        sweeps_dir, selected_sweep, create
    )
    return os.path.join(sweep_dir, f"{run_id}.json")


def get_run_data(sweeps_dir: str, selected_sweep: str, run_id: int) -> Dict[str, Any]:
    """Load a data dump for a specific combination

    Args:
        sweeps_dir (str): The path to the directory where all sweeps are stored
        selected_sweep (str): The name of the sweep of interest
        run_id (int): Unique ID for the run of interest

    Returns:
        Dict[str, Any]: Unserialised data dump of the specified run
    """

    run_data_path = make_run_data_path(sweeps_dir, selected_sweep, run_id)

    with open(run_data_path, "rt") as reader:
        data = json.loads(reader.read())

    return data


def load_dataframe(sweeps_dir: str, selected_sweep: str, run_id: int) -> Dict[str, Any]:
    """Load a data dump for a specific run

    Args:
        sweeps_dir (str): The path to the directory where all sweeps are stored
        selected_sweep (str): The name of the sweep of interest
        run_id (int): interID of the run of interest

    Returns:
        Dict[str, Any]: Unserialised data dump of the specified run
    """

    sweep_dir = export.sweeps.make_selected_sweep_dir(sweeps_dir, selected_sweep)
    run_data_path = os.path.join(sweep_dir, f"{run_id}.json")

    # Load the selected simulation run from disk
    with open(run_data_path, "rt") as run_file:
        # Load the dataframe-as-json from disk
        # There is no need to really turn it into a dataframe, downstream will figure it out
        json_object = json.loads(run_file.read())

        return json_object
