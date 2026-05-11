import pandas as pd

from typing import Dict, Tuple, List, Union

# Parameters used by the application. These are not parameters
RESERVED_KEYWORDS = ["sweep", "filter", "aggregate", "run"]


def build_mapping(
    run_infos: pd.DataFrame,
) -> Tuple[Dict[str, List[str]], Dict[str, str]]:
    """Build a mapping from a run infos dataframe. Teases out constants and parameters.

    Args:
        run_infos (pd.DataFrame): Run infos dataframe, containing info about all runs in a sweep. The mappings will be based on this dataframe.

    Returns:
        Tuple[Dict[str, List[str]], Dict[str, str]]: Tuple (parameter_mapping, constants_mapping)
    """

    parameter_mapping = {}
    constants_mapping = {}

    for column in run_infos:
        # Do not process housekeeping column names
        if column in ["run_id", "combination_id", "seed", "iteration"]:
            continue

        unique_values = run_infos[column].unique().tolist()
        unique_values.sort()
        # If there is only one unique value, there is nothing to set
        # This means for this sweep, this parameter is a constant
        if len(unique_values) == 1:
            constants_mapping[column] = str(unique_values[0])
        # Else, list all unique values as parameter options
        else:
            parameter_mapping[column] = [str(value) for value in unique_values]

    return parameter_mapping, constants_mapping


def remove_aggregate_parameter_from_selected(
    aggregate_parameter: str, selected_parameters: Dict[str, str]
) -> Dict[str, str]:
    """If an aggregate parameter is set and it appears as a parameter in the parameter mapping, remove it (since a value cannot be chosen for this parameter)

    Args:
        aggregate_parameter (str): The chosen aggregate parameter
        selected_parameters (Dict[str, List[str]]): The current parameter mapping

    Returns:
        Dict[str, List[str]]: The modified parameter mapping, without the aggregate parameter
    """

    if aggregate_parameter in selected_parameters:
        del selected_parameters[aggregate_parameter]

    return selected_parameters


def find_eligible_runs(
    run_infos: pd.DataFrame,
    selected_parameters: Dict[str, str],
) -> pd.DataFrame:
    """Find eligible runs from the selected parameters

    Args:
        run_infos (pd.DataFrame): Run infos dataframe, containing info about all runs in a sweep
        selected_parameters (Dict[str, str]): The selected parameters, which will be used to filter all available runs in the sweep.

    Returns:
        pd.DataFrame: Dataframe containing info about all model runs corresponding to the selected parameters
    """

    # Create a mask to select the right model
    mask = pd.Series(True, index=run_infos.index)

    # Add the selected parameter to the mask
    for column, value in selected_parameters.items():
        if column in RESERVED_KEYWORDS:
            continue
        mask &= run_infos[column].astype(str) == str(value)

    # Filter the data frame
    selected_runs = run_infos[mask]

    return selected_runs
