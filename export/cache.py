import os
import export.files

from typing import List, Union, Optional


def get_combination_id(selected_run_ids: List[int]) -> int:
    """Turn a list of selected runs into a unique ID identifying those runs

    Args:
        selected_run_ids (List[int]): List of selected run IDs

    Returns:
        int: The unique ID for the specified combination of runs
    """

    return sum(selected_run_ids)


def get_cache_combination_id(combination_ids: Union[int, List[int]]) -> int:
    """Turn combination of combination IDs into a cache ID

    Args:
        combination_ids (Union[int, List[int]]): List of combination IDs, or a single one

    Returns:
        int: _description_
    """

    if isinstance(combination_ids, list):
        cache_combination_id = get_combination_id(combination_ids)
    elif isinstance(combination_ids, int):
        cache_combination_id = combination_ids

    return cache_combination_id


def make_temp_sweep_figures_dir(selected_sweep: str, figures_output_dir: str) -> str:
    """Create the directory where figures for this sweep will be stored temporarily.

    Args:
        selected_sweep (str): Name of the selected sweep
        figures_output_dir (str): Path to where the figures are stored

    Returns:
        str: Path to where figures will be stored for the selected sweep
    """

    # This is where we will store the graphs output

    # We create a directory for the selected sweep
    temp_sweep_figures_dir = os.path.join(figures_output_dir, selected_sweep)

    if not os.path.exists(temp_sweep_figures_dir):
        os.makedirs(temp_sweep_figures_dir, exist_ok=True)

    return temp_sweep_figures_dir


def make_temp_runs_figures_dir(
    selected_sweep: str,
    combination_id: int,
    figures_output_dir: str,
    single_run_id: Optional[int] = None,
) -> str:
    """Create the directory where figures for this sweep and parameter combination will be stored temporarily.

    Args:
        selected_sweep (str): Name of the selected sweep
        combination_id (int): Unique ID for the parameter selection
        figures_output_dir (str): Path to where figures are stored
        single_run_id (Optional[int], optional): ID of the single run, if applicable. Defaults to None.

    Returns:
        str: Path to where figures will be stored for the selected sweep and parameter combination
    """

    temp_run_figures_dir = make_temp_sweep_figures_dir(
        selected_sweep, figures_output_dir
    )

    # We create a directory for the selected parameter selection
    temp_models_figures_dir = os.path.join(temp_run_figures_dir, str(combination_id))

    if not os.path.exists(temp_models_figures_dir):
        os.makedirs(temp_models_figures_dir, exist_ok=True)

    # Create nested directory for a singular run too
    if single_run_id is not None:
        temp_models_figures_dir = os.path.join(
            temp_models_figures_dir, str(single_run_id)
        )

        if not os.path.exists(temp_models_figures_dir):
            os.makedirs(temp_models_figures_dir, exist_ok=True)

    return temp_models_figures_dir


def is_graph_in_cache(
    selected_sweep: str,
    combination_id: int,
    graph_name: str,
    profile_name: str,
    figures_output_dir: str,
    single_run_id: Optional[int] = None,
) -> bool:
    """Check whether the specified graph was already created for this sweep and parameter selection

    Args:
        selected_sweep (str): Name of the selected sweep
        combination_id (int): Unique ID for the parameter selection
        graph_name (str): Name of the graph to check
        single_run_id (int, optional). Unique ID when singling out a single run. Defaults to None.

    Returns:
        bool: Whether the specified graph is cached
    """

    temp_models_figures_dir = make_temp_runs_figures_dir(
        selected_sweep, combination_id, figures_output_dir, single_run_id=single_run_id
    )
    graph_filename = export.files.get_figure_filename(profile_name, graph_name)

    # Where graph would typically be saved
    graph_path = os.path.join(temp_models_figures_dir, graph_filename)

    # If it exists, it sits in cache
    return os.path.exists(graph_path)


def get_cached_graphs(
    selected_sweep: str,
    combination_id: int,
    graphs: List[str],
    profile_name: str,
    figures_output_dir: str,
    single_run_id: Optional[int] = None,
) -> List[str]:
    """Retrieve a list of all cached graphs for the selected sweep with the specified parameter selection ID

    Args:
        selected_sweep (str): Name of the selected sweep
        combination_id (int): Unique ID for the parameter selection
        graphs (List[str]): List with names of graphs to check whether in cache
        profile_name (str): Name of the selected profile
        figures_output_dir (str): Path where figures are written
        single_run_id (int, optional). Unique ID when singling out a single run. Defaults to None.

    Returns:
        List[str]: List of names with all cached graphs
    """

    # Check if the graphs we need already exist
    cached_graphs = []

    for graph_name in graphs:
        if is_graph_in_cache(
            selected_sweep,
            combination_id,
            graph_name,
            profile_name,
            figures_output_dir,
            single_run_id=single_run_id,
        ):
            cached_graphs.append(graph_name)

    return cached_graphs
