import os
import matplotlib.figure

from typing import Any


def export_files(
    graphs: dict[str, matplotlib.figure.Figure], profile_name: str, export_folder: str
):
    """Export graphs as images in the specified directory

    Args:
        graphs (dict[str, matplotlib.figure.Figure]): Dict which holds all figures. Figure names as keys, figure objects as values
        profile_name (str): Name of the associated profile. Most relevant when batch exporting graphs
        export_folder (str): Path to the directory where images will be stored
    """

    for graph_name in graphs:
        filename = get_figure_filename(profile_name, graph_name)
        print(f"Writing {filename}")

        file_path = os.path.join(export_folder, filename)
        graphs[graph_name].savefig(file_path)


def get_figure_filename(profile_name: str, graph_name: str) -> str:
    """Compose the fill filename of a figure, depending on the current profile and the graph name

    Args:
        profile_name (str): Name of the associated profile
        graph_name (str): Name of the figure

    Returns:
        str: Full filename of the figure
    """

    return "".join(["fig-", profile_name, "-", graph_name, ".png"])
