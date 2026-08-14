import matplotlib.pyplot as plt
import matplotlib.figure
from matplotlib.axes import Axes

from typing import Callable, Tuple, List


def combine(
    plot_functions: List[List[Callable]],
    figsize: Tuple[float, float] = (16, 6),
) -> matplotlib.figure.Figure:
    # (rows -> columns)

    # Outer length = number of rows
    num_rows = len(plot_functions)

    if num_rows == 0:
        raise ValueError("Number of rows or columns cannot be zero")

    # Max inner length = number of columns
    num_cols = max(len(inner) for inner in plot_functions) if plot_functions else 0

    if num_rows == 0 or num_cols == 0:
        raise ValueError("Number of rows or columns cannot be zero")

    # Create subplots grid
    fig, axes = plt.subplots(num_rows, num_cols, figsize=figsize)

    # If only one row and one column, axes is not an array, so we need to make it one
    if num_rows == 1 and num_cols == 1:
        axes = [[axes]]
    elif num_rows == 1:
        axes = [axes]
    elif num_cols == 1:
        axes = [[ax] for ax in axes]

    for i in range(num_rows):
        for j in range(num_cols):
            # Skip if beyond row column length
            if j >= len(plot_functions[i]):
                continue

            plot_functions[i][j](axes[i][j])

    plt.tight_layout()

    return fig
