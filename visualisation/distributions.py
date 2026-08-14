from typing import List, Tuple

import numpy as np
import matplotlib.axes
import matplotlib.figure
import matplotlib.pyplot as plt
import model.sampling
import visualisation.core
import visualisation.distributions


def visualise_priors(
    priors: np.ndarray,
    ax: matplotlib.axes.Axes | None = None,
    title: str | None = None,
    legend_title: str | None = None,
    legend_values: List[str] | None = None,
    disable_title: bool = False,
) -> Tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
    rng = np.random.default_rng(123456)

    fig, ax = visualisation.core.check_ax(ax, disable_title)

    if len(priors.shape) != 2:
        raise ValueError("Input matrix dimensions should be 2D")

    num_priors_list = priors.shape[0]
    num_constructions = priors.shape[1]

    for index in range(num_priors_list):
        priors_i = priors[index, :]

        label = None
        if legend_values is not None:
            label = legend_values[index]

        ax.plot(
            list(range(0, num_constructions)),
            priors_i,
            color=visualisation.core.COLOURS[index],
            label=label,
        )
        ax.set_ylim(0, 0.35)

    if title is not None and not disable_title:
        ax.set_title(title)

    if legend_title is not None:
        ax.legend().set_title(legend_title)

    ax.set_ylabel("Probability")
    ax.set_xlabel("Rank")

    visualisation.core.set_y_axis_percent(ax)

    output_fig = visualisation.core.get_ax_figure(ax)
    plt.close(output_fig)

    return (output_fig, ax)


def plot_priors_graph(
    zipf_params: List[float],
    title: str,
    legend_title: str,
    legend_values: List[str],
    disable_title: bool = False,
):
    rng = np.random.default_rng(123456)

    priors_matrix = np.array(
        [
            model.sampling.ZipfianSampling(zipf_param=zipf_param).get_priors(rng)[1]
            for zipf_param in zipf_params
        ]
    )

    return visualisation.distributions.visualise_priors(
        priors_matrix,
        title=title,
        legend_title=legend_title,
        legend_values=legend_values,
        disable_title=disable_title,
    )[0]
