import matplotlib.axes
import matplotlib.figure


import model.model
import visualisation.core

from typing import Optional, List, Union, Tuple, Any


def plot_energy(
    data: model.model.ReductionModel | List[float],
    attributes: str | List[str],
    y_max: int,
    **kwargs: Any,
) -> Tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
    """Plot the mean entropy across agents

    Args:
        model.model.ReductionModel | List[float]: Either a model instance or a list of values.
        attributes (str | List[str]): The column to fetch data from. Always supply, even if input data is not a model, so dimensionality of the data can be assessed.
        y_max (int): The maximum y value
        **kwargs: Additional keyword arguments passed to parent plotting function.

    Returns:
        Tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]: The finished graph
    """

    return visualisation.core.plot_value(
        data,
        attributes,
        ylim=[0, y_max],
        title=f"Mean construction energy across agents",
        **kwargs,
    )


def plot_energy_per_ctx(
    data: model.model.ReductionModel | List[float],
    attributes: str,
    y_max: int,
    step: int = -1,
    **kwargs: Any,
) -> Tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
    """Plot the energy for each construction of a given parameter combination

    Args:
        data (List[float]): Energy per construction. Wrapped in a list because you never know
        attributes (str): The column to fetch data from. Always supply, even if input data is not a model, so dimensionality of the data can be assessed.
        step (int): Step to get the data from. On the scale of the datacollector.
        attribute (str): The name of the series to model.

    Returns:
        Tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]: The finished graph
    """

    return visualisation.core.plot_error_bar(
        data,
        attributes,
        ylim=[0, y_max],
        title=f"Average L1 value in the base model (per construction, across agents",
        y_label="Energy",
        **kwargs,
    )
