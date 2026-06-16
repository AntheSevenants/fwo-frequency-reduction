import matplotlib.axes
import matplotlib.figure
import numpy as np

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
        step=step,
        **kwargs,
    )


def plot_energy_per_ctx_per_dim(
    data: model.model.ReductionModel | List[List[float]],
    attributes: str,
    y_max: int,
    n: int,
    num_dims: int = 10,
    **kwargs: Any,
) -> Tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
    """Plot the energy for each construction of a given parameter combination, but show the dimensionality too

    Args:
        data (List[float]): Energy per construction. Wrapped in a list because you never know
        attributes (str): The column to fetch data from. Always supply, even if input data is not a model, so dimensionality of the data can be assessed.
        step (int): Step to get the data from. On the scale of the datacollector.
        attribute (str): The name of the series to model.

    Returns:
        Tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]: The finished graph
    """

    return visualisation.core.plot_error_bar_horizontal(
        data,
        attributes,
        [str(i) for i in range(0, n)],
        x=[str(i) for i in range(0, num_dims)],
        ylim=[0, y_max],
        title=f"L1 values in the base model (per construction, across agents)",
        y_label="Energy",
        n=n,
        **kwargs,
    )


def plot_energy_std_per_ctx_per_dim(
    data: model.model.ReductionModel | List[List[float]],
    attributes: str,
    y_max: int,
    n: int,
    num_dims: int = 10,
    **kwargs: Any,
) -> Tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
    """Plot the energy standard deviation for each construction of a given parameter combination, but show the dimensionality too

    Args:
        data (List[float]): Energy per construction. Wrapped in a list because you never know
        attributes (str): The column to fetch data from. Always supply, even if input data is not a model, so dimensionality of the data can be assessed.
        step (int): Step to get the data from. On the scale of the datacollector.
        attribute (str): The name of the series to model.

    Returns:
        Tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]: The finished graph
    """

    return visualisation.core.plot_error_bar_horizontal(
        data,
        attributes,
        [str(i) for i in range(0, n)],
        x=[str(i) for i in range(0, num_dims)],
        ylim=[0, np.round(y_max / num_dims)],
        title=f"Energy deviation (per construction, across agents)",
        y_label="Energy",
        n=n,
        **kwargs,
    )


def plot_energy_per_ctx_per_dim_norm(
    data: model.model.ReductionModel | List[List[float]],
    attributes: List[str],
    y_min: int,
    y_max: int,
    n: int,
    step: int = -1,
    **kwargs: Any,
) -> Tuple[matplotlib.figure.Figure, None]:
    """Plot the energy for each construction of a given parameter combnation, plot each dimension as a normal distribution

    Args:
        data (model.model.ReductionModel | List[List[float]]): Either a model instance or a list of values. Length must be two: first item = means, second item = sigmas
        attributes (List[str]): The columns to fetch data from. Always supply, even if input data is not a model, so dimensionality of the data can be assessed.
        y_min (int): Minimum energy value
        y_max (int): Maximum energy value
        n (int): Top n constructions to show distributionsfor. Defaults to None (= show all constructions).
        step (int, optional): Step to get the data from. On the scale of the datacollector. Defaults to -1 (= last step).

    Returns:
        Tuple[matplotlib.figure.Figure, None]: The finished graph
    """

    return visualisation.core.plot_norm_dist_pass(
        data, attributes, [y_min, y_max], n=n, step=step, **kwargs
    )


def plot_energy_differences(
    data: model.model.ReductionModel | List[float],
    attributes: str | List[str],
    **kwargs: Any,
) -> Tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
    """Plot the energy difference across vectors

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
        ylim=[-0.1, 0.1],
        title=f"Mean vector difference energy",
        **kwargs,
    )
