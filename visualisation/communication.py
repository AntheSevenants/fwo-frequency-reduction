import matplotlib.axes
import matplotlib.figure


import model.model
import model.enums
import model.success
import model.entropy
import visualisation.core

from typing import Optional, List, Union, Tuple, Any


def plot_communication(
    data: model.model.ReductionModel | List[List[float]],
    attributes: str | List[str],
    title_override: str | None = None,
    **kwargs: Any,
) -> Tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
    """Plot the communication outcome across agents

    Args:
        data (Union[model.model.ReductionModel, List[float]]): Either a model instance or a list of values.
        attributes (str | List[str]): The column to fetch data from. Always supply, even if input data is not a model, so dimensionality of the data can be assessed.
        title_override: str | None: Overrides the default title. Defaults to None (disabled).
        **kwargs: Additional keyword arguments passed to parent plotting function.

    Returns:
        Tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]: The finished graph
    """

    title = "Communication outcome across agents"
    if title_override:
        title = title_override

    return visualisation.core.plot_ratio(
        data,
        attributes,
        model.enums.to_dict(model.success.CommunicationResult),
        title=title,
        x_label="Steps in the simulation",
        y_label=r"% successful turns",
        **kwargs,
    )


def plot_confusion(
    data: model.model.ReductionModel | List[float],
    attributes: str,
    **kwargs: Any,
) -> Tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
    """Plot the communication outcome across agents

    Args:
        data (Union[model.model.ReductionModel, List[float]]): Either a model instance or a list of values.
        attributes (str): The column to fetch data from. Always supply, even if input data is not a model, so dimensionality of the data can be assessed.
        **kwargs: Additional keyword arguments passed to parent plotting function.

    Returns:
        Tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]: The finished graph
    """

    return visualisation.core.plot_confusion(
        data,
        attributes,
        title=f"Confusion matrix across agents",
        **kwargs,
    )


def plot_reentrance(
    data: model.model.ReductionModel | List[List[float]],
    attributes: str,
    **kwargs: Any,
) -> Tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
    """Plot re-entrance usage across agents

    Args:
        data (Union[model.model.ReductionModel, List[float]]): Either a model instance or a list of values.
        attributes (str): The column to fetch data from. Always supply, even if input data is not a model, so dimensionality of the data can be assessed.
        **kwargs: Additional keyword arguments passed to parent plotting function.

    Returns:
        Tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]: The finished graph
    """

    return visualisation.core.plot_ratio(
        data,
        attributes,
        model.enums.to_dict(model.enums.ReentranceUsage),
        title="Re-entrance usage across agents",
        x_label="Steps in the simulation",
        y_label=r"% re-entrance activated",
        **kwargs,
    )


def plot_decision_entropy(
    data: model.model.ReductionModel | List[float],
    attributes: str,
    num_constructions: int,
    **kwargs: Any,
) -> Tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
    """Plot the decision entropy across agents

    Args:
        data (Union[model.model.ReductionModel, List[float]]): Either a model instance or a list of values.
        attributes (str): The column to fetch data from. Always supply, even if input data is not a model, so dimensionality of the data can be assessed.
        num_constructions (int): Number of constructions in the simulations, used to compute maximum entropy.
        **kwargs: Additional keyword arguments passed to parent plotting function.

    Returns:
        Tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]: The finished graph
    """

    return visualisation.core.plot_value(
        data,
        attributes,
        ylim=[0, model.entropy.compute_maximum_entropy(num_constructions)],
        title="Decision entropy across agents",
        x_label="Steps in the simulation",
        y_label="Decision entropy",
        **kwargs,
    )
