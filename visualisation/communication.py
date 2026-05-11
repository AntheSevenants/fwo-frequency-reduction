import matplotlib.axes
import matplotlib.figure


import model.model
import model.enums
import visualisation.core

from typing import Optional, List, Union, Tuple, Any


def plot_communication(
    data: model.model.ReductionModel | List[List[float]],
    attributes: str | List[str],
    **kwargs: Any,
) -> Tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
    """Plot the communication outcome across agents

    Args:
        data (Union[model.model.ReductionModel, List[float]]): Either a model instance or a list of values.
        attributes (str | List[str]): The column to fetch data from. Always supply, even if input data is not a model, so dimensionality of the data can be assessed.
        **kwargs: Additional keyword arguments passed to parent plotting function.

    Returns:
        Tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]: The finished graph
    """

    return visualisation.core.plot_ratio(
        data,
        attributes,
        model.enums.to_dict(model.enums.CommunicationResult),
        title=f"Communication outcome across agents",
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
