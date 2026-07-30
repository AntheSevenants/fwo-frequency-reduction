import math

import model.model
import model.reporters_model
import model.reporters_agent

import matplotlib.figure
import matplotlib.axes
import matplotlib.ticker as mtick
import matplotlib.pyplot as plt
import numpy as np

import scipy.stats

from typing import Dict, List, Optional, Union, Any, Tuple, Dict

COLOURS = ["blue", "orange", "green", "red", "purple", "brown"]
LINE_STYLES = ["-", "dotted", "dashdot", "dashed"]


# If there are multiple groups in a graph, this can be either
class MultiGroupContext:
    # ... because we programmed a conservator / innovator distinction
    CONSERVATOR_INNOVATOR = 0
    # ... because we are generalising over multiple combinations
    AGGREGATE_EXTENSION = 1


def get_multi_group_context(aggregate_extension: bool) -> int:
    """Return what multi group context we're in depending on the value for aggregate extension

    Args:
        aggregate_extension (bool): Whether this is an aggregate extension graph

    Returns:
        int: MultiGroupContext enum
    """

    return (
        MultiGroupContext.CONSERVATOR_INNOVATOR
        if not aggregate_extension
        else MultiGroupContext.AGGREGATE_EXTENSION
    )


def formatter(x: float, pos: float, scale: int):
    del pos
    return str(int(x * scale))


def scale_x_axis(ax: matplotlib.axes.Axes, scale: int = 100):
    # Do nothing if scale is 1
    if scale == 1:
        return

    ax.xaxis.set_major_formatter(lambda x, pos: formatter(x, pos, scale=scale))


def set_y_axis_percent(ax: matplotlib.axes.Axes):
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, decimals=0))


def check_ax(
    ax: Optional[matplotlib.axes.Axes] = None, disable_title: bool = False
) -> Tuple[
    matplotlib.figure.Figure | matplotlib.figure.SubFigure | None, matplotlib.axes.Axes
]:
    """Check if an Axis is defined. If not, create a new subfigure.

    Args:
        ax (Optional[matplotlib.axes.Axes], optional): The axis variable to be checked. Defaults to None.
        disable_title (bool, optional): Whether the title will be disabled for this figure. Defaults to False.

    Returns:
        Tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]: A tuple containing the figure and axis objects
    """

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))

    fig = ax.get_figure()

    if disable_title:
        plt.tight_layout()

        # Make sure axis titles do not clip
        if isinstance(fig, matplotlib.figure.Figure):
            fig.set_layout_engine("tight")

    return fig, ax


def convert_step(exact_step: float | int, total_steps: int) -> int:
    """Convert a fraction to an absolute step, or return an already absolute step as-is.

    Args:
        exact_step (float | int): The fraction or absolute step
        total_steps (int): The total number of steps

    Returns:
        int: An absolute step
    """

    if exact_step > 0 and exact_step < 1:
        return fraction_to_step(exact_step, total_steps)
    else:
        return int(exact_step)


def fraction_to_step(frac: float, total_steps: int) -> int:
    """Convert a fraction (between 0 and 1) to an absolute step number from the total steps

    Args:
        frac (float): The fraction to be converted to a step
        total_steps (int): The total number of steps

    Returns:
        int: The absolute step
    """

    return round(frac * (total_steps - 1))


def check_attributes(attributes: str | List[str]) -> List[str]:
    """Turn an attributes argument into a list, always!

    Args:
        attributes (str | List[str]): _description_

    Returns:
        List[str]: _description_
    """

    if isinstance(attributes, str):
        attributes = [
            attributes
        ]  # Convert single string to list for uniform processing

    return attributes


def get_line_style(index: int, total_groups: int):
    if total_groups == 1:
        return LINE_STYLES[0]
    else:
        return LINE_STYLES[index + 1]


def get_line_style_by_group_context(
    index: int, total_groups: int, multi_group_context: int
):
    return (
        get_line_style(index, total_groups)
        if multi_group_context == MultiGroupContext.CONSERVATOR_INNOVATOR
        else LINE_STYLES[0]
    )


def get_colour(index: int):
    return COLOURS[index % len(COLOURS)]


def make_legend_label(agent_type_index: int, construction_index: int | None = None):
    agent_type_translation = {
        model.reporters_agent.AgentType.INNOVATOR: "innovator",
        model.reporters_agent.AgentType.CONSERVATOR: "conservator",
    }

    construction_type_translation = {0: "new ctx", 1: "old ctx"}

    if construction_index is not None:
        return f"{construction_type_translation[construction_index]} ({agent_type_translation[agent_type_index]})"
    else:
        return agent_type_translation[agent_type_index]


def make_legend_suffix(micro_macro_index: int) -> str:
    legend = ["(MICRO)", "(MACRO)"]

    return f" {legend[micro_macro_index]}"


def make_legend_label_by_group_context(
    attribute_idx: int,
    construction_index: int | None = None,
    aggregate_extension_x: List[str] | None = None,
):
    if aggregate_extension_x is None:
        return make_legend_label(attribute_idx, construction_index)
    else:
        return aggregate_extension_x[attribute_idx]


def get_ax_figure(ax: matplotlib.axes.Axes):
    """Retrieve the associated figure of an axis

    Args:
        ax (matplotlib.axes.Axes): The axis of which to retrieve the figure

    Returns:
        matplotlib.figure.Figure: The associated figure
    """

    if isinstance(ax.figure, matplotlib.figure.SubFigure):
        return ax.figure.figure
    else:
        return ax.figure


def filter_for_agent(
    matrix: np.ndarray, agent_filter: Optional[int] = None
) -> np.ndarray:
    """Filter an input matrix for the data associated with a specific agent

    Args:
        matrix (np.ndarray): The input matrix (from the DataCollector)
        agent_filter (Optional[int], optional): The index of the specified agent. Defaults to None.

    Returns:
        np.ndarray: The filtered matrix
    """

    # If needed, index data for a specific agent
    if agent_filter is not None:
        # 3D matrix
        dimensionality = len(matrix.shape)

        if dimensionality == 3:
            matrix = matrix[:, agent_filter, :]
        else:
            matrix = matrix[:, agent_filter]

    return matrix


def get_value_lists(
    data: Union[
        model.model.ReductionModel,
        Union[List[float], List[List[float]]],
        List[List[List[float]]],
    ],
    attributes: Union[str, List[str]],
    agent_filter: Optional[int] = None,
) -> List[np.ndarray]:
    """Return a list of values based on a model instance or a list of values

    Args:
        data (Union[model.model.ReductionModel, List[float], List[List[float]]], List[List[List[float]]]]): Either a model instance or a list of values
        attributes (Union[str, List[str]]]): The names of the series to plot. Always supply, even if input data is not a model, so dimensionality of the data can be assessed.
        agent_filter (Optional[int], optional): The index of the agent you want to filter for. If not supplied, no filtering is applied. Defaults to None.

    Raises:
        ValueError: Attribute must be specified when plotting data from a model instance.

    Returns:
        List[np.ndarray]: List of model result value matrices
    """

    # Convert single string to list for uniform processing
    if isinstance(attributes, str):
        attributes = [attributes]

    if attributes is not None:
        if len(attributes) > len(LINE_STYLES):
            raise ValueError(
                f"Number of attributes cannot exceed number of line styles (= {len (LINE_STYLES)})"
            )

    value_lists = []

    # Model data comes from the model directly
    if isinstance(data, model.model.ReductionModel):
        # This can only work if the attribute is defined
        if attributes is None:
            raise ValueError(
                "Attribute must be specified when plotting data from a model instance."
            )

        df = data.datacollector.get_model_vars_dataframe()

        for attribute in attributes:
            if attribute is None:
                raise ValueError("Supplied attribute cannot be None")

            value_list = np.stack(df[attribute].tolist())
            # If needed, index data for a specific agent
            value_list = filter_for_agent(value_list, agent_filter)

            value_lists.append(value_list)
    else:
        if len(data) == 0:
            raise ValueError("Supplied value list cannot have zero length")

        # If just a single value list is supplied, wrap in an outer list
        if len(attributes) == 1:
            _data = data
            # _data = [data]
            pass
        else:
            _data = data

        # Go over each inner list and conver to numpy array
        for value_list in _data:
            # Assume a valid list of data
            value_lists.append(np.array(value_list))

    return value_lists


def check_min_max_data(
    data: Union[model.model.ReductionModel, List[float], List[List[float]]],
    min_data: Union[List[float], List[List[float]], None],
    max_data: Union[List[float], List[List[float]], None],
) -> Union[Tuple[np.ndarray, np.ndarray], Tuple[None, None]]:
    """Check the supplied minimal and maximal value lists and raise errors if the supplied data does not make sense.

    Args:
        data (Union[model.model.ReductionModel, List[float]], List[List[float]]): Either a model instance or a list of values
        min_data (Union[List[float], List[List[float]], None]): List of minimal values. Needs to be defined together with max_data.
        max_data (Union[List[float], List[List[float]], None]): List of maximal values. Needs to be defined together with min_data.

    Raises:
        ValueError: Data cannot be a model instance if min_data and max_data are defined
        ValueError: max_data cannot be defined if min_data is undefined
        ValueError: min_data cannot be defined if max_data is undefined

    Returns:
        Union[Tuple[np.ndarray, np.ndarray], Tuple[None, None]]: If the check was successful, (min_data, max_data) as numpy arrays, else (None, None)
    """

    if (
        isinstance(data, model.model.ReductionModel)
        and min_data is not None
        and max_data is not None
    ):
        raise ValueError(
            "Supplied data cannot be a model instance if min_data and max_data are defined"
        )

    if min_data is not None and max_data is None:
        raise ValueError("max_data cannot be None if min_data is set")

    if max_data is not None and min_data is None:
        raise ValueError("min_data cannot be None if max_data is set")

    if max_data is not None and min_data is not None:
        return np.array(min_data), np.array(max_data)

    return None, None


def plot_value(
    data: Union[model.model.ReductionModel, List[float]],
    attributes: str | List[str],
    ylim: Optional[List[float]] = None,
    x_scale_factor: int = 1,
    ax: Optional[matplotlib.axes.Axes] = None,
    agent_filter: Optional[int] = None,
    min_data: List[float] | List[List[float]] | None = None,
    max_data: List[float] | List[List[float]] | None = None,
    step: int | float | None = None,
    title: Optional[str] = None,
    legend_title: Optional[str] = None,
    legend_labels: List[str] | None = None,
    x_label: str | None = None,
    y_label: str | None = None,
    disable_title: bool = False,
    aggregate_extension_x: List[str] | None = None,
    plot_mean: bool = False,
) -> Tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
    """Plot a desired series of values from a model run

    Args:
        data (Union[model.model.ReductionModel, List[float]]): Either a model instance or a list of values
        attributes (Union[str, List[str]]): The names of the series to model. Always supply, even if input data is not a model, so dimensionality of the data can be assessed.
        ylim (Optional[List[float]], optional): The expected range of values for y axis. Defaults to None.
        x_scale_factor (int, optional): The factor to scale the x axis ticks by. Defaults to 1.
        ax (Optional[matplotlib.axes.Axes], optional): A pre-existing axis. Pass if you are building a multi-plot. Defaults to None.
        agent_filter (Optional[int], optional): The index of the agent you want to filter values for. If not supplied, no filtering is applied. Defaults to None.
        min_data (List[float] | List[List[float]] | None, optional): List of minimal values. Needs to be defined together with max_data. Defaults to None.
        max_data (List[float] | List[List[float]] | None, optional): List of maximal values. Needs to be defined together with min_data. Defaults to None.
        step (int | float): Step to highlighted in the graph (on the scale of the datacollector). Can also be a fraction, will be converted to an absolute step. Defaults to None (= no highlight).
        title (Optional[str], optional): The title for the graph. Defaults to None.
        legend_title (Optional[str], optional): The title for the legend. Defaults to None.
        legend_labels (List[str], optional): The labels for the legend. Defaults to None.
        x_label (str, optional): The label for the X axis. Defaults to None.
        y_label (str, optional): The label for the Y axis. Defaults to None.
        disable_title (bool, optional): Whether to show a title for this graph. Defaults to False.
        aggregate_extension_x (List[str], optional): A list of values for the legend of an aggregate extension graph. Defaults to None.
        plot_mean (bool, optional): Whether to indicate the mean. Defaults to False.

    Returns:
        Tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]: The finished graph
    """

    # Convert single string to list for uniform processing
    attributes = check_attributes(attributes)

    # Get the right data based on the supplied arguments
    value_lists = get_value_lists(data, attributes, agent_filter)
    num_groups = len(value_lists)
    # Check if min and max data are supplied correctly
    _min_data, _max_data = check_min_max_data(data, min_data, max_data)

    fig, ax = check_ax(ax, disable_title)

    multi_group_context = get_multi_group_context(aggregate_extension_x is not None)

    for attribute_idx, value_list in enumerate(value_lists):
        # Across parameter combinations, different colours fit better
        line_colour = COLOURS[attribute_idx]
        # I'm attributing line style to micro/macro
        line_style = get_line_style_by_group_context(
            attribute_idx, num_groups, multi_group_context
        )
        if legend_labels is None:
            legend_label = make_legend_label_by_group_context(
                attribute_idx, aggregate_extension_x=aggregate_extension_x
            )
        else:
            legend_label = legend_labels[attribute_idx]

        ax.plot(value_list, color=line_colour, linestyle=line_style, label=legend_label)

        # Plot the shaded area between min and max values
        if _min_data is not None and _max_data is not None:
            ax.fill_between(
                x=range(len(value_list)),
                y1=_min_data[attribute_idx],
                y2=_max_data[attribute_idx],
                color=line_colour,
                alpha=0.2,
            )

        if plot_mean:
            value_mean = float(np.mean(value_list))
            ax.axhline(value_mean, color="gray")

    # Draw step focus line if required
    if step is not None:
        _step = convert_step(step, len(value_lists[0]))
        print(_step)
        ax.axvline(_step, color="red")

    scale_x_axis(ax, x_scale_factor)

    if ylim is not None:
        ax.set_ylim(*ylim)

    if x_label is not None:
        ax.set_xlabel(x_label)
    if y_label is not None:
        ax.set_ylabel(y_label)

    if title is not None and not disable_title:
        ax.set_title(title)

    if num_groups > 1:
        legend_kwargs = {}

        if legend_title is not None:
            legend_kwargs["title"] = legend_title

        ax.legend(**legend_kwargs)

    output_fig = get_ax_figure(ax)
    plt.close(output_fig)

    return (output_fig, ax)


def plot_ratio(
    data: Union[model.model.ReductionModel, List[List[float]]],
    attributes: Union[str, List[str]],
    enum_translation: Dict[int, str],
    filter_dimension: int | None = None,
    filter_matrix_dimension: int | None = None,
    ylim: List[float] = [0, 1],
    x_scale_factor: int = 1,
    ax: Optional[matplotlib.axes.Axes] = None,
    agent_filter: Optional[int] = None,
    min_data: List[float] | List[List[float]] | None = None,
    max_data: List[float] | List[List[float]] | None = None,
    step: int | float | None = None,
    title: Optional[str] = None,
    legend_title: Optional[str] = None,
    legend_labels: List[str] | None = None,
    x_label: str | None = None,
    y_label: str | None = None,
    disable_title: bool = False,
    aggregate_extension_x: List[str] | None = None,
) -> Tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
    """Plot a desired series of ratio values from a model run

    Args:
        data (Union[model.model.ReductionModel, List[List[float]]): Either a model instance or a list of values
        attributes (Union[str, List[str]]): The names of the series to model. Always supply, even if input data is not a model, so dimensionality of the data can be assessed.
        enum_translation (Dict[int, str]): Provide the translation for the dimensions in the array.
        filter_dimension (int | None): Only plot one specific dimension of the array. Defaults to None.
        filter_matrix_dimension (int | None): Only plot one specific dimension of the inner matrix. Defaults to None.
        ylim (List[float], optional): The expected range of values, will be the y axis. Defaults to [0, 1].
        x_scale_factor (int, optional): The factor to scale the x axis ticks by. Defaults to 1.
        ax (Optional[matplotlib.axes.Axes], optional): A pre-existing axis. Pass if you are building a multi-plot. Defaults to None.
        agent_filter (Optional[int], optional): The index of the agent you want to filter values for. If not supplied, no filtering is applied. Defaults to None.
        min_data (List[float] | List[List[float]] | None, optional): List of minimal values. Needs to be defined together with max_data. Defaults to None.
        max_data (List[float] | List[List[float]] | None, optional): List of maximal values. Needs to be defined together with min_data. Defaults to None.
        step (int | float): Step to highlighted in the graph (on the scale of the datacollector). Can also be a fraction, will be converted to an absolute step. Defaults to None (= no highlight).
        title (Optional[str], optional): The title for the graph. Defaults to None.
        legend_title (Optional[str], optional): The title for the legend. Defaults to None.
        legend_labels (List[str], optional): The labels for the legend. Defaults to None.
        x_label (str, optional): The label for the X axis. Defaults to None.
        y_label (str, optional): The label for the Y axis. Defaults to None.
        disable_title (bool, optional): Whether to show a title for this graph. Defaults to False.
        aggregate_extension_x (List[str], optional): A list of values for the legend of an aggregate extension graph. Defaults to None.


    Raises:
        ValueError: If the number of attributes to plot is larger than the supported number of line styles

    Returns:
        Tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]: The finished graph
    """

    if isinstance(attributes, str):
        attributes = [
            attributes
        ]  # Convert single string to list for uniform processing

    # Get the right data based on the supplied arguments
    value_lists = get_value_lists(data, attributes, agent_filter)
    num_groups = len(value_lists)
    # Check if min and max data are supplied correctly
    _min_data, _max_data = check_min_max_data(data, min_data, max_data)

    fig, ax = check_ax(ax, disable_title)

    multi_group_context = get_multi_group_context(aggregate_extension_x is not None)

    for attribute_idx, matrix in enumerate(value_lists):
        is_micro_macro = matrix.ndim == 3
        start_idx = filter_dimension if filter_dimension is not None else 0

        # if there is a micro macro layer first, there is another dimension
        # inbetween time and the different properties

        # Determine the range of the iteration based on dimensionality
        # If 2D: iterate over columns (dim 1)
        # If 3D: iterate over properties (dim 2)
        iter_dim = 2 if is_micro_macro else 1
        max_iter = matrix.shape[iter_dim]

        for i in range(start_idx, max_iter):
            # Determine label once per 'i' iteration
            if legend_labels is None:
                legend_label = (
                    enum_translation[i]
                    if aggregate_extension_x is None
                    else aggregate_extension_x[attribute_idx]
                )
            else:
                legend_label = legend_labels[attribute_idx]

            if not is_micro_macro:
                # --- 2D CASE ---
                line_style = get_line_style_by_group_context(
                    attribute_idx, num_groups, multi_group_context
                )

                ax.plot(
                    matrix[:, i],
                    color=COLOURS[attribute_idx],
                    linestyle=line_style,
                    label=legend_label,
                )

                if _min_data is not None and _max_data is not None:
                    ax.fill_between(
                        x=range(matrix.shape[0]),
                        y1=_min_data[attribute_idx, :, i],
                        y2=_max_data[attribute_idx, :, i],
                        color=COLOURS[attribute_idx],
                        alpha=0.2,
                    )
            else:
                start_idx_j = (
                    filter_matrix_dimension
                    if filter_matrix_dimension is not None
                    else 0
                )

                # --- 3D CASE ---
                # We iterate through the secondary dimension (j)
                for j in range(start_idx_j, matrix.shape[1]):
                    line_style = LINE_STYLES[j]

                    legend_label_suffixed = legend_label
                    if filter_matrix_dimension is None:
                        legend_label_suffixed += make_legend_suffix(j)

                    ax.plot(
                        matrix[:, j, i],
                        color=COLOURS[attribute_idx],
                        linestyle=line_style,
                        label=legend_label_suffixed,
                    )

                    if _min_data is not None and _max_data is not None:
                        ax.fill_between(
                            x=range(matrix.shape[0]),
                            y1=_min_data[attribute_idx, :, j, i],
                            y2=_max_data[attribute_idx, :, j, i],
                            color=COLOURS[attribute_idx],
                            alpha=0.2,
                        )

                    # If we are filtering, we only process the first valid index and move on
                    if filter_matrix_dimension is not None:
                        break

            # If we are filtering, we only process the first valid index and move on
            if filter_dimension is not None:
                break

    if title is not None and not disable_title:
        ax.set_title(title)

    # Draw step focus line if required
    if step is not None:
        _step = convert_step(step, len(value_lists[0]))
        ax.axvline(_step, color="red")

    scale_x_axis(ax, x_scale_factor)

    ax.set_ylim(*ylim)
    ax.set_yticks(np.arange(ylim[0], ylim[1] + 0.1, 0.1))

    set_y_axis_percent(ax)

    if x_label is not None:
        ax.set_xlabel(x_label)
    if y_label is not None:
        ax.set_ylabel(y_label)

    # if num_groups > 1:
    legend_kwargs = {}

    if legend_title is not None:
        legend_kwargs["title"] = legend_title

    ax.legend(**legend_kwargs)

    output_fig = get_ax_figure(ax)
    plt.close(output_fig)

    return (output_fig, ax)


def plot_ratio_pass(
    data: Union[model.model.ReductionModel, List[List[float]], List[List[List[float]]]],
    attributes: str,
    ylim: Optional[List[float]] = None,
    y_scale_factor: int = 1,
    baseline: Optional[float] = None,
    ax: Optional[matplotlib.axes.Axes] = None,
    title: Optional[str] = None,
    disable_title: Optional[bool] = False,
) -> matplotlib.figure.Figure:
    """Plot a desired series of ratio values for all agents at once for a given model run

    Args:
        data (Union[model.model.ReductionModel, List[List[float]], List[List[float]]): Either a model instance or a list of values
        attributes (Optional[str], optional): The name of the series to model.
        ylim (Optional[List[float]], optional): The expected range of values for y axis. Defaults to None.
        y_scale_factor (int, optional): The factor to scale the y axis ticks by. Defaults to 1.
        baseline (Optional[float], optional): The baseline to show in each subplot. Can mark a default value. Defaults to None.
        ax (Optional[matplotlib.axes.Axes], optional): A pre-existing axis. Please do not pass any axes currently. Defaults to None.
        title (Optional[str], optional): The title for the graph. Defaults to None.
        disable_title (Optional[bool], optional): Whether to show a title for this graph.. Defaults to False.

    Raises:
        ValueError: Passing an Axis through ax is currently not supported
        ValueError: Input matrix dimensions can only be 2 or 3

    Returns:
        matplotlib.figure.Figure: The created graph
    """

    # Get the right data based on the supplied arguments
    matrix = get_value_lists(data, attributes)[0]

    if ax is not None:
        raise ValueError(
            "Cannot do mosaic plots for this graph type. Please do not pass an axis."
        )

    # num agents = size of list ite
    num_agents = matrix[0].shape[0]
    fig, axes = plt.subplots(nrows=1, ncols=num_agents, figsize=(15, 10), sharey=True)

    num_steps = matrix.shape[0]
    time_steps = np.arange(num_steps)

    num_dimensions = len(matrix.shape)

    baseline_to_plot = None
    if baseline is not None:
        # Vertical baseline which shows 0.5
        baseline_to_plot = np.full(num_steps, baseline)

    for i, _ax in enumerate(fig.axes):
        # Plot baselines first
        if baseline_to_plot is not None:
            _ax.plot(
                baseline_to_plot,
                time_steps,
                color="gray",
                alpha=0.1,
                linestyle="dashed",
            )

        if num_dimensions == 3:
            _ax.plot(matrix[:, i, 0], time_steps, color="blue")
        elif num_dimensions == 2:
            _ax.plot(matrix[:, i], time_steps, color="blue")
        else:
            raise ValueError("Invalid number of dimensions")

        if ylim is not None:
            _ax.set_xlim(*ylim)
        _ax.set_title(f"{i + 1}")
        _ax.set_xticks([])
        # ax.set_xlabel('Construction 0 usage')
        _ax.grid(True)

        # X will become Y further down
        scale_x_axis(_ax, y_scale_factor)

        # Disable ugly boxes
        for spine in _ax.spines.values():
            spine.set_visible(False)

    fig.axes[0].set_ylabel("Time steps in the simulation")
    fig.axes[0].invert_yaxis()

    plt.close(fig)

    return fig


def check_if_none(variable_name: str, value: Any):
    """Check if a value is None when it should not be.

    Args:
        variable_name (str): Name of the variable that is being checked.
        value (Any): Value of the variable that is being checked.

    Raises:
        ValueError: Raised if the value is None.
    """

    if value is None:
        raise ValueError(f'"{variable_name}" cannot be None')


def plot_histogram(
    data: Union[model.model.ReductionModel, List[List[float]]],
    attributes: str,
    ax: Optional[matplotlib.axes.Axes] = None,
    bin_range: Optional[List[float]] = None,
    title: Optional[str] = None,
    disable_title: bool = False,
    aggregate_extension_x: Any = None,
) -> Tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
    """Plot a desired series of values from a model run

    Args:
        data (Union[model.model.ReductionModel, List[List[float]]]): A list of values
        attribute (Optional[str], optional): The name of the series to model.
        ax (Optional[matplotlib.axes.Axes], optional): A pre-existing axis. Pass if you are building a multi-plot. Defaults to None.
        title (Optional[str], optional): The title for the graph. Defaults to None.
        disable_title (bool, optional): Whether to show a title for this graph. Defaults to False.

    Returns:
        Tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]: The finished graph
    """

    # Get the right data based on the supplied arguments
    value_list = get_value_lists(data, attributes)[0]

    fig, ax = check_ax(ax, disable_title)

    _bins = 10
    if bin_range is not None:
        _bins = bin_range

    ax.hist(value_list, bins=_bins, edgecolor="black")
    ax.set_xlabel("Slope values")
    ax.set_ylabel("Frequency")

    if title is not None and not disable_title:
        ax.set_title(title)

    output_fig = get_ax_figure(ax)
    plt.close(output_fig)

    return (output_fig, ax)


def plot_error_bar(
    data: model.model.ReductionModel | List[float] | List[List[float]],
    attributes: str | List[str],
    x: List[str] | None = None,
    ylim: Optional[List[float]] = None,
    ax: Optional[matplotlib.axes.Axes] = None,
    min_data: List[float] | List[List[float]] | None = None,
    max_data: List[float] | List[List[float]] | None = None,
    step: int | float = -1,
    n: int | None = None,
    x_label: Optional[str] = None,
    y_label: Optional[str] = None,
    title: Optional[str] = None,
    legend_title: Optional[str] = None,
    legend_labels: List[str] | None = None,
    disable_title: bool = False,
    aggregate_extension_x: List[str] | None = None,
) -> Tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
    """Plot a bar chart with values from a model run

    Args:
        data (model.model.ReductionModel | List[float] | List[List[float]]): A list of values
        attributes (Optional[str], optional): The name of the series to model.
        x (List[str]): A list of values for the X axis
        ylim (Optional[List[float]], optional): The expected range of values for y axis. Defaults to None.
        ax (Optional[matplotlib.axes.Axes], optional): A pre-existing axis. Pass if you are building a multi-plot. Defaults to None.
        min_data (List[float] | List[List[float]] | None, optional): List of minimal values. Needs to be defined together with max_data. Defaults to None.
        max_data (List[float] | List[List[float]] | None, optional): List of maximal values. Needs to be defined together with min_data. Defaults to None.
        step (float): Step to get the data from (on the scale of the datacollector). Can also be a fraction, will be converted to an absolute step.
        n (int | None): Maximum number of items in the value lists to be plotted. Can be used for limiting the number of vectors plotted. Defaults to None (= show all items).
        x_label (Optional[str], optional): The label for the X axis. Defaults to None.
        y_label (Optional[str], optional): The label for the Y axis. Defaults to None.
        title (Optional[str], optional): The title for the graph. Defaults to None.
        legend_title (Optional[str], optional): The title for the legend. Defaults to None.
        legend_labels (List[str], optional): The labels for the legend. Defaults to None.
        disable_title (bool, optional): Whether to show a title for this graph. Defaults to False.
        aggregate_extension_x (List[str] | None, optional): Labels for aggregate values, to be used as group labels. Defaults to None.

    Returns:
        Tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]: The finished graph
    """

    # Convert single string to list for uniform processing
    attributes = check_attributes(attributes)

    # Get the right data based on the supplied arguments
    value_lists = get_value_lists(data, attributes)
    num_groups = len(value_lists)

    # Check if min and max data are supplied correctly
    _min_data, _max_data = check_min_max_data(data, min_data, max_data)

    fig, ax = check_ax(ax, disable_title)

    multi_group_context = get_multi_group_context(aggregate_extension_x is not None)

    # For the vectors plot. Quit early if needed!
    _n = n if n is not None else len(value_lists)

    for attribute_idx, value_list in enumerate(value_lists):
        # Convert to absolute step
        _step = convert_step(step, len(value_list))

        # Get the data from the right step
        value_list = value_list[_step, :]

        __min_data, __max_data = None, None
        if _min_data is not None:
            __min_data = _min_data[attribute_idx][step, :]
        if _max_data is not None:
            __max_data = _max_data[attribute_idx][step, :]

        if x is None:
            _x = [str(x) for x in list(range(len(value_list)))]
        else:
            _x = x

        _yerr = (
            None
            if __min_data is None or __max_data is None
            else [np.abs(value_list - __min_data), np.abs(__max_data - value_list)]
        )

        line_colour = get_colour(attribute_idx)
        if legend_labels is None:
            legend_label = make_legend_label_by_group_context(
                attribute_idx, aggregate_extension_x=aggregate_extension_x
            )
        else:
            legend_label = legend_labels[attribute_idx]

        ax.errorbar(
            _x,
            value_list,
            yerr=_yerr,
            fmt="s",
            capsize=5,
            ecolor="lightgray",
            color=line_colour,
            elinewidth=1.5,
            label=legend_label,
        )

        if (attribute_idx + 1) >= _n:
            break

    if ylim is not None:
        ax.set_ylim(*ylim)

    if x_label is not None:
        ax.set_xlabel(x_label)
    if y_label is not None:
        ax.set_ylabel(y_label)

    if title is not None and not disable_title:
        ax.set_title(title)

    if num_groups > 1:
        legend_kwargs = {}

        if legend_title is not None:
            legend_kwargs["title"] = legend_title

        ax.legend(**legend_kwargs)

    output_fig = get_ax_figure(ax)
    plt.close(output_fig)

    return (output_fig, ax)


def plot_error_bar_horizontal(
    data: model.model.ReductionModel | List[float] | List[List[float]],
    attributes: str | List[str],
    group_labels: List[str],
    x: List[str] | None = None,
    ylim: Optional[List[float]] = None,
    ax: Optional[matplotlib.axes.Axes] = None,
    min_data: List[float] | List[List[float]] | None = None,
    max_data: List[float] | List[List[float]] | None = None,
    step: int | float = -1,
    n: int | None = None,
    x_label: Optional[str] = None,
    y_label: Optional[str] = None,
    title: Optional[str] = None,
    legend_title: Optional[str] = None,
    legend_labels: List[str] | None = None,
    disable_title: bool = False,
) -> Tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
    """Plot a bar chart with values from a model run

    Args:
        data (model.model.ReductionModel | List[float] | List[List[float]]): A list of values
        attributes (Optional[str], optional): The name of the series to model.
        group_labels (List[str]). Group labels. Makes this graph type incompatible with aggregate extension.
        x (List[str]): A list of values for the X axis
        ylim (Optional[List[float]], optional): The expected range of values for y axis. Defaults to None.
        ax (Optional[matplotlib.axes.Axes], optional): A pre-existing axis. Pass if you are building a multi-plot. Defaults to None.
        min_data (List[float] | List[List[float]] | None, optional): List of minimal values. Needs to be defined together with max_data. Defaults to None.
        max_data (List[float] | List[List[float]] | None, optional): List of maximal values. Needs to be defined together with min_data. Defaults to None.
        step (float): Step to get the data from (on the scale of the datacollector). Can also be a fraction, will be converted to an absolute step.
        n (int | None): Maximum number of items in the value lists to be plotted. Can be used for limiting the number of vectors plotted. Defaults to None (= show all items).
        x_label (Optional[str], optional): The label for the X axis. Defaults to None.
        y_label (Optional[str], optional): The label for the Y axis. Defaults to None.
        title (Optional[str], optional): The title for the graph. Defaults to None.
        legend_title (Optional[str], optional): The title for the legend. Defaults to None.
        legend_labels (List[str], optional): The labels for the legend. Defaults to None.
        disable_title (bool, optional): Whether to show a title for this graph. Defaults to False.

    Returns:
        Tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]: The finished graph
    """

    # Convert single string to list for uniform processing
    attributes = check_attributes(attributes)

    # Get the right data based on the supplied arguments
    value_list = get_value_lists(data, attributes)[0]
    num_groups = len(value_list)

    # Check if min and max data are supplied correctly
    _min_data, _max_data = check_min_max_data(data, min_data, max_data)

    # Convert to absolute step
    _step = convert_step(step, len(value_list))

    # Get the data from the right step
    value_list = value_list[_step, :]

    __min_data, __max_data = None, None
    if _min_data is not None:
        __min_data = _min_data[0, _step]
    if _max_data is not None:
        __max_data = _max_data[0, _step]

    fig, ax = check_ax(ax, disable_title)

    if x is None:
        _x = [str(x) for x in list(range(len(value_list)))]
    else:
        _x = x

    # For the vectors plot. Quit early if needed!
    _n = n if n is not None else len(value_list)

    for attribute_idx, vector in enumerate(value_list):
        _yerr = (
            None
            if __min_data is None or __max_data is None
            else [
                np.abs(vector - __min_data[attribute_idx]),
                np.abs(__max_data[attribute_idx] - vector),
            ]
        )

        line_colour = get_colour(attribute_idx)
        if legend_labels is None:
            legend_label = group_labels[attribute_idx]
        else:
            legend_label = legend_labels[attribute_idx]

        ax.errorbar(
            _x,
            vector,
            yerr=_yerr,
            fmt="s",
            capsize=5,
            ecolor="lightgray",
            color=line_colour,
            elinewidth=1.5,
            label=legend_label,
        )

        if (attribute_idx + 1) >= _n:
            break

    if ylim is not None:
        ax.set_ylim(*ylim)

    if x_label is not None:
        ax.set_xlabel(x_label)
    if y_label is not None:
        ax.set_ylabel(y_label)

    if title is not None and not disable_title:
        ax.set_title(title)

    if num_groups > 1:
        legend_kwargs = {}

        if legend_title is not None:
            legend_kwargs["title"] = legend_title

        ax.legend(**legend_kwargs)

    output_fig = get_ax_figure(ax)
    plt.close(output_fig)

    return (output_fig, ax)


def plot_bar(
    data: model.model.ReductionModel | List[float],
    attributes: str,
    x: List[str] | None = None,
    ylim: Optional[List[float]] = None,
    ax: Optional[matplotlib.axes.Axes] = None,
    step: int | float = -1,
    x_label: Optional[str] = None,
    y_label: Optional[str] = None,
    title: Optional[str] = None,
    disable_title: bool = False,
) -> Tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
    """Plot a bar chart with values from a model run

    Args:
        data (List[float]): A list of values
        attributes (Optional[str], optional): The name of the series to model.
        x (List[str]): A list of values for the X axis
        ylim (Optional[List[float]], optional): The expected range of values for y axis. Defaults to None.
        ax (Optional[matplotlib.axes.Axes], optional): A pre-existing axis. Pass if you are building a multi-plot. Defaults to None.
        step (float): Step to get the data from (on the scale of the datacollector). Can also be a fraction, will be converted to an absolute step.
        x_label (Optional[str], optional): The label for the X axis. Defaults to None.
        y_label (Optional[str], optional): The label for the Y axis. Defaults to None.
        title (Optional[str], optional): The title for the graph. Defaults to None.
        disable_title (bool, optional): Whether to show a title for this graph. Defaults to False.

    Returns:
        Tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]: The finished graph
    """

    # Get the right data based on the supplied arguments
    value_list = get_value_lists(data, attributes)[0]
    # Convert to absolute step
    _step = convert_step(step, len(value_list))
    # Get the data from the right step
    value_list = value_list[_step, :]

    fix, ax = check_ax(ax, disable_title)

    if x is None:
        _x = [str(x) for x in list(range(len(value_list)))]
    else:
        _x = x

    ax.bar(_x, value_list, edgecolor="black")

    if ylim is not None:
        ax.set_ylim(*ylim)

    if x_label is not None:
        ax.set_xlabel(x_label)
    if y_label is not None:
        ax.set_ylabel(y_label)

    if title is not None and not disable_title:
        ax.set_title(title)

    output_fig = get_ax_figure(ax)
    plt.close(output_fig)

    return (output_fig, ax)


def plot_confusion(
    data: model.model.ReductionModel | List[float],
    attributes: str,
    ax: Optional[matplotlib.axes.Axes] = None,
    step: int | float | None = -1,
    n: int | None = None,
    x_label: Optional[str] = None,
    y_label: Optional[str] = None,
    title: Optional[str] = None,
    disable_title: bool = False,
) -> Tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
    """Plot a bar chart with values from a model run

    Args:
        data (List[float]): A list of values
        attributes (Optional[str], optional): The name of the series to model.
        ax (Optional[matplotlib.axes.Axes], optional): A pre-existing axis. Pass if you are building a multi-plot. Defaults to None.
        step (int | None): Step to get the data from (on the scale of the datacollector). Can also be a fraction, will be converted to an absolute step. If set to None, will take the mean across all steps.
        n (int | None): Top n constructions to show confusion for. Defaults to None.
        x_label (Optional[str], optional): The label for the X axis. Defaults to None.
        y_label (Optional[str], optional): The label for the Y axis. Defaults to None.
        title (Optional[str], optional): The title for the graph. Defaults to None.
        disable_title (bool, optional): Whether to show a title for this graph. Defaults to False.

    Returns:
        Tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]: The finished graph
    """

    # Get the right data based on the supplied arguments
    value_list = get_value_lists(data, attributes)[0]

    if step is not None:
        # Get the data from the right step
        # Convert to absolute step
        _step = convert_step(step, len(value_list))
        matrix = value_list[_step, :]

        # Adjust title
        if title is not None:
            title = f"{title} (t={_step})"
    else:
        # Take mean across step axis
        matrix = value_list.mean(axis=0)

    # How many rows to show?
    _n = matrix.shape[0] if n is None else n

    fix, ax = check_ax(ax, disable_title)

    row_sums = matrix.sum(axis=1, keepdims=True)
    normalized_confusion_matrix = matrix / row_sums
    ax.matshow(normalized_confusion_matrix[0:_n, 0:_n])

    if x_label is not None:
        ax.set_xlabel(x_label)
    if y_label is not None:
        ax.set_ylabel(y_label)

    if title is not None and not disable_title:
        ax.set_title(title)

    output_fig = get_ax_figure(ax)
    plt.close(output_fig)

    return (output_fig, ax)


def plot_norm_dist_pass(
    data: Union[model.model.ReductionModel, List[List[float]]],
    attributes: List[str],
    xlim: List[float],
    ylim: List[float] | None = None,
    auto_xlim: bool = False,
    ax: Optional[matplotlib.axes.Axes] = None,
    step: int | float = -1,
    n: int | None = None,
    dims_to_plot: List[int] | None = None,
    title: Optional[str] = None,
    disable_title: Optional[bool] = False,
) -> Tuple[matplotlib.figure.Figure, None]:
    """Plot a desired series of normal distributions across all dimensions.

    Args:
        data (Union[model.model.ReductionModel, List[List[float]]): Either a model instance or a list of values
        attributes (List[str]): The names of the series to model.
        xlim (List[float]): The expected range of values for x axis. Defaults to None.
        ylim (List[float] | None, optional): The expected range of values for y axis. Defaults to None.
        auto_xlim (bool): Automatically set the xlim. Defaults to False.
        ax (Optional[matplotlib.axes.Axes], optional): A pre-existing axis. Please do not pass any axes currently. Defaults to None.
        step (float): Step to get the data from (on the scale of the datacollector). Can also be a fraction, will be converted to an absolute step. Defaults to -1.
        n (int | None): Maximum number of items in the value lists to be plotted. Can be used for limiting the number of distributions plotted. Defaults to None (= show all items).
        dims_to_plot (List[int] | None). List of dimensions that should be plotted. Defaults to all (= None).
        title (Optional[str], optional): The title for the graph. Defaults to None.
        disable_title (Optional[bool], optional): Whether to show a title for this graph. Defaults to False.

    Raises:
        ValueError: Passing an Axis through ax is currently not supported
        ValueError: Input matrix dimensions can only be 2 or 3

    Returns:
        Tuple[matplotlib.figure.Figure, None]: The created graph
    """

    # Get the right data based on the supplied arguments
    value_lists = get_value_lists(data, attributes)
    if len(value_lists) != 2:
        raise ValueError(
            "Supplied data should be of dimensionality two (= means + sigmas)"
        )

    if ax is not None:
        raise ValueError(
            "Cannot do mosaic plots for this graph type. Please do not pass an axis."
        )

    # data dimensionality =
    # layer 1: 0 = mean, 1 = sigma
    # layer 2: steps
    # layer 3: matrix, dim 0 = #ctxs, dim 1 = #vector dims

    # num_dimensions = shape 1
    if dims_to_plot is None:
        num_dims = value_lists[0][0].shape[1]
        dims_to_plot = list(range(0, num_dims))
    else:
        num_dims = len(dims_to_plot)

    # Convert to absolute step
    _step = convert_step(step, len(value_lists[0]))
    # How many constructions to plot?
    _n = n if n is not None else value_lists[0][0].shape[0]

    fig, axes = plt.subplots(
        nrows=num_dims, ncols=1, figsize=(6, 2.5 * num_dims), sharex=True
    )

    # Possible vector values
    x = np.arange(xlim[0], xlim[1] + 1, 1)

    THRESHOLD = 0.001

    current_min = math.inf
    current_max = -math.inf

    for i, _ax in enumerate(fig.axes):
        for ctx_index in range(_n):
            dim_index = dims_to_plot[i]

            mu = value_lists[0][_step][ctx_index, dim_index]
            sigma = value_lists[1][_step][ctx_index, dim_index]

            y = scipy.stats.norm.pdf(x, mu, sigma)
            # Mask values below the threshold by replacing them with NaN
            y[y < THRESHOLD] = np.nan

            # Get border values
            is_nan = np.isnan(y)
            changes = np.diff(is_nan.astype(int))
            start_nan_indices = np.where(changes == -1)[0][0]
            end_nan_indices = np.where(changes == 1)[0][0]

            lowest = x[start_nan_indices]
            highest = x[end_nan_indices + 1]

            if lowest < current_min:
                current_min = lowest
            if highest > current_max:
                current_max = highest

            colour = get_colour(ctx_index)
            label = f"Ctx {ctx_index + 1}"

            _ax.plot(x, y, color=colour, label=label)
            _ax.fill_between(x, y, color=colour, alpha=0.2)

            _ax.set_title(f"Dimension {dim_index + 1}", loc="left")
            # _ax.set_xticks([])
            _ax.grid(True)

            # Disable ugly boxes
            for spine in _ax.spines.values():
                spine.set_visible(False)

    fig.axes[0].set_ylabel("Density")
    fig.axes[0].set_ylabel("Energy")
    # fig.axes[0].invert_yaxis()

    if auto_xlim:
        current_min = 5 * (math.floor(current_min / 5))
        current_max = 5 * (math.ceil(current_max / 5))
        xlim = [current_min, current_max]

    for i, _ax in enumerate(fig.axes):
        if xlim is not None:
            _ax.set_xlim(*xlim)

    plt.legend()

    plt.close(fig)

    if title is not None and not disable_title:
        fig.suptitle(title)

    if disable_title:
        fig.tight_layout()

    return (fig, None)
