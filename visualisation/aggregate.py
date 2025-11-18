import matplotlib.pyplot as plt

import export.aggregate.tools


def ratio_bar(parameter_mapping, ax, title):
    return bar(
        parameter_mapping,
        ax,
        ylim=[0, 1],
        title=title,
    )

def communicative_success(parameter_mapping, ax):
    return ratio_bar(
        parameter_mapping, ax, "Communicative success across selected parameter"
    )


def communicative_success_macro(parameter_mapping, ax):
    return ratio_bar(
        parameter_mapping, ax, "Macro communicative success across selected parameter"
    )

def mean_l1(parameter_mapping, ax):
    return bar(
        parameter_mapping, ax, "Mean L1 across selected parameter"
    )


def bar(parameter_mapping, ax, ylim=None, title=None):
    parameter_mapping = export.aggregate.tools.mean(parameter_mapping)

    x = list(parameter_mapping.keys())
    ax.bar(x, list(parameter_mapping.values()))

    # Make sure string x ticks get rendered correctly
    if type(x[0]) == str:
        plt.xticks(x, x)

    if ylim is not None:
        ax.set_ylim(ylim)

    if title is not None:
        ax.set_title(title)

    fig = ax.get_figure()
    if title is None:
        fig.tight_layout()

    return ax
