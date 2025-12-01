import matplotlib.pyplot as plt

import export.aggregate.tools


def ratio_bar(parameter_mapping, ax, title, disable_title=False):
    return bar(
        parameter_mapping,
        ax,
        ylim=[0, 1],
        title=title,
        disable_title=disable_title
    )

def ratio_stem(parameter_mapping, ax, title, log=False, disable_title=False):
    return stem(
        parameter_mapping,
        ax,
        ylim=[0, 1],
        title=title,
        log=log,
        disable_title=disable_title
    )

def communicative_success(parameter_mapping, ax, disable_title=False):
    return ratio_stem(
        parameter_mapping,
        ax,
        "Communicative success across selected parameter",
        disable_title=disable_title
    )


def communicative_success_macro(parameter_mapping, ax, disable_title=False):
    return ratio_bar(
        parameter_mapping,
        ax,
        "Macro communicative success across selected parameter",
        disable_title=disable_title
    )

def mean_l1(parameter_mapping, ax, disable_title=False):
    return bar(
        parameter_mapping,
        ax,
        "Mean L1 across selected parameter",
        disable_title=disable_title
    )

def entropy(parameter_mapping, ax, disable_title=False):
    return ratio_stem(
        parameter_mapping,
        ax,
        "L1 entropy across elected parameter",
        disable_title=disable_title
    )

def bar(parameter_mapping, ax, ylim=None, title=None, disable_title=False):
    parameter_mapping = export.aggregate.tools.mean(parameter_mapping)

    x = list(parameter_mapping.keys())
    ax.bar(x, list(parameter_mapping.values()))

    if disable_title:
        title = None

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

def stem(parameter_mapping, ax, ylim=None, title=None, log=False, disable_title=False):
    parameter_mapping = export.aggregate.tools.mean(parameter_mapping)

    x = list(parameter_mapping.keys())
    
    if log:
        ax.set_xscale("log")
    ax.stem(x, list(parameter_mapping.values()))

    if disable_title:
        title = None

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
