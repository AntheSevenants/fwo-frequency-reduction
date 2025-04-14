import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.signal import savgol_filter
from visualisation.meta import formatter

def make_general_plot(model, attribute, smooth=True, ax=None, title=None, ratio=False):
    df = model.datacollector.get_model_vars_dataframe()

    if ax is None:
        ax = plt

        if ratio:
            plt.ylim([0, 1])
    else:
        if ratio:
            ax.set_ylim([0, 1])


    
    if smooth:
        window_length = 100
        polyorder = 1
        
        y_smooth_repairs = savgol_filter(df[attribute], window_length, polyorder)
        
        ax.plot(y_smooth_repairs, color="blue")
    else:
        ax.plot(df[attribute], color="green")

    if title is not None:
        ax.set_title(title)
    ax.xaxis.set_major_formatter(lambda x, pos: formatter(x, pos, scale=model.datacollector_step_size))
    
    return ax

def make_mean_l1_plot(model, smooth=True, ax=None):
    return make_general_plot(model, "mean_agent_l1", ax=ax, smooth=smooth, title="Mean L1 (across tokens, across agents)")

def make_communicative_success_macro_plot(model, smooth=True, ax=None):
    return make_general_plot(model, "communicative_success_macro", ax=ax, smooth=smooth, title="Global communicative success (macro avg across tokens)", ratio=True)

def property_plot_first_n(model, attribute, n=10, jitter_strength=0.2, ax=None, title=None, ratio=False):
    df = model.datacollector.get_model_vars_dataframe()
    matrix_3d = np.stack(df[attribute].to_numpy())

    if ax is None:
        ax = plt

        if ratio:
            plt.ylim([0, 1])
    else:
        if ratio:
            ax.set_ylim([0, 1])


    chosen_word_indices = range(0, n)
    legend_values = [ model.tokens[chosen_word_index] for chosen_word_index in chosen_word_indices ]

    # Get frequencies and normalize them
    frequencies = np.array([model.frequencies[i] for i in chosen_word_indices])
    log_freq = np.log1p(frequencies)  # log1p to avoid log(0)
    log_freq = (log_freq - log_freq.min()) / (log_freq.max() - log_freq.min())  # Normalize between 0 and 1

    # Create colors (darker for more frequent)
    colors = [plt.cm.Blues(f) for f in log_freq]

    # Plot each word with its corresponding color
    for i, color in zip(chosen_word_indices, colors):
        jitter = np.random.uniform(-jitter_strength, jitter_strength)
        ax.plot(matrix_3d[:, i] + jitter, color=color, label=f"{model.tokens[i]} {model.ranks[i]}")

    ax.legend(legend_values)

    ax.xaxis.set_major_formatter(lambda x, pos: formatter(x, pos, scale=model.datacollector_step_size))

    if title is not None:
        ax.set_title(title)

def words_l1_plot_first_n(model, n=10, jitter_strength=0.02, ax=None):
    property_plot_first_n(model, "mean_token_l1", n, jitter_strength, ax, "Mean L1 per token (across agents)")

def words_mean_exemplar_count_first_n(model, n=10, jitter_strength=0.02, ax=None):
    property_plot_first_n(model, "mean_exemplar_count", n, jitter_strength, ax, "Mean exemplar count per token (across agents)")

def communicative_success_first_n(model, n=10, jitter_strength=0.02, ax=None):
    property_plot_first_n(model, "success_per_token", n, jitter_strength, ax, "Mean communicative success per token (across agents)", ratio=True)

def token_good_origin_first_n(model, n=10, jitter_strength=0.02, ax=None):
    property_plot_first_n(model, "token_good_origin", n, jitter_strength, ax, "Ratio exemplars from non-confused interactions per token (across agents)", ratio=True)

def words_mean_exemplar_count_bar(model, ax=None):
    if ax is None:
        ax = plt
    else:
        pass

    frequency_counts = model.datacollector.get_model_vars_dataframe()["mean_exemplar_count"].iloc[-1]
    ax.bar(model.tokens, frequency_counts)   

    ax.set_title("Mean exemplar count per token (across agents)") 

def make_fail_reason_plot(model, ax=None):
    # Get the fail reason data from the data collector
    df = model.datacollector.get_model_vars_dataframe()
    # Turn it into a pandas dataframe
    fail_reason = pd.DataFrame.from_records(df["fail_reason"])

    # If we aggregate datacollector steps, correct for it here
    group_size = 100 if model.datacollector_step_size == 1 else 1

    # Group all data in groups of 100 steps
    grouped_df = fail_reason.groupby(np.arange(len(fail_reason)) // group_size).sum()
    # Make percentual overview
    grouped_df = grouped_df.div(grouped_df.sum(axis=1), axis=0).multiply(100)

    if ax is None:
        fig, ax = plt.subplots()
    else:
        pass

    # Stacked bar plot
    grouped_df.plot(kind="bar", stacked=True, ax=ax, title="Communication failure reason")
    
    ax.xaxis.set_major_formatter(lambda x, pos: formatter(x, pos, scale=model.datacollector_step_size))