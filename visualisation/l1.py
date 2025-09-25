import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from scipy.signal import savgol_filter
from visualisation.meta import formatter

def make_general_plot(model, attribute, smooth=True, ax=None, title=None, ratio=False, disable_title=False):
    df = model.datacollector.get_model_vars_dataframe()

    if ax is None:
        ax = plt

        if ratio:
            plt.ylim([0, 1])
    else:
        if ratio:
            ax.set_ylim([0, 1])

    fig = ax.get_figure()
    if disable_title:
        fig.tight_layout()
    
    if smooth:
        window_length = 100
        polyorder = 1
        
        y_smooth_repairs = savgol_filter(df[attribute], window_length, polyorder)
        
        ax.plot(y_smooth_repairs, color="blue")
    else:
        ax.plot(df[attribute], color="green")

    if title is not None and not disable_title:
        ax.set_title(title)
    ax.xaxis.set_major_formatter(lambda x, pos: formatter(x, pos, scale=model.datacollector_step_size))

    return ax

def make_mean_l1_plot(model, smooth=True, ax=None, disable_title=False):
    return make_general_plot(model, "mean_agent_l1", ax=ax, smooth=smooth, title="Mean L1 (across tokens, across agents)", disable_title=disable_title)

def make_communicative_success_macro_plot(model, smooth=True, ax=None, disable_title=False):
    return make_general_plot(model, "communicative_success_macro", ax=ax, smooth=smooth, title="Global communicative success (macro avg across tokens)", ratio=True, disable_title=disable_title)

def make_mean_exemplar_age_plot(model, smooth=True, ax=None, disable_title=False):
    return make_general_plot(model, "mean_exemplar_age", ax=ax, smooth=smooth, title="Mean exemplar age (macro avg across agents)", disable_title=disable_title)

def make_reduction_success_plot(model, smooth=True, ax=None, disable_title=False):
    return make_general_plot(model, "reduction_success", ax=ax, smooth=smooth, title="Global reduction success ratio", ratio=True, disable_title=disable_title)

def property_plot_first_n(model, attribute, n=10, jitter_strength=0.2, ax=None, title=None, ratio=False, disable_title=False):
    df = model.datacollector.get_model_vars_dataframe()
    matrix_3d = np.stack(df[attribute].to_numpy())

    if ax is None:
        ax = plt.gca()

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

    if title is not None and not disable_title:
        ax.set_title(title)

    fig = ax.get_figure()
    if disable_title:
        fig.tight_layout()

    return ax

def words_l1_plot_first_n(model, n=10, jitter_strength=0.02, ax=None, disable_title=False):
    return property_plot_first_n(model, "mean_token_l1", n, jitter_strength, ax, "Mean L1 per token (across agents)", disable_title=disable_title)

def words_mean_exemplar_count_first_n(model, n=10, jitter_strength=0.02, ax=None, disable_title=False):
    return property_plot_first_n(model, "mean_exemplar_count", n, jitter_strength, ax, "Mean exemplar count per token (across agents)", disable_title=disable_title)

def communicative_success_first_n(model, n=10, jitter_strength=0.02, ax=None, disable_title=False):
    return property_plot_first_n(model, "success_per_token", n, jitter_strength, ax, "Mean communicative success per token (across agents)", ratio=True, disable_title=disable_title)

def token_good_origin_first_n(model, n=10, jitter_strength=0.02, ax=None, disable_title=False):
    return property_plot_first_n(model, "token_good_origin", n, jitter_strength, ax, "Ratio of authentic exemplars per token (across agents)", ratio=True, disable_title=disable_title)

def words_mean_exemplar_count_bar(model, ax=None, disable_title=False):
    if ax is None:
        ax = plt
    else:
        pass

    frequency_counts = model.datacollector.get_model_vars_dataframe()["mean_exemplar_count"].iloc[-1]
    ax.bar(model.tokens, frequency_counts)   

    if not disable_title:
        ax.set_title("Mean exemplar count per token (across agents)")

    fig = ax.get_figure()
    if disable_title:
        fig.tight_layout()

    return ax

def words_mean_l1_bar(model, step, ax=None, disable_title=False):
    if ax is None:
        ax = plt
        no_ax = True
    else:
        no_ax = False
        
    if model.datacollector_step_size != 1:
        step = step // model.datacollector_step_size

    token_l1 = model.datacollector.get_model_vars_dataframe()["mean_token_l1"].iloc[step]
    ax.bar(model.tokens, token_l1)

    x_tokens = [ str(i) if i % 10 == 0 or i == 1 else "" for i in range(1, len(model.tokens)) ]
    if not no_ax:
        ax.set_xticklabels(x_tokens, rotation=90)
    else:
        plt.xticks(x_tokens, rotation=90)

    step = step * model.datacollector_step_size
    title = f"Mean L1 per token across agents (t = {step})"
    
    if not disable_title:
        if not no_ax:
            ax.set_title(title) 
            ax.set_ylim([0, 7000])
        else:
            ax.title(title)

    fig = ax.get_figure()
    if disable_title:
        fig.tight_layout()

    return ax

def make_fail_reason_plot(model, include_success=False, ax=None):
    # Get the fail reason data from the data collector
    df = model.datacollector.get_model_vars_dataframe()
    # Turn it into a pandas dataframe
    if not include_success:
        fail_reason = pd.DataFrame.from_records(df["fail_reason"])
    else:
        fail_reason = pd.DataFrame.from_records(df["outcomes"])

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
    if not include_success:
        title = "Communication failure reason"
    else:
        title = "Communication outcomes"
    grouped_df.plot(kind="bar", stacked=True, ax=ax, title=title)
    
    ax.xaxis.set_major_formatter(lambda x, pos: formatter(x, pos, scale=model.datacollector_step_size))

    return ax

def compute_half_time(model, step):
    if model.datacollector_step_size != 1:
        step = step // model.datacollector_step_size
    
    l1_values = model.datacollector.get_model_vars_dataframe()["mean_token_l1"]

    token_l1_start = l1_values.iloc[0]
    half_level = 0.5 * token_l1_start

    half_level_times = np.full(model.num_tokens, np.nan)
    max_step = model.current_step // model.datacollector_step_size

    for i in range(model.num_tokens):
        search_step = 0
        while search_step < max_step:
            if l1_values.iloc[search_step][i] <= half_level[i]:
                half_level_times[i] = search_step * model.datacollector_step_size
                break

            search_step += 1

    return half_level_times

def half_time_bar(model, step, ax=None, disable_title=False):
    if ax is None:
        ax = plt
        no_ax = True
    else:
        no_ax = False
    
    half_level_times = compute_half_time(model, step)

    ax.bar(model.tokens, half_level_times)

    x_tokens = [ str(i) if i % 10 == 0 or i == 1 else "" for i in range(1, len(model.tokens)) ]
    if not no_ax:
        ax.set_xticklabels(x_tokens, rotation=90)
    else:
        plt.xticks(x_tokens, rotation=90)
        
    title = f"Mean L1 half life across agents (t = {step})"
    if not disable_title:
        if not no_ax:
            ax.set_title(title) 
        else:
            ax.title(title)

    fig = ax.get_figure()
    if disable_title:
        fig.tight_layout()

    return ax