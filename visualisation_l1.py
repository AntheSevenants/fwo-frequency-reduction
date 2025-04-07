import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.signal import savgol_filter

def make_mean_l1_plot(model, smooth=True, ax=None):
    df = model.datacollector.get_model_vars_dataframe()

    if ax is None:
        ax = plt
    
    if smooth:
        window_length = 100
        polyorder = 1
        
        y_smooth_repairs = savgol_filter(df["mean_agent_l1"], window_length, polyorder)
        
        ax.plot(y_smooth_repairs, color="blue")
    else:
        ax.plot(df["mean_agent_l1"], color="green")
    
    return ax

def make_l1_token_plot(model, show_all_words=False, ax=None):
    df = model.datacollector.get_model_vars_dataframe()
    matrix_3d = np.stack(df["mean_token_l1"].to_numpy())

    if ax is None:
        ax = plt
        # plt.ylim([0, 1])
    else:
        pass
        # ax.set_ylim([0, 1])

    steps = math.floor(len(model.tokens) / 10)

    ax.plot(matrix_3d[::,::steps if not show_all_words else 1])
    chosen_word_indices = range(0, model.num_tokens, steps if not show_all_words else 1)
    legend_values = [ f"{model.tokens[chosen_word_index]} {model.ranks[chosen_word_index]}" for chosen_word_index in chosen_word_indices ]
    ax.legend(legend_values)

def words_l1_plot_first_n(model, n=10, jitter_strength=0.02, ax=None):
    df = model.datacollector.get_model_vars_dataframe()
    matrix_3d = np.stack(df["mean_token_l1"].to_numpy())

    if ax is None:
        ax = plt
        # plt.ylim([0, 1])
    else:
        pass
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

def make_fail_reason_plot(model, ax=None):
    # Get the fail reason data from the data collector
    df = model.datacollector.get_model_vars_dataframe()
    # Turn it into a pandas dataframe
    fail_reason = pd.DataFrame.from_records(df["fail_reason"])
    # Group all data in groups of 100 steps
    grouped_df = fail_reason.groupby(np.arange(len(fail_reason)) // 100).sum()
    # Make percentual overview
    grouped_df = grouped_df.div(grouped_df.sum(axis=1), axis=0).multiply(100)

    if ax is None:
        fig, ax = plt.subplots()
    else:
        pass

    # Stacked bar plot
    grouped_df.plot(kind="bar", stacked=True, ax=ax)