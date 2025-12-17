import matplotlib.pyplot as plt
import numpy as np
import math

from scipy.signal import savgol_filter
from visualisation.meta import formatter

def words_reduction_plot(model, show_all_words=False, ax=None):
    df = model.datacollector.get_model_vars_dataframe()
    matrix_3d = np.stack(df["words_zero_ratio"].to_numpy())

    if ax is None:
        ax = plt
        plt.ylim([0, 1])
    else:
        ax.set_ylim([0, 1])

    steps = math.floor(len(model.tokens) / 10)

    ax.plot(matrix_3d[::,::steps if not show_all_words else 1])
    chosen_word_indices = range(0, model.num_tokens, steps if not show_all_words else 1)
    legend_values = [ f"{model.tokens[chosen_word_index]} {model.ranks[chosen_word_index]}" for chosen_word_index in chosen_word_indices ]
    ax.legend(legend_values)

def words_reduction_plot_first_n(model, n=10, jitter_strength=0.02, ax=None):
    df = model.datacollector.get_model_vars_dataframe()
    matrix_3d = np.stack(df["words_zero_ratio"].to_numpy())

    if ax is None:
        ax = plt
        plt.ylim([0, 1])
    else:
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

def words_communication_success_first_n(model, n=10, attr="mean_success_per_token", jitter_strength=0.02, ax=None):
    df = model.datacollector.get_model_vars_dataframe()
    matrix_3d = np.stack(df[attr].to_numpy())

    if ax is None:
        ax = plt
        plt.ylim([0, 1])
    else:
        ax.set_ylim([0, 1])

    chosen_word_indices = range(0, n)
    legend_values = [ model.tokens[chosen_word_index] for chosen_word_index in chosen_word_indices ]

    # Get success for each token
    frequencies = np.array([model.frequencies[i] for i in chosen_word_indices])
    log_freq = np.log1p(frequencies)  # log1p to avoid log(0)
    log_freq = (log_freq - log_freq.min()) / (log_freq.max() - log_freq.min())  # Normalize between 0 and 1

    # Create colors (darker for more frequent)
    colors = [plt.cm.Blues(f) for f in log_freq]

    # Plot each word with its corresponding color
    for i, color in zip(chosen_word_indices, colors):
        window_length = 1000
        polyorder = 1
        
        y_smooth_success = savgol_filter(matrix_3d[:, i], window_length, polyorder)
        ax.plot(y_smooth_success, label=f"{model.tokens[i]} {model.ranks[i]}")

    ax.legend(legend_values)

def make_communication_plot(model, smooth=True, ax=None):
    df = model.datacollector.get_model_vars_dataframe()

    if ax is None:
        ax = plt
        plt.ylim([0, 1])
    else:
        ax.set_ylim([0, 1])
    
    if smooth:
        window_length = 500
        polyorder = 1
        
        y_smooth_success = savgol_filter(df["communicative_success"], window_length, polyorder)
        y_smooth_failure = savgol_filter(df["communicative_failure"], window_length, polyorder)
        
        ax.plot(y_smooth_success, color="green")
        ax.plot(y_smooth_failure, color="red")
    else:
        ax.plot(df["communicative_success"], color="green")
        ax.plot(df["communicative_failure"], color="red")

    ax.set_title("Global communicative success")
    ax.xaxis.set_major_formatter(lambda x, pos: formatter(x, pos, scale=model.datacollector_step_size))
    
    return ax

def make_communication_plot_combined(model, smooth=True, ax=None, disable_title=False):
    df = model.datacollector.get_model_vars_dataframe()

    if ax is None:
        ax = plt.gca()
        
    ax.set_ylim([0, 1])

    print(df["communicative_success"].iloc[-1])
    
    if smooth:
        window_length = 500
        polyorder = 1
        
        y_smooth_success = savgol_filter(df["communicative_success"], window_length, polyorder)
        y_smooth_success_macro = savgol_filter(df["communicative_success_macro"], window_length, polyorder)
        
        ax.plot(y_smooth_success, color="blue", linestyle="dashed")
        ax.plot(y_smooth_success_macro, color="blue", linestyle="dotted")
    else:
        ax.plot(df["communicative_success"], color="blue", linestyle="dashed", label="Micro-average")
        ax.plot(df["communicative_success_macro"], color="blue", linestyle="dotted", label="Macro-average")

    if not disable_title:
        title = f"Global communicative success"
        ax.set_title(title)

    plt.legend()
    ax.xaxis.set_major_formatter(lambda x, pos: formatter(x, pos, scale=model.datacollector_step_size))

    fig = ax.get_figure()
    if disable_title:
        fig.tight_layout()
    
    return ax

def make_repairs_plot(model, smooth=True, ax=None):
    df = model.datacollector.get_model_vars_dataframe()

    if ax is None:
        ax = plt
    
    if smooth:
        window_length = 100
        polyorder = 1
        
        y_smooth_repairs = savgol_filter(df["total_repairs"], window_length, polyorder)
        
        ax.plot(y_smooth_repairs, color="blue")
    else:
        ax.plot(df["total_repairs"], color="green")
    
    return ax

def make_communicative_success_probability_plot(model, smooth=True, ax=None):
    df = model.datacollector.get_model_vars_dataframe()

    if ax is None:
        ax = plt
        plt.ylim([0, 1])
    else:
        ax.set_ylim([0, 1])
    
    if smooth:
        window_length = 1000
        polyorder = 1
        
        y_smooth_success = savgol_filter(df["average_communicative_success_probability"], window_length, polyorder)
        
        ax.plot(y_smooth_success, color="blue")
    else:
        ax.plot(df["average_communicative_success_probability"], color="blue")

    ax.xaxis.set_major_formatter(lambda x, pos: formatter(x, pos, scale=model.datacollector_step_size))
    
    return ax

def make_words_distribution_plot(model, ax=None):
    df = model.datacollector.get_model_vars_dataframe()
    
    tokens_chosen = df["tokens_chosen"].iloc[-1]
    ranks = range(1, len(tokens_chosen) + 1)

    if ax is None:
        ax = plt
    
    ax.tick_params(axis='x', labelrotation=90, labelsize=8)
    ax.bar(ranks, tokens_chosen)
    
    return ax