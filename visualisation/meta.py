import numpy as np
import matplotlib.pyplot as plt

from umap import UMAP

def combine_plots(model, ax1_func, ax2_func, ax3_func, ax4_func, ax5_func, ax6_func):
    fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2, figsize=(12, 12))

    # Call the first plotting function and get the axes
    ax1_func(model, ax=ax1)
    ax1 = plt.gca()  # Get the current axes

    # Call the second plotting function and get the axes
    ax2_func(model, ax=ax2)
    ax2 = plt.gca() # Get the current axes

    # Call the third plotting function and get the axes
    ax3_func(model, ax=ax3)
    ax3 = plt.gca()  # Get the current axes

    # Call the fourth plotting function and get the axes
    ax4_func(model, ax=ax4)
    ax4 = plt.gca()  # Get the current axes

    # Call the fifth plotting function and get the axes
    ax5_func(model, ax=ax5)
    ax5 = plt.gca()  # Get the current axes

    # Call the sixth plotting function and get the axes
    ax6_func(model, ax=ax6)
    ax6 = plt.gca()  # Get the current axes

    # Adjust layout
    plt.tight_layout()
    plt.show()

def make_layout_plot(model, plot_function, steps=[100, 1000, 5000, 10000], **kwargs):
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 12))

    plot_function(model, step=steps[0] - 1, ax=ax1, **kwargs)
    plot_function(model, step=steps[1] - 1, ax=ax2, **kwargs)
    plot_function(model, step=steps[2] - 1, ax=ax3, **kwargs)
    plot_function(model, step=model.steps - 1, ax=ax4, **kwargs)

def make_confusion_plot(model, step, n=35, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(15,10))

    df = model.datacollector.get_model_vars_dataframe()
    confusion_matrix = df["confusion_matrix"].iloc[step]

    row_sums = confusion_matrix.sum(axis=1, keepdims=True)
    normalized_confusion_matrix = confusion_matrix / row_sums
    ax.matshow(normalized_confusion_matrix[0:n,0:n])

    labels = model.tokens[0:n]

    ax.set_xticks(range(0, n))
    ax.set_yticks(range(0, n))
    ax.tick_params(axis='x', labelrotation=90)
    
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)

    ax.set_title(f"Confusion matrix (t={step})", y=0.92, color="white")

    return ax

def make_umap_plot_inner(vocabulary, percentiles, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(5,5))

    umap_2d = UMAP(n_components=2, init='random', random_state=0)
    proj_2d = umap_2d.fit_transform(np.asarray(vocabulary))
    x,y = zip(*proj_2d)
    ax.scatter(x, y, c=percentiles, cmap='gray')

def make_umap_plot(model, step, ax=None):
    df = model.datacollector.get_model_vars_dataframe()
    if step < 0:
        vocabulary = model.vectors
    else:
        vocabulary = df["average_vocabulary"].iloc[step]

    make_umap_plot_inner(model, vocabulary, model.percentiles, ax)