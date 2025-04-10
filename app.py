
import solara
import umap
import numpy as np

from mesa.visualization import SolaraViz, make_plot_component, make_space_component
from mesa.visualization.utils import update_counter
from matplotlib.figure import Figure

from model.model import ReductionModel
from model.helpers import load_vectors

NUM_AGENTS = 100
vectors, tokens, frequencies, percentiles = load_vectors("materials/vectors.txt")

model_params = {
    "vectors": vectors,
    "tokens": tokens,
    "frequencies": frequencies,
    "percentiles": percentiles,
    "num_agents": {
        "type": "SliderInt",
        "value": 50,
        "label": "Number of agents:",
        "min": 10,
        "max": 100,
        "step": 1,
    },
    "show_all_words": {
        "type": "Checkbox",
        "value": False,
        "label": "Plot all words"
    },
    "zipfian_token_distribution": {
        "type": "Checkbox",
        "value": True,
        "label": "Choose words based on Zipfian distribution"
    }
}

def agent_portrayal(agent):
    portrayal = {
        "color": "tab:grey",
        "size": 50,
    }

    if agent.speaking or agent.hearing:
        portrayal["color"] = "tab:green"

    if agent.hearing and not agent.model.turns[-1]:
        portrayal["color"] = "tab:red"

    # agent.speaking = False
    # agent.hearing = False

    return portrayal

def make_dark(ax):
    ax.set_facecolor('#292929')
    
    ax.spines['bottom'].set_color('white')
    ax.spines['top'].set_color('white')
    ax.spines['right'].set_color('white')
    ax.spines['left'].set_color('white')
    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')

@solara.component
def words_reduction_plot(model):
    update_counter.get()
    fig = Figure()
    ax = fig.subplots()
    df = model.datacollector.get_model_vars_dataframe()

    if len(df.words_zero_ratio.value_counts()) > 0:
        matrix_3d = np.stack(df["words_zero_ratio"].to_numpy())

        # If showing all words, plot all words. If not, plot only every ten
        ax.plot(matrix_3d[::100,::10 if not model.show_all_words else 1])

    ax.set_ylabel("words_zero_ratio")
    ax.set_ylim([0, 1])

    chosen_word_indices = range(0, model.num_tokens, 10 if not model.show_all_words else 1)
    legend_values = [ model.tokens[chosen_word_index] for chosen_word_index in chosen_word_indices ]
    ax.legend(legend_values)

    solara.FigureMatplotlib(fig, bbox_inches="tight")

@solara.component
def make_words_distribution_plot(model):
    #update_counter.get()
    fig = Figure()
    ax = fig.subplots()
    ax.tick_params(axis='x', labelrotation=90)
    ax.bar(model.tokens, model.frequencies)

    solara.FigureMatplotlib(fig, bbox_inches="tight")

@solara.component
def make_vector_plot(model):
    umap_model = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=model.seed)
    coords = umap_model.fit_transform(np.asarray(model.agents[0].vocabulary))

    update_counter.get()
    fig = Figure()
    ax = fig.subplots()
    ax.set_ylim([0, 1])
    ax.scatter(coords[:,0], coords[:,1])

    solara.FigureMatplotlib(fig, bbox_inches="tight")

@solara.component
def make_communication_plot(model):
    update_counter.get()
    df = model.datacollector.get_model_vars_dataframe()
    fig = Figure()
    ax = fig.subplots()

    #make_dark(ax)
    
    ax.plot(df["communicative_success"], color="green")
    ax.plot(df["communicative_failure"], color="red")
    
    solara.FigureMatplotlib(fig, bbox_inches="tight")

# Initial model instance
model = ReductionModel(NUM_AGENTS, vectors, tokens, frequencies, percentiles, reduction_prior=1)

SpaceGraph = make_space_component(agent_portrayal, post_process=make_dark)

page = SolaraViz(
    model,
    components=[SpaceGraph, make_communication_plot],
    model_params=model_params,
    name="FWO Frequency Reduction Model"
)
page