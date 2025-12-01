import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import entropy

import export.models
import export.aggregate.mapping

import visualisation.aggregate

AGGREGATE_GRAPHS = ["success-graph", "success-graph-macro"]


def get_aggregate_graph_names():
    return AGGREGATE_GRAPHS


def generate_graphs(
    selected_run,
    selected_model_ids,
    selected_models,
    aggregate_parameter,
    runs_dir,
    graphs,
    disable_title=False,
):
    # First, load the models required
    datacollector_dataframes = export.models.get_datacollector_dataframes(
        runs_dir, selected_run=selected_run, selected_model_ids=selected_model_ids
    )

    # Now, we can build the desired graphs and save them
    graphs_output = {}

    for graph_name in graphs:
        figure = create_graph(
            graph_name,
            datacollector_dataframes,
            selected_model_ids,
            selected_models,
            aggregate_parameter,
            disable_title=disable_title,
        )

        graphs_output[graph_name] = figure

    return graphs_output


def create_graph(
    graph_name,
    datacollector_dataframes,
    selected_model_ids,
    selected_models,
    aggregate_parameter,
    disable_title=False,
):
    fig, ax = plt.subplots()

    get_mapping = (
        lambda outcome_variable: export.aggregate.mapping.get_parameter_value_mapping(
            datacollector_dataframes,
            selected_model_ids,
            selected_models,
            aggregate_parameter,
            outcome_variable,
        )
    )

    if graph_name == "success-graph":
        parameter_mapping = get_mapping("communicative_success")
        figure = visualisation.aggregate.communicative_success(
            parameter_mapping,
            ax,
            disable_title=disable_title
        )
    elif graph_name == "success-graph-macro":
        parameter_mapping = get_mapping("communicative_success_macro")
        figure = visualisation.aggregate.communicative_success_macro(
            parameter_mapping,
            ax,
            disable_title=disable_title
        )
    elif graph_name == "l1-entropy":
        parameter_mapping = get_mapping("mean_token_l1")
        for parameter in parameter_mapping:
            parameter_mapping[parameter] = [
                entropy(
                    np.divide(l1_per_construction,
                              np.sum(l1_per_construction)),
                    base=l1_per_construction.shape[0])
                for l1_per_construction
                in parameter_mapping[parameter]
            ]
            parameter_mapping[parameter] = np.mean(parameter_mapping[parameter])

        figure = visualisation.aggregate.entropy(parameter_mapping, ax)
    else:
        raise ValueError(f"Unrecognised aggregate graph: {graph_name}")

    return figure
