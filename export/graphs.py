import math

import matplotlib.pyplot as plt

import export.runs

import visualisation
import visualisation.l1
import visualisation.meta
import visualisation.dimscrap
import visualisation.angle
import visualisation.shims

ANALYSIS_REGULAR_GRAPHS = [ "mosaic_1", "mosaic_2", "confusion_mosaic", "l1_plot" ]
ANALYSIS_NON_LIGHT_SERIALISATION_GRAPHS = [ "umap_mosaic", "memory_mosaic" ]
ANALYSIS_TOROIDAL_GRAPHS = [ "angle_vocabulary_plot_2d_moisaic", "angle_vocabulary_plot_3d" ]

EXPORT_REGULAR_GRAPHS = [ "l1-general", "l1-per-construction", "success", "matrix", "confusion-ratio", "half-life-per-construction" ]
EXPORT_TOROIDAL_GRAPHS = [ "angle-vocabulary-plot-2d-begin", "angle-vocabulary-plot-2d-end", "angle-vocabulary-plot-3d-begin" ]

def generate_graphs(selected_run, selected_model_ids, selected_models, runs_dir, token_infos, graphs):
    datacollector_dataframes = export.models.get_datacollector_dataframes(runs_dir, selected_run=selected_run, selected_model_ids=selected_model_ids)

    # TODO implement min_steps
    # Build an aggregate model from all the different datacollector dataframes
    # Then we can build one beautiful big graph
    model = visualisation.shims.Model(datacollector_dataframes, selected_models.iloc[0], token_infos, min_steps=None)

    # Now, we can build the desired graphs and save them
    graphs_output = {}

    # TODO change 'n'
    for graph_name in graphs:
        figure = export.graphs.create_graph(graph_name=graph_name, model=model, n=35, ylim=model.value_ceil * model.num_dimensions, disable_title=False)

        graphs_output[graph_name] = figure

    return graphs_output

def create_graph(graph_name, model, disable_title, n=35, ylim=7000):
    fig, ax = plt.subplots()

    if graph_name == "l1-general":
        figure = visualisation.l1.make_mean_l1_plot(model, ax=ax, smooth=False, disable_title=disable_title)
    elif graph_name == "l1-per-construction-mosaic":
        figure = visualisation.meta.make_layout_plot(model, visualisation.l1.words_mean_l1_bar,
                                                    steps=[math.floor(model.current_step / 4) * 1,
                                                        math.floor(model.current_step / 4) * 2,
                                                        math.floor(model.current_step / 4) * 3,
                                                        model.current_step],
                                                    disable_title=disable_title)
    elif graph_name == "l1-per-construction":
        figure = visualisation.l1.words_mean_l1_bar(model, model.current_step, ax=ax, disable_title=disable_title)
    elif  graph_name == "success":
        figure = visualisation.dimscrap.make_communication_plot_combined(model, smooth=False, ax=ax, disable_title=disable_title)
    elif graph_name == "matrix":
        figure = visualisation.meta.make_confusion_plot(model, model.current_step, n=n, ax=ax, disable_title=disable_title)
    elif graph_name == "matrix-mosaic":
        figure = visualisation.meta.make_layout_plot(model, visualisation.meta.make_confusion_plot, n=n,
                                                     steps=[math.floor(model.current_step / 4) * 1,
                                                        math.floor(model.current_step / 4) * 2,
                                                        math.floor(model.current_step / 4) * 3,
                                                        model.current_step],
                                                    disable_title=disable_title)
    elif graph_name == "confusion-ratio":
        figure = visualisation.l1.token_good_origin_first_n(model, ax=ax, disable_title=disable_title)
    elif graph_name in [ "angle-vocabulary-plot-2d-begin", "angle-vocabulary-plot-2d-end" ]:
        if graph_name == "angle-vocabulary-plot-2d-begin":
            step = 0
        elif graph_name == "angle-vocabulary-plot-2d-end":
            step = 700

        figure = visualisation.angle.make_angle_vocabulary_plot_2d(model, step, agent_filter=0, disable_title=disable_title)
    elif graph_name == "angle-vocabulary-plot-3d-begin":
        step = 0

        figure = visualisation.angle.make_angle_vocabulary_plot_3d(model, step, model.num_tokens, agent_filter=0, disable_title=disable_title)
    elif graph_name == "half-life-per-construction":
        figure = visualisation.l1.half_time_bar(model, model.current_step, ax=ax, disable_title=disable_title)
    elif graph_name == "mosaic_1":
        figure = visualisation.meta.combine_plots(model,
            lambda model, ax: visualisation.dimscrap.make_communication_plot_combined(model, ax=ax, smooth=False),
            visualisation.l1.communicative_success_first_n,
            lambda model, ax: visualisation.l1.make_reentrance_ratio_plot(model, ax=ax, smooth=False),
            lambda model, ax: visualisation.l1.token_good_origin_first_n(model, ax=ax),
            lambda model, ax: visualisation.l1.make_mean_exemplar_age_plot(model, ax=ax, smooth=False),
            lambda model, ax: visualisation.l1.make_reduction_success_plot(model, ax=ax, smooth=False),
            )
    elif graph_name == "mosaic_2":
        figure = visualisation.meta.combine_plots(model,
            lambda model, ax: visualisation.l1.make_fail_reason_plot(model, ax=ax, include_success=True),
            visualisation.l1.words_l1_plot_first_n,
            lambda model, ax: visualisation.l1.make_mean_l1_plot(model, ax=ax, smooth=False),
            lambda model, ax: visualisation.dimscrap.make_communication_plot(model, ax=ax, smooth=False),
            visualisation.l1.words_mean_exemplar_count_first_n,
            visualisation.l1.words_mean_exemplar_count_bar)
    elif graph_name == "confusion_mosaic":
        figure = visualisation.meta.make_layout_plot(model,
                                    visualisation.meta.make_confusion_plot,
                                    n=n, steps=[math.floor(model.current_step / 4) * 1,
                                                 math.floor(model.current_step / 4) * 2,
                                                 math.floor(model.current_step / 4) * 3, model.current_step])
    elif graph_name == "l1_plot":
        figure = visualisation.meta.make_layout_plot(model,
                                    visualisation.l1.words_mean_l1_bar,
                                    steps=[math.floor(model.current_step / 4) * 1,
                                                 math.floor(model.current_step / 4) * 2,
                                                 math.floor(model.current_step / 4) * 3, model.current_step],
                                    ylim=ylim)
    elif graph_name == "umap_mosaic":
        figure = visualisation.meta.make_layout_plot(model,
                                    visualisation.meta.make_umap_plot,
                                    steps=[math.floor(model.current_step / 4) * 1,
                                           math.floor(model.current_step / 4) * 2,
                                           math.floor(model.current_step / 4) * 3,
                                           model.current_step])
    else:
        raise ValueError(f"Unrecognised graph: {graph_name}")
    
    return figure

def get_analysis_graph_names(toroidal=False):
    if not toroidal:
        return ANALYSIS_REGULAR_GRAPHS
    else:
        return ANALYSIS_REGULAR_GRAPHS + ANALYSIS_TOROIDAL_GRAPHS