import math

import matplotlib.pyplot as plt

import visualisation
import visualisation.l1
import visualisation.meta
import visualisation.dimscrap
import visualisation.angle
import visualisation.shims

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
            lambda model, ax: visualisation.l1.make_communicative_success_macro_plot(model, ax=ax, smooth=False),
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