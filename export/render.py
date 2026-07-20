from typing import List, Union

import export.cache
import export.files
import export.graphs


def prerender_profile_graphs(
    figures_dir: str,
    sweeps_dir: str,
    selected_sweep: str,
    combination_ids: Union[int, List[int]],
    graphs: List[str],
    PROFILE_NAME: str,
    aggregate_parameter: str | None = None,
    selected_run: int | None = None,
    selected_step: int | None = None,
    disable_title: bool = False,
    legend_title_override: str | None = None,
    legend_labels_override: List[str] | None = None,
    exporting: bool = False,
) -> None:
    if selected_run is not None and aggregate_parameter is not None:
        raise ValueError(
            "Single run cannot be isolated if aggregate parameter is defined"
        )

    cache_combination_id = export.cache.get_cache_combination_id(combination_ids)

    # Get cached graphs
    if not exporting:
        cached_graphs = export.cache.get_cached_graphs(
            selected_sweep,
            cache_combination_id,
            graphs,
            PROFILE_NAME,
            figures_dir,
            single_run_id=selected_run,
            selected_step=selected_step,
        )
        non_cached_graph_count = len(list(set(graphs) - set(cached_graphs)))
    else:
        non_cached_graph_count = -1

    if non_cached_graph_count == 0:
        pass
    # If we still need some graphs, just build all of them again
    else:
        # Generate the directory where we will put the figures
        if not exporting:
            temp_models_figures_dir = export.cache.make_temp_runs_figures_dir(
                selected_sweep,
                cache_combination_id,
                figures_dir,
                single_run_id=selected_run,
                selected_step=selected_step,
            )
        else:
            temp_models_figures_dir = figures_dir

        # All graphs in a dict representation
        # Create profile graphs
        if aggregate_parameter is None:
            aggregate_settings = None
        # Else, create aggregate graphs
        else:
            if isinstance(combination_ids, list):
                aggregate_settings = export.graphs.AggregateSettings(
                    sweeps_dir,
                    selected_sweep,
                    combination_ids,
                    aggregate_parameter,
                )
            else:
                raise ValueError(
                    "Cannot aggregate with only one combination of parameters"
                )

        graphs_output = export.graphs.generate_graphs(
            sweeps_dir,
            selected_sweep,
            combination_ids,
            graphs,
            single_run=selected_run,
            selected_step=selected_step,
            aggregate=aggregate_settings,
            disable_title=disable_title,
            legend_title=legend_title_override,
            legend_labels=legend_labels_override,
        )

        # Save the files to disk!
        export.files.export_files(graphs_output, PROFILE_NAME, temp_models_figures_dir)
