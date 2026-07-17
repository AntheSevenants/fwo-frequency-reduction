import os
import argparse
import export.cache
import export.sweeps
import export.files
import export.render
import export.parameters
import export.graphs

import model.model_defaults

import pandas as pd

from flask import Flask, request, render_template, redirect, url_for, send_file

from typing import List, Optional, Union

PROFILE_NAME = "dashboard"

app = Flask(
    __name__, template_folder="dashboard/templates/", static_folder="dashboard/static/"
)


def url_add_param(endpoint, args, key, value):
    params = args.to_dict()
    params[key] = value
    return url_for(endpoint, **params)


# Register the filter
app.jinja_env.filters["url_add_param"] = url_add_param


@app.route("/live/")
def live():
    return show_interface(live=True)


@app.route("/")
def index():
    return show_interface()


# Live = are we looking at graphs from the jupyter notebook?
def show_interface(live: bool = False):
    # "sweep" = a complete batch run with multiple parameter combinations
    sweeps = export.sweeps.get_sweeps(args.sweeps_dir)
    # "selected sweep" = one of those batch runs
    selected_sweep = request.args.get("sweep")
    # You can filter for specific graphs
    selected_filter = request.args.get("filter")
    # You can filter for a specific run
    selected_run = request.args.get("run")

    # Aggregate parameter allows you to aggregate over multiple parameter combinations
    aggregate = request.args.get("aggregate")

    # Selected step is an extra parameter to inspect the model state at a specific step
    selected_step = request.args.get("step")
    if selected_step is not None:
        selected_step = int(selected_step)

    # Information about how the sweep was run (datacollector step size)
    sweep_info = None
    ticks = []

    # Combinatino of parameters selected
    selected_parameters = dict(request.args)
    parameter_mapping = None
    constants_mapping = None
    disable_selection = (
        False  # if only one combination exists, skip parameter selection
    )
    combination_ids = None  # the ID connected to the selected set of parameters
    cache_combination_id = None

    # Flag which indicates we have to aggregate over multiple models
    do_aggregate = False

    # Temp value
    graphs = []
    GRAPHS = []
    matched_run_ids = []

    # There are keywords used by the application, these do not appear as parameters
    # We filter to check whether the user has made an actual parameter selection
    no_selection = (
        len(list(set(selected_parameters) - set(export.parameters.RESERVED_KEYWORDS)))
        == 0
    )

    # If no filter is active, step from arguments and redirect
    if (
        selected_filter is None or selected_filter == "none"
    ) and selected_step is not None:
        del selected_parameters["step"]
        return redirect(url_for("index", _external=False, **selected_parameters))

    # If filter is active, but there is no step argument, set step to zero and redirect
    if (
        selected_filter is not None and selected_filter != "none"
    ) and selected_step is None:
        selected_parameters["step"] = "0"
        return redirect(url_for("index", _external=False, **selected_parameters))

    # Clear empty run selection
    if selected_run == "none":
        selected_run = None

    # Run selection logic
    if selected_sweep is not None:
        # Get information about all runs in the sweep as a  dataframe
        run_infos = export.sweeps.get_run_infos(
            args.sweeps_dir, selected_sweep, hashable_safe=True
        )

        sweep_info = export.sweeps.get_sweep_info(args.sweeps_dir, selected_sweep)
        # Set ticks for time expansion
        tick_step = round(int(sweep_info["num_steps"]) / 10)
        ticks = list(
            range(0, int(sweep_info["num_steps"]) + tick_step, tick_step),
        )

        parameter_mapping, constants_mapping = export.parameters.build_mapping(
            run_infos
        )

        if aggregate is not None:
            if aggregate in selected_parameters:
                selected_parameters = (
                    export.parameters.remove_aggregate_parameter_from_selected(
                        aggregate, selected_parameters
                    )
                )
                return redirect(
                    url_for("index", _external=False, **selected_parameters)
                )

        # If no parameter combination was made, create a parameter selection ourselves
        if no_selection:
            for parameter in parameter_mapping:
                selected_parameters[parameter] = parameter_mapping[parameter][0]

            if len(parameter_mapping) > 0:
                return redirect(
                    url_for("index", _external=False, **selected_parameters)
                )
            else:
                no_selection = False

        # These are runs that adhere to the parameter selection made
        selected_runs = export.parameters.find_eligible_runs(
            run_infos=run_infos, selected_parameters=selected_parameters
        )

        if selected_runs.shape[0] == 0:
            raise ValueError("No runs found with the selected parameter combination")

        unique_combination_ids = selected_runs["combination_id"].unique().tolist()
        if len(unique_combination_ids) > 1 and aggregate is None:
            raise ValueError(
                "Parameter selection does not single out a unique parameter combination"
            )
        elif len(unique_combination_ids) > 1 and aggregate is not None:
            combination_ids = unique_combination_ids
        else:
            combination_ids = unique_combination_ids[0]
            # Get the IDs of all runs which belong to the search results
            matched_run_ids = selected_runs["run_id"].unique().tolist()

            if selected_run is not None and selected_run != "none":
                if int(selected_run) not in matched_run_ids:
                    raise ValueError(
                        "Specified run filter does not belong to the selected parameter combination"
                    )

        # GRAPH TYPES
        if aggregate is None:
            GRAPHS = export.graphs.get_graph_names(
                export.graphs.GraphContext.DASHBOARD,
                is_single_run=selected_run is not None,
            )
        else:
            GRAPHS = export.graphs.get_aggregate_graph_names(
                export.graphs.GraphContext.DASHBOARD
            )

        # Filter logic (what graph should we show?)
        if selected_filter == "no":
            selected_filter = None
        elif selected_filter in GRAPHS:
            graphs = [selected_filter]
        else:
            selected_filter = None

        if selected_filter is None:
            graphs = GRAPHS.copy()

        # Cast as int to satisfy type check
        if selected_run is not None:
            selected_run = int(selected_run)

        export.render.prerender_profile_graphs(
            args.figures_dir,
            args.sweeps_dir,
            selected_sweep,
            combination_ids,
            graphs,
            aggregate_parameter=aggregate,
            selected_run=selected_run,
            selected_step=selected_step,
        )

        cache_combination_id = export.cache.get_cache_combination_id(combination_ids)

    if live:
        selected_sweep = "live"
        cache_combination_id = "live"
        live = True
        no_selection = False
        selected_run = -1

        GRAPHS = export.graphs.get_graph_names(
            export.graphs.GraphContext.DASHBOARD, is_single_run=True
        )
        graphs = GRAPHS

    return render_template(
        "index.html",
        sweeps=sweeps,
        selected_sweep=selected_sweep,
        combination_id=cache_combination_id,
        aggregate_parameter=aggregate,
        selected_parameters=selected_parameters,
        selected_filter=selected_filter,
        selected_step=selected_step,
        parameter_mapping=parameter_mapping,
        constants_mapping=constants_mapping,
        live=live,  # opus
        no_selection=no_selection,
        graphs=graphs,
        all_graphs=GRAPHS,
        runs=matched_run_ids,
        selected_run=selected_run,
        sweep_info=sweep_info,
        ticks=ticks,
        get_enum_name=get_enum_name,
        enum_mapping=model.model_defaults.PARAMETER_ENUM_MAPPING,
    )


@app.route(
    "/graph/<string:selected_sweep>/<string:combination_id>/<string:single_run_id>/<string:selected_step>/<string:graph_name>"
)
def send_single_run_graph(
    graph_name: str,
    selected_sweep: str,
    combination_id: str,
    single_run_id: str,
    selected_step: str,
):
    return send_graph(
        graph_name,
        selected_sweep,
        combination_id,
        single_run_id=single_run_id,
        selected_step=selected_step,
    )


@app.route(
    "/graph/<string:selected_sweep>/<string:combination_id>/<string:selected_step>/<string:graph_name>"
)
def send_combination_graph(
    graph_name: str, selected_sweep: str, combination_id: str, selected_step: str
):
    return send_graph(
        graph_name, selected_sweep, combination_id, selected_step=selected_step
    )


def send_graph(
    graph_name, selected_sweep, combination_id, single_run_id=None, selected_step=None
):
    # Live graphs live in the same folder always, so we do not need to compute where to find them
    if selected_sweep == "live" and combination_id == "live":
        temp_models_figures_dir = args.figures_dir_live
        profile = "jupyter"
    else:
        if selected_step == "_":
            selected_step = None

        # Where our figures are stored for this parameter combination
        temp_models_figures_dir = export.cache.make_temp_runs_figures_dir(
            selected_sweep,
            combination_id,
            args.figures_dir,
            single_run_id=single_run_id,
            selected_step=selected_step,
        )
        profile = PROFILE_NAME

    # Figure filename
    figure_filename = export.files.get_figure_filename(profile, graph_name)
    graph_path = os.path.join(temp_models_figures_dir, figure_filename)

    return send_file(graph_path, mimetype="image/png")


# From Le Chat
def get_enum_name(attribute: str, value: str):
    cls = model.model_defaults.PARAMETER_ENUM_MAPPING[attribute]

    # Get all attributes of the provided class
    attributes = [
        (name, getattr(cls, name)) for name in dir(cls) if not name.startswith("__")
    ]

    # Create a mapping of values to names
    enum_mapping = {
        str(value): name for name, value in attributes if isinstance(value, int)
    }

    # Return the corresponding enum name or "Unknown" if not found
    return enum_mapping.get(value, "Unknown")


parser = argparse.ArgumentParser(description="dashboard - what's cooking?")
parser.add_argument("sweeps_dir", help="Directory where all sweeps are stored")
parser.add_argument("figures_dir", help="Directory where figures will be stored")
parser.add_argument("figures_dir_live", help="Directory where live figures are stored")
args = parser.parse_args()

app.run(debug=True, port=8080, host="0.0.0.0")
