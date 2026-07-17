import os
import argparse

import pandas as pd

import export.sweeps
import export.parameters
import export.graphs
import export.render

parser = argparse.ArgumentParser(description="export - signed, sealed, delivered")
parser.add_argument("sweeps_dir", help="Directory where all sweeps are stored")
parser.add_argument("selected_sweep", type=str, help="Name of the sweep")
parser.add_argument(
    "--filter",
    action="append",
    help="Filter by parameter name and value. Can be used multiple times. Format: key=value",
)
parser.add_argument(
    "--aggregate", type=str, help="Aggregate over a specific parameter", default=None
)
parser.add_argument(
    "--step", type=int, help="Inspect the model at a specific step", default=None
)
parser.add_argument(
    "export_dir", type=str, help="Directory where figures will be stored"
)
parser.add_argument(
    "output_profile", type=str, help="Name of the profile, will be output prefix"
)
parser.add_argument(
    "--disable_titles", action="store_true", help="Remove titles from the graphs"
)
args = parser.parse_args()

sweeps = export.sweeps.get_sweeps(args.sweeps_dir)
selected_sweep = args.selected_sweep
aggregate = args.aggregate
selected_step = None
if args.step is not None:
    selected_step = int(args.step)

selected_parameters = {}
if args.filter:
    for item in args.filter:
        if "=" in item:
            k, v = item.split("=", 1)
            selected_parameters[k] = v
        else:
            print(f"Warning: Skipping invalid filter '{item}'")

combination_ids = None

run_infos = export.sweeps.get_run_infos(
    args.sweeps_dir, selected_sweep, hashable_safe=True
)
sweep_info = export.sweeps.get_sweep_info(args.sweeps_dir, selected_sweep)

parameter_mapping, constants_mapping = export.parameters.build_mapping(run_infos)

if aggregate is not None:
    if aggregate in selected_parameters:
        selected_parameters = (
            export.parameters.remove_aggregate_parameter_from_selected(
                aggregate, selected_parameters
            )
        )

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

if aggregate is None:
    GRAPHS = export.graphs.get_graph_names(
        export.graphs.GraphContext.EXPORT,
        is_single_run=False,
    )
else:
    GRAPHS = export.graphs.get_aggregate_graph_names(export.graphs.GraphContext.EXPORT)

export.render.prerender_profile_graphs(
    args.export_dir,
    args.sweeps_dir,
    selected_sweep,
    combination_ids,
    GRAPHS,
    args.output_profile,
    aggregate_parameter=aggregate,
    selected_run=None,
    selected_step=selected_step,
)
