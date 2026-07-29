import os
import sys
import argparse
from typing import Any, Dict, List

import pandas as pd

import export.sweeps
import export.parameters
import export.graphs
import export.render


def parse_kv_pair(s: str):
    """
    Helper to parse key=value strings into a dictionary
    """

    try:
        key, value = s.split("=", 1)
        return (key, value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"Filter must be in 'key=value' format: {s}")


def kv_pair_to_dict(pairs: List[Dict[str, str]]) -> Dict[str, str]:
    selected_parameters = {}
    for key, value in pairs:
        selected_parameters[key] = value

    return selected_parameters


parser = argparse.ArgumentParser(description="export - signed, sealed, delivered")
parser.add_argument("sweeps_dir", help="Directory where all sweeps are stored")
parser.add_argument("selected_sweep", type=str, help="Name of the sweep")
parser.add_argument(
    "--parameter_set",
    type=str,
    help="Specify the parameter set to export graphs for. Not required.",
    default=None,
)
parser.add_argument(
    "--filter",
    nargs="+",
    type=parse_kv_pair,
    help="Filter by parameter name and value. Can be used multiple times. Format: key=value",
)
parser.add_argument(
    "--aggregate", type=str, help="Aggregate over a specific parameter", default=None
)
parser.add_argument(
    "--overlay",
    nargs="+",
    action="append",
    type=parse_kv_pair,
    help="Manually add an overlay group to an aggregate graph. Can be used multiple times. Format: key=value",
)
parser.add_argument(
    "--step", type=int, help="Inspect the model at a specific step", default=None
)
parser.add_argument(
    "--legend_title",
    type=str,
    help="Set a legend title when aggregating over a specific parameter",
    default=None,
)
parser.add_argument(
    "--legend_labels",
    nargs="+",
    help="List of legend labels to override. Must match number of groups plotted.",
    default=None,
)
parser.add_argument(
    "export_dir", type=str, help="Directory where figures will be stored"
)
parser.add_argument(
    "output_profile", type=str, help="Name of the profile, will be output prefix"
)
parser.add_argument(
    "--disable_titles",
    action="store_true",
    help="Remove titles from the graphs",
    default=False,
)
args = parser.parse_args()

sweeps = export.sweeps.get_sweeps(args.sweeps_dir)
selected_sweep = args.selected_sweep
aggregate = args.aggregate
parameter_set = args.parameter_set
selected_step = None
if args.step is not None:
    selected_step = int(args.step)

selected_parameters = {}
if args.filter:
    selected_parameters = kv_pair_to_dict(args.filter)

overlays: List[Dict[str, str]] = []
if args.overlay:
    for overlay in args.overlay:
        overlays.append(kv_pair_to_dict(overlay))

combination_ids = None

run_infos = export.sweeps.get_run_infos(
    args.sweeps_dir, selected_sweep, hashable_safe=True
)
sweep_info = export.sweeps.get_sweep_info(args.sweeps_dir, selected_sweep)

# Get all parameter sets that span across runs in a sweep
if "parameter_set" in run_infos:
    parameter_sets = run_infos["parameter_set"].unique().tolist()

parameter_mapping, constants_mapping = export.parameters.build_mapping(run_infos)

if aggregate is not None:
    if aggregate in selected_parameters:
        selected_parameters = (
            export.parameters.remove_aggregate_parameter_from_selected(
                aggregate, selected_parameters
            )
        )

if parameter_set is not None:
    # Filter run_infos by selected parameter set
    if parameter_set not in parameter_sets:
        raise ValueError("Parameter set not in available parameter sets")

    selected_runs = run_infos[run_infos["parameter_set"] == parameter_set]
else:
    selected_runs = run_infos

selected_runs = export.parameters.find_eligible_runs(
    run_infos=selected_runs, selected_parameters=selected_parameters
)

if selected_runs.shape[0] == 0:
    raise ValueError("No runs found with the selected parameter combination")

unique_combination_ids = selected_runs["combination_id"].unique().tolist()
overlay_ids = None
if len(unique_combination_ids) > 1 and aggregate is None:
    raise ValueError(
        "Parameter selection does not single out a unique parameter combination"
    )
elif len(unique_combination_ids) > 1 and aggregate is not None:
    combination_ids = unique_combination_ids
    overlay_ids = []

    # Also get the unique IDs for overlays
    for overlay_idx, overlay in enumerate(overlays):
        overlay_runs = export.parameters.find_eligible_runs(
            run_infos=run_infos, selected_parameters=overlay
        )

        if overlay_runs.shape[0] == 0:
            raise ValueError(
                f"Overlay #{overlay_idx} does not single out a single parameter combination"
            )

        selected_overlays = overlay_runs["combination_id"].unique().tolist()
        overlay_ids += selected_overlays
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
    overlay_ids=overlay_ids,
    selected_run=None,
    selected_step=selected_step,
    exporting=True,
    disable_title=args.disable_titles,
    legend_title_override=args.legend_title,
    legend_labels_override=args.legend_labels,
)
