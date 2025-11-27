import os
import argparse

import pandas as pd
from pathlib import Path

import export.runs
import export.profiles
import export.parameters
import export.graphs
import export.files

FIGURES_DIR = "figures/"
RUNS_DIR = "models/"

parser = argparse.ArgumentParser(
    description='export - make graphs for the reduction model')
parser.add_argument('selected_run', type=str, help='name of the run')
parser.add_argument('profile', type=str,
                    help='all | base-model | reentrance-model | no-shared-code-model | no-zipfian-model | single-exemplar-model | no-reduction-model | (cone-model)')
parser.add_argument("--no_titles", action="store_true", help="removes titles from the graphs")
args = parser.parse_args()

# Get the run information dataframe
# (information about parameter selection for each run)
run_infos = export.runs.get_run_infos(RUNS_DIR, args.selected_run)

# Check if the profile we are asked to export actually exists
if not args.profile in (export.profiles.ALL_PROFILE_NAMES):
    raise ValueError("Unknown profile")

# Convert profiles internally to a selection of profiles if necessary
profiles_to_process = [ args.profile ]

# Go over each profile and create the graphs for this profile
for profile in profiles_to_process:
    print(f"Processing profile '{profile}'")

    # We are now taking the exact parameter specification from the associated profile
    # So: Zipfian = True, dimensions = 10 etc.
    specification = export.profiles.PROFILES[profile]

    # Find the models adhering to this specification
    selected_models = export.parameters.find_eligible_models(run_infos, specification)
        
    if selected_models.shape[0] == 0:
        raise ValueError(f"No models found for profile '{profile}'")
        
    # Get the IDs from the selected models
    selected_model_ids = selected_models["run_id"].to_list()

    # Cone / toroidal model?
    toroidal = False
    if "toroidal" in selected_models.iloc[0]:
        toroidal = selected_models.iloc[0]["toroidal"]

    # We need different graphs depending on what profile we use
    # Also a different size for the confusion matrix
    graphs = export.graphs.get_export_graph_names(toroidal=toroidal)

    graphs_output = export.graphs.generate_graphs(
        args.selected_run,
        selected_model_ids,
        selected_models,
        RUNS_DIR,
        graphs,
        disable_title=True,
        toroidal=toroidal
    )

    # Save the files to disk!
    export.files.export_files(graphs_output, profile, FIGURES_DIR)