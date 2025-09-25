import argparse
import pickle
import os
import math
import time
import json

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from pathlib import Path

import visualisation
import visualisation.l1
import visualisation.meta
import visualisation.dimscrap
import visualisation.angle
import visualisation.shims

FIGURES_FOLDER = "figures/"
MODELS_FOLDER = "models/"
REGULAR_GRAPH_NAMES = [ "l1-general", "l1-per-construction", "success", "matrix", "confusion-ratio", "half-life-per-construction" ]
TOROIDAL_GRAPH_NAMES = [ "angle-vocabulary-plot-2d-begin", "angle-vocabulary-plot-2d-end", "angle-vocabulary-plot-3d-begin" ]

base_model = {
    "memory_size": 1000,
    "jumble_vocabulary": False,
    "zipfian_sampling": True,
    "disable_reduction": False,
    "self_check": False
}

PROFILES = {
    "base-model": base_model,
    "reentrance-model": {**base_model, "self_check": True},
    "no-shared-code-model": {**base_model, "self_check": True, "jumble_vocabulary": True},
    "no-zipfian-model": {**base_model, "self_check": True, "zipfian_sampling": False},
    "single-exemplar-model": {**base_model, "self_check": True, "memory_size": 100},
    "no-reduction-model": {**base_model, "self_check": True, "disable_reduction": True}
}
ALL_PROFILE_NAMES = list(PROFILES.keys())

parser = argparse.ArgumentParser(
    description='export - make graphs for the reduction model')
parser.add_argument('selected_run', type=str, help='name of the run')
parser.add_argument('profile', type=str,
                    help='all | base-model | reentrance-model | no-shared-code-model | no-zipfian-model | single-exemplar-model | no-reduction-model | (cone-model)')
parser.add_argument('--overwrite_tokens_path', nargs='?', type=str, default=False, help='path to a tokens file to overwrite the existing tokens')
parser.add_argument("--no_titles", action="store_true", help="removes titles from the graphs")
parser.add_argument("--neutral_tokens", action="store_true", help="make tokens neutral (C1, C2, C3 ...)")
args = parser.parse_args()

# Load the selected run
selected_run_dir = f"{MODELS_FOLDER}{args.selected_run}"

model_infos_path = os.path.join(selected_run_dir, "run_infos.csv")
if not os.path.exists(model_infos_path):
    raise FileNotFoundError("Run infos CSV does nost exist")

# Now, let's single out the model(s) that adhere to the profile
run_infos = pd.read_csv(model_infos_path)

if not args.profile in ALL_PROFILE_NAMES and args.profile != "all" and args.profile != "cone-model":
    raise ValueError("Unknown profile")

# Flag which indicates we have to aggregate over multiple models
do_aggregate = False

profiles_to_process = [ args.profile ]
if args.profile == "all":
    profiles_to_process = ALL_PROFILE_NAMES

if args.profile == "cone-model":
    profiles_to_process = [ "cone-model" ]

for profile_name in profiles_to_process:
    print(f"Processing profile {profile_name}")

    if profile_name != "cone-model":
        profile = PROFILES[profile_name]

        # Create a mask to select the right model
        mask = pd.Series(True, index=run_infos.index)
        for column, value in profile.items():
            mask &= (run_infos[column].astype(str) == str(value))

        # Filter the data frame
        selected_model = run_infos[mask]
        if selected_model.shape[0] != 1:
            if selected_model.iloc[0]["iterations"] == selected_model.shape[0]:
                print("Multiple iterations detected")
                do_aggregate = True
            else:
                raise ValueError("Selected parameters do not single out a single model")
    else:
        selected_model = run_infos

    selected_run_ids = selected_model["run_id"].to_list()
    
    toroidal = False
    if "toroidal" in selected_model.iloc[0]:
        toroidal = selected_model.iloc[0]["toroidal"]

    if toroidal:
        GRAPH_NAMES = TOROIDAL_GRAPH_NAMES + REGULAR_GRAPH_NAMES
        n = 10
    else:
        GRAPH_NAMES = REGULAR_GRAPH_NAMES
        n = 35

    # These are the model data frames
    dfs = []

    for selected_run_id in selected_run_ids:
        model_path = os.path.join(selected_run_dir, f"{selected_run_id}.json")
        # Now, load the selected simulation
        with open(model_path, "rt") as model_file:
            df = json.loads(model_file.read())

            df = pd.DataFrame(df)

            for column in df.columns:
                if type(df[column].iloc[0]) == list:
                    df[column] = df[column].apply(lambda x: np.array(x))
                
            # Add to list of data frames
            dfs.append(df)

    # This info is the same across model runs and parameter combinations
    token_infos_path = os.path.join(selected_run_dir, "token_infos.csv")
    if not os.path.exists(token_infos_path):
        raise FileNotFoundError("Token infos CSV does not exist")
    token_infos = pd.read_csv(token_infos_path)

    if args.overwrite_tokens_path and not args.neutral_tokens:
        if not os.path.exists(args.overwrite_tokens_path):
            raise FileNotFoundError("Overwrite tokens path does not exist")

        extension = Path(args.overwrite_tokens_path).suffix
        if extension == ".csv":
            overwrite_tokens = pd.read_csv(args.overwrite_tokens_path)
        elif extension == ".tsv":
            overwrite_tokens = pd.read_table(args.overwrite_tokens_path)
        else:
            raise ValueError("File type not recognised for overwrite tokens file")

        for column_name in [ "token", "rank" ]:
            if not column_name in overwrite_tokens.columns:
                raise ValueError(f"No '{column_name}' column in overwrite tokens file")

        token_infos = token_infos.drop('tokens', axis=1)
        token_infos = pd.merge(token_infos, overwrite_tokens, left_on="ranks", right_on="rank")

        if not "tokens" in token_infos:
            token_infos = token_infos.rename(columns={"token": "tokens"})

    if args.neutral_tokens:
        token_infos["tokens"] = [str(i+1) for i in token_infos.index]
    
    # Fake model to satisfy my shoddy programming
    model = visualisation.shims.Model(dfs, run_infos.iloc[0], token_infos)

    graphs = {}

    for graph_name in GRAPH_NAMES:
        fig, ax = plt.subplots()

        if graph_name == "l1-general":
            figure = visualisation.l1.make_mean_l1_plot(model, ax=ax, smooth=False, disable_title=args.no_titles)
        elif graph_name == "l1-per-construction-mosaic":
            figure = visualisation.meta.make_layout_plot(model, visualisation.l1.words_mean_l1_bar,
                                                        steps=[math.floor(model.current_step / 4) * 1,
                                                            math.floor(model.current_step / 4) * 2,
                                                            math.floor(model.current_step / 4) * 3,
                                                            model.current_step],
                                                        disable_title=args.no_titles)
        elif graph_name == "l1-per-construction":
            figure = visualisation.l1.words_mean_l1_bar(model, model.current_step, ax=ax, disable_title=args.no_titles)
        elif  graph_name == "success":
            figure = visualisation.dimscrap.make_communication_plot_combined(model, smooth=False, ax=ax, disable_title=args.no_titles)
        elif graph_name == "matrix":
            figure = visualisation.meta.make_confusion_plot(model, model.current_step, n=n, ax=ax, disable_title=args.no_titles)
        elif graph_name == "matrix-mosaic":
            figure = visualisation.meta.make_layout_plot(model, visualisation.meta.make_confusion_plot, n=n,
                                                         steps=[math.floor(model.current_step / 4) * 1,
                                                            math.floor(model.current_step / 4) * 2,
                                                            math.floor(model.current_step / 4) * 3,
                                                            model.current_step],
                                                        disable_title=args.no_titles)
        elif graph_name == "confusion-ratio":
            figure = visualisation.l1.token_good_origin_first_n(model, ax=ax, disable_title=args.no_titles)
        elif graph_name in [ "angle-vocabulary-plot-2d-begin", "angle-vocabulary-plot-2d-end" ]:
            if graph_name == "angle-vocabulary-plot-2d-begin":
                step = 0
            elif graph_name == "angle-vocabulary-plot-2d-end":
                step = 700

            figure = visualisation.angle.make_angle_vocabulary_plot_2d(model, step, agent_filter=0, disable_title=args.no_titles)
        elif graph_name == "angle-vocabulary-plot-3d-begin":
            step = 0

            figure = visualisation.angle.make_angle_vocabulary_plot_3d(model, step, model.num_tokens, agent_filter=0, disable_title=args.no_titles)
        elif graph_name == "half-life-per-construction":
            figure = visualisation.l1.half_time_bar(model, model.current_step, ax=ax, disable_title=args.no_titles)

        graphs[graph_name] = figure

    for graph_name in graphs:
        filename = "".join(["fig-", profile_name, "-", graph_name, ".png"])
        print(f"Writing {filename}")

        file_path = os.path.join(FIGURES_FOLDER, filename)
        graphs[graph_name].figure.savefig(file_path)    

print("All done! Time for drinks!")