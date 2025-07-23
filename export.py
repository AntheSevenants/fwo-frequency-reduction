import argparse
import pickle
import os
import math
import time

import matplotlib.pyplot as plt
import pandas as pd

import visualisation
import visualisation.l1
import visualisation.meta
import visualisation.dimscrap
import visualisation.angle

FIGURES_FOLDER = "figures/"
MODELS_FOLDER = "models/"
GRAPH_NAMES = [ "l1-general", "l1-per-construction", "success", "matrix", "confusion-ratio" ]

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
    "no-reduction-model": {**base_model, "self_check": True, "memory_size": 100}
}
ALL_PROFILE_NAMES = list(PROFILES.keys())

parser = argparse.ArgumentParser(
    description='export - make graphs for the reduction model')
parser.add_argument('selected_run', type=str, help='name of the run')
parser.add_argument('profile', type=str,
                    help='all | base-model | reentrance-model | no-shared-code-model | no-zipfian-model | single-exemplar-model | no-reduction-model')
args = parser.parse_args()

# Load the selected run
selected_run_dir = f"{MODELS_FOLDER}{args.selected_run}"

model_infos_path = os.path.join(selected_run_dir, "run_infos.csv")
if not os.path.exists(model_infos_path):
    raise FileNotFoundError("Run infos CSV does nost exist")

# Now, let's single out the model(s) that adhere to the profile
run_infos = pd.read_csv(model_infos_path)

if not args.profile in ALL_PROFILE_NAMES and args.profile != "all":
    raise ValueError("Unknown profile")

profiles_to_process = [ args.profile ]
if args.profile == "all":
    profiles_to_process = ALL_PROFILE_NAMES

for profile_name in profiles_to_process:
    print(f"Processing profile {profile_name}")
    profile = PROFILES[profile_name]

    # Create a mask to select the right model
    mask = pd.Series(True, index=run_infos.index)
    for column, value in profile.items():
        mask &= (run_infos[column].astype(str) == str(value))

    # Filter the data frame
    selected_model = run_infos[mask]
    if selected_model.shape[0] != 1:
        raise ValueError("Selected parameters do not single out a single model")

    selected_run_id = str(selected_model.iloc[0]["run_id"])

    model_path = os.path.join(selected_run_dir, selected_run_id)
    # Now, load the selected simulation
    with open(model_path, "rb") as model_file:
        model = pickle.load(model_file)

    graphs = {}

    for graph_name in GRAPH_NAMES:
        fig, ax = plt.subplots()

        if graph_name == "l1-general":
            figure = visualisation.l1.make_mean_l1_plot(model, ax=ax, smooth=False)
        elif graph_name == "l1-per-construction":
            figure = visualisation.meta.make_layout_plot(model,
                                                                   visualisation.l1.words_mean_l1_bar,
                                                                   steps=[math.floor(model.current_step / 4) * 1,
                                                                          math.floor(
                                                                              model.current_step / 4) * 2,
                                                                          math.floor(model.current_step / 4) * 3, model.current_step])
        elif  graph_name == "success":
            figure = visualisation.dimscrap.make_communication_plot_combined(model, smooth=False, ax=ax)
        elif graph_name == "matrix":
            figure = visualisation.meta.make_layout_plot(model,
                                                      visualisation.meta.make_confusion_plot,
                                                      n=35, steps=[math.floor(model.current_step / 4) * 1,
                                                                   math.floor(
                                                          model.current_step / 4) * 2,
                                                          math.floor(model.current_step / 4) * 3, model.current_step])
        elif graph_name == "confusion-ratio":
            figure = visualisation.l1.token_good_origin_first_n(model, ax=ax)

        graphs[graph_name] = figure

    for graph_name in graphs:
        filename = "".join(["fig-", profile_name, "-", graph_name, ".png"])
        print(f"Writing {filename}")

        file_path = os.path.join(FIGURES_FOLDER, filename)
        graphs[graph_name].figure.savefig(file_path)    

print("All done! Time for drinks!")