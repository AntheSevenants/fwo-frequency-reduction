from flask import Flask, render_template, request, redirect, url_for, session, send_file
#from flask_socketio import SocketIO, emit
from io import BytesIO
import os
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import math
import pickle
import json
import argparse
import threading

import model.types
import model.types.feedback
import model.types.neighbourhood
import model.types.production
import model.types.reduction
import model.types.repair

import visualisation
import visualisation.l1
import visualisation.meta
import visualisation.dimscrap
import visualisation.angle

MODELS_DIR = "models/"
ENUM_MAPPING = {
    "repair": model.types.repair.Repair,
    "neighbourhood_type": model.types.neighbourhood.NeighbourhoodTypes,
    "production_model": model.types.production.ProductionModels,
    "reduction_mode": model.types.reduction.ReductionModes,
    "reducution_method": model.types.reduction.ReductionMethod,
    "feedback_type": model.types.feedback.FeedbackTypes
}

matplotlib.use('Agg')

GRAPHS = [ "mosaic_1", "mosaic_2", "confusion_mosaic", "l1_plot", "umap_mosaic", "memory_mosaic" ]

models_in_memory = {}
graphs_in_memory = {}

app = Flask(__name__, template_folder="dashboard/templates/", static_folder="dashboard/static/")
app.secret_key = 'secret_key_for_sessions'

graph_lock = threading.Lock()

@app.route('/')
def index():
    runs = get_runs()
    selected_run = request.args.get('run')
    selected_filter = request.args.get('filter')
    selected_run_id = None
    selected_parameters = dict(request.args)
    parameter_mapping = None
    constants_mapping = None

    

    reserved_keywords = [ "run", "filter" ]
    no_selection = len(list(set(selected_parameters) - set(reserved_keywords))) == 0

    if selected_filter == "no":
        selected_filter = None
    elif selected_filter in GRAPHS:
        graphs = [ selected_filter ]
    else:
        selected_filter = None
    
    if selected_filter is None:
        graphs = GRAPHS.copy()

    if selected_run is not None:
        run_infos = get_run_infos(selected_run)

        if run_infos.shape[0] == 1:
            no_selection = False

        parameter_mapping = {}
        constants_mapping = {}
    
        for column in run_infos:
            if column == "run_id":
                continue
            
            unique_values = run_infos[column].unique()
            unique_values.sort()

            if len(unique_values) == 1:
                constants_mapping[column] = str(unique_values[0])
            else:
                parameter_mapping[column] = [ str(value) for value in unique_values.tolist() ]

        if no_selection:
            for parameter in parameter_mapping:
                selected_parameters[parameter] = parameter_mapping[parameter][0]

            return redirect(url_for("index", **selected_parameters))
        # Look for the right model under the hood
        else:
            # Create a mask to select the right model
            mask = pd.Series(True, index=run_infos.index)
            for column, value in selected_parameters.items():
                if column in reserved_keywords:
                    continue

                mask &= (run_infos[column].astype(str) == value)

            # Filter the data frame
            selected_model = run_infos[mask]

            if selected_model.shape[0] != 1:
                raise ValueError("Selected parameters do not single out a single model")

            selected_run_id = str(selected_model.iloc[0]["run_id"])
            check_load_model(selected_run, selected_run_id)

            print("Model loaded")

    return render_template('index.html',
                           runs=runs,
                           selected_run=selected_run,
                           selected_run_id=selected_run_id,
                           selected_parameters=selected_parameters,
                           selected_filter=selected_filter,
                           parameter_mapping=parameter_mapping,
                           constants_mapping=constants_mapping,
                           no_selection=no_selection,
                           graphs=graphs,
                           all_graphs=GRAPHS,
                           enum_mapping=ENUM_MAPPING,
                           get_enum_name=get_enum_name)

def generate_plot(graph_name, model):
    if graph_name == "mosaic_1":
        return visualisation.meta.combine_plots(model,
            lambda model, ax: visualisation.dimscrap.make_communication_plot(model, ax=ax, smooth=False),
            visualisation.l1.communicative_success_first_n,
            lambda model, ax: visualisation.l1.make_communicative_success_macro_plot(model, ax=ax, smooth=False),
            lambda model, ax: visualisation.l1.token_good_origin_first_n(model, ax=ax),
            lambda model, ax: visualisation.l1.make_mean_exemplar_age_plot(model, ax=ax, smooth=False),
            lambda model, ax: visualisation.l1.make_reduction_success_plot(model, ax=ax, smooth=False),
            )
    elif graph_name == "mosaic_2":
        return visualisation.meta.combine_plots(model,
            lambda model, ax: visualisation.l1.make_fail_reason_plot(model, ax=ax, include_success=True),
            visualisation.l1.words_l1_plot_first_n,
            lambda model, ax: visualisation.l1.make_mean_l1_plot(model, ax=ax, smooth=False),
            lambda model, ax: visualisation.dimscrap.make_communication_plot(model, ax=ax, smooth=False),
            visualisation.l1.words_mean_exemplar_count_first_n,
            visualisation.l1.words_mean_exemplar_count_bar)
    elif graph_name == "confusion_mosaic":
        return visualisation.meta.make_layout_plot(model,
                                    visualisation.meta.make_confusion_plot,
                                    n=35, steps=[math.floor(model.current_step / 4) * 1,
                                                 math.floor(model.current_step / 4) * 2,
                                                 math.floor(model.current_step / 4) * 3, model.current_step])
    elif graph_name == "umap_mosaic":
        return visualisation.meta.make_layout_plot(model,
                                    visualisation.meta.make_umap_plot,
                                    steps=[math.floor(model.current_step / 4) * 1,
                                           math.floor(model.current_step / 4) * 2,
                                           math.floor(model.current_step / 4) * 3,
                                           model.current_step])
    elif graph_name == "memory_mosaic":
        return visualisation.meta.make_layout_plot(model,
                                    visualisation.meta.make_umap_full_vocabulary_plot,
                                    steps=[100,
                                           math.floor(model.current_step / 4) * 2,
                                           math.floor(model.current_step / 4) * 3,
                                           model.current_step],
                                    n=9, agent_filter=0)
    elif graph_name == "l1_plot":
        return visualisation.meta.make_layout_plot(model,
                                    visualisation.l1.words_mean_l1_bar,
                                    steps=[math.floor(model.current_step / 4) * 1,
                                                 math.floor(model.current_step / 4) * 2,
                                                 math.floor(model.current_step / 4) * 3, model.current_step])
    else:
        return "invalide"

@app.route('/graph/<string:graph_name>/<string:selected_run>/<string:selected_run_id>/')
def generate_graph(graph_name, selected_run, selected_run_id):
    if selected_run is None:
        raise ValueError("Please specify what run to create a graph for")
    
    if selected_run_id is None:
        raise ValueError("Please specify what model to create a graph for")
    
    check_load_model(selected_run, selected_run_id)
    model = models_in_memory[selected_run][selected_run_id]

    if not selected_run in graphs_in_memory:
        graphs_in_memory[selected_run] = {}

    if not selected_run_id in graphs_in_memory[selected_run]:
        graphs_in_memory[selected_run][selected_run_id] = {}

    with graph_lock:
        if not graph_name in graphs_in_memory[selected_run][selected_run_id]:
            figure = generate_plot(graph_name, model)
            buffer = BytesIO()
            figure.savefig(buffer, format='png')
            # Store in cache
            image_bytes = buffer.getvalue()
            graphs_in_memory[selected_run][selected_run_id][graph_name] = image_bytes
        else:
            image_bytes = graphs_in_memory[selected_run][selected_run_id][graph_name]
            #buffer.seek(0)

    # Wrap in new buffer (else I get a closed buffer error)
    return send_file(BytesIO(image_bytes), mimetype='image/png')

def get_runs():
    run_dirs = next(os.walk(MODELS_DIR))[1]

    return run_dirs

def load_model(selected_run, selected_run_id):
    selected_run_id = str(selected_run_id)

    selected_run_dir = os.path.join(MODELS_DIR, selected_run)
    selected_model_path = os.path.join(selected_run_dir, selected_run_id)
    if not os.path.exists(selected_model_path):
        raise FileNotFoundError("Selected model does not have a path associated with it")
    
    with open(selected_model_path, "rb") as model_file:
        model = pickle.load(model_file)

    models_in_memory[selected_run][selected_run_id] = model

def check_load_model(selected_run, selected_run_id):
    if not selected_run in models_in_memory:
        models_in_memory[selected_run] = {}

    selected_run_id = str(selected_run_id)
    if not selected_run_id in models_in_memory[selected_run]:
        load_model(selected_run, selected_run_id)

def make_selected_run_dir(selected_run):
    selected_run_dir = os.path.join(MODELS_DIR, selected_run)
    if not os.path.exists(selected_run_dir):
        raise FileNotFoundError("Run directory does not exist")
    
    return selected_run_dir

def get_run_infos(selected_run):
    selected_run_dir = make_selected_run_dir(selected_run)

    # I am not a French speaker, I just like using the word "infos" because it is goofy
    model_infos_path = os.path.join(selected_run_dir, "run_infos.csv")
    if not os.path.exists(model_infos_path):
        raise FileNotFoundError("Run infos CSV does nost exist")
    
    return pd.read_csv(model_infos_path)

# From Le Chat
def get_enum_name(cls, value):
    # Get all attributes of the provided class
    attributes = [(name, getattr(cls, name))
                  for name in dir(cls)
                  if not name.startswith("__")]

    # Create a mapping of values to names
    enum_mapping = {str(value): name for name, value in attributes if isinstance(value, int)}

    # Return the corresponding enum name or "Unknown" if not found
    return enum_mapping.get(value, "Unknown")

app.run(debug=True, port=8080, host="0.0.0.0")