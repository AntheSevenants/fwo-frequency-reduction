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
import model.types.vector

import export
import export.runs
import export.files
import export.graphs
import export.parameters
import export.models

import visualisation

RUNS_DIR = "models/"
FIGURES_OUTPUT_DIR = "/tmp/reduction-dashboard/"
PROFILE_NAME = "dashboard"
ENUM_MAPPING = {
    "repair": model.types.repair.Repair,
    "neighbourhood_type": model.types.neighbourhood.NeighbourhoodTypes,
    "production_model": model.types.production.ProductionModels,
    "reduction_mode": model.types.reduction.ReductionModes,
    "reducution_method": model.types.reduction.ReductionMethod,
    "feedback_type": model.types.feedback.FeedbackTypes,
    "vector": model.types.vector.VectorTypes,
    "reduction_method": model.types.reduction.ReductionMethod,
}

models_in_memory = {}
graphs_in_memory = {}

app = Flask(__name__, template_folder="dashboard/templates/", static_folder="dashboard/static/")
app.secret_key = 'secret_key_for_sessions'

graph_lock = threading.Lock()


@app.route('/')
def index():
    # run = a complête batch run with multiple parameter combinatinos
    runs = export.runs.get_runs(RUNS_DIR)
    # selected run = one of those batch runs
    selected_run = request.args.get('run')
    # you can filter for specific graphs
    selected_filter = request.args.get('filter')
    
    # Combination of parameters selected
    selected_parameters = dict(request.args)
    parameter_mapping = None
    constants_mapping = None
    disable_selection = False # if only one combination exists, skip parameter selection
    parameter_selection_id = None # the ID connected to the selected set of parameters

    # Flag which indicates we have to aggregate over multiple models
    do_aggregate = False

    # Cone / toroidal model?
    toroidal = False

    # There are keywords used by the application, these do not appear as parameters
    # We filter to check whether the user has made an actual parameter selection
    no_selection = len(list(set(selected_parameters) - set(export.parameters.RESERVED_KEYWORDS))) == 0

    # What graphs to show changes depending on whether the model is toroidal or not
    GRAPHS = export.graphs.get_analysis_graph_names(toroidal=toroidal)

    # Filter logic (what graph should we show?)
    if selected_filter == "no":
        selected_filter = None
    elif selected_filter in GRAPHS:
        graphs = [ selected_filter ]
    else:
        selected_filter = None

    if selected_filter is None:
        graphs = GRAPHS.copy()

    # Model selection logic
    if selected_run is not None:
        # Get the run information dataframe (parameter selection for each run)
        run_infos = export.runs.get_run_infos(RUNS_DIR, selected_run)

        # We build a representation of the different values that appear in the different columns.
        # If there are columns with multiple possible values, this means that there is a difference
        # So we can choose!
        parameter_mapping, constants_mapping = export.parameters.build_mapping(run_infos=run_infos)

        # If no model was selected, create a parameter selection ourselves
        if no_selection:
            for parameter in parameter_mapping:
                selected_parameters[parameter] = parameter_mapping[parameter][0]

            return redirect(url_for("index", **selected_parameters))

        # Now, let's look for the models that adhere to the parameter selection that we found
        selected_models = export.parameters.find_eligible_models(run_infos=run_infos, selected_parameters=selected_parameters)

        if selected_models.shape[0] == 0:
            raise ValueError("No models found with the selected parameter combination")
        
        # Get the IDs from the selected models
        selected_model_ids = selected_models["run_id"].to_list()
        # Remember the ID for this specific parameter selection
        parameter_selection_id = get_parameter_selection_id(selected_model_ids=selected_model_ids)

        # Get cached graphs
        cached_graphs = get_cached_graphs(selected_run, parameter_selection_id, graphs)
        non_cached_graph_count = len(list(set(graphs) - set(cached_graphs)))

        if non_cached_graph_count == 0:
            pass
        # If we still need some graphs, just build all of them again
        else:
            # Generate the directory where we will put the figures
            temp_models_figures_dir = make_temp_models_figures_dir(selected_run=selected_run, parameter_selection_id=parameter_selection_id)
            
            graphs_output = export.graphs.generate_graphs(selected_run, selected_model_ids, selected_models, RUNS_DIR, graphs)

            # Save the files to disk!
            export.files.export_files(graphs_output, PROFILE_NAME, temp_models_figures_dir)

    return render_template('index.html',
                           runs=runs,
                           selected_run=selected_run,
                           parameter_selection_id=parameter_selection_id,
                           selected_parameters=selected_parameters,
                           selected_filter=selected_filter,
                           parameter_mapping=parameter_mapping,
                           constants_mapping=constants_mapping,
                           no_selection=no_selection,
                           graphs=graphs,
                           all_graphs=graphs,
                           enum_mapping=ENUM_MAPPING,
                           get_enum_name=get_enum_name)

@app.route('/graph/<string:selected_run>/<string:parameter_selection_id>/<string:graph_name>')
def send_graph(graph_name, selected_run, parameter_selection_id):
    # Where our figures are stored for this parameter combination
    temp_models_figures_dir = make_temp_models_figures_dir(selected_run, parameter_selection_id)
    
    # Figure filename
    figure_filename = export.files.get_figure_filename(PROFILE_NAME, graph_name)

    graph_path = os.path.join(temp_models_figures_dir, figure_filename)
    return send_file(graph_path, mimetype='image/png')

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

def make_temp_run_figures_dir(selected_run):
    # This is where we will store the graphs output

    # We create a directory for the selected run
    temp_run_figures_dir = os.path.join(FIGURES_OUTPUT_DIR, selected_run)

    if not os.path.exists(temp_run_figures_dir):
        os.makedirs(temp_run_figures_dir, exist_ok=True)

    return temp_run_figures_dir

def make_temp_models_figures_dir(selected_run, parameter_selection_id):
    temp_run_figures_dir = make_temp_run_figures_dir(selected_run)
    
    # We create a directory for the selected parameter selection
    temp_models_figures_dir = os.path.join(temp_run_figures_dir, str(parameter_selection_id))

    if not os.path.exists(temp_models_figures_dir):
        os.makedirs(temp_models_figures_dir, exist_ok=True)

    return temp_models_figures_dir

# Turn a list of selected ids into an ID
def get_parameter_selection_id(selected_model_ids):
    # I think this will work because we're always working with unique numbers
    return sum(selected_model_ids)

def is_graph_in_cache(selected_run, parameter_selection_id, graph_name):
    temp_models_figures_dir = make_temp_models_figures_dir(selected_run, parameter_selection_id)
    graph_filename = export.files.get_figure_filename(PROFILE_NAME, graph_name)
    
    # Where graph would typically be saved
    graph_path = os.path.join(temp_models_figures_dir, graph_filename)

    # If it exists, it sits in cache
    return os.path.exists(graph_path)

def get_cached_graphs(selected_run, parameter_selection_id, graphs):
    # Check if the graphs we need already exist
    cached_graphs = []

    for graph_name in graphs:
        if is_graph_in_cache(selected_run, parameter_selection_id, graph_name):
            cached_graphs.append(graph_name)

    return cached_graphs

app.run(debug=True, port=8080, host="0.0.0.0")