import os
import pandas as pd

def get_runs(runs_dir):
    run_dirs = next(os.walk(runs_dir))[1]

    return run_dirs

def get_run_infos(runs_dir, selected_run):
    selected_run_dir = make_selected_run_dir(runs_dir, selected_run)

    # I am not a French speaker, I just like using the word "infos" because it is goofy
    model_infos_path = os.path.join(selected_run_dir, "run_infos.csv")
    if not os.path.exists(model_infos_path):
        raise FileNotFoundError("Run infos CSV does nost exist")
    
    return pd.read_csv(model_infos_path)

def make_selected_run_dir(runs_dir, selected_run):
    selected_run_dir = os.path.join(runs_dir, selected_run)
    if not os.path.exists(selected_run_dir):
        raise FileNotFoundError("Run directory does not exist")
    
    return selected_run_dir

def get_token_infos(runs_dir, selected_run):
    selected_run_dir = make_selected_run_dir(runs_dir, selected_run)

    # This info is the same across model runs and parameter combinations
    token_infos_path = os.path.join(selected_run_dir, "token_infos.csv")
    if not os.path.exists(token_infos_path):
        raise FileNotFoundError("Token infos CSV does not exist")
    
    token_infos = pd.read_csv(token_infos_path)
    return token_infos