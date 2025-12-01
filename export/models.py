import os
import ujson as json
import pandas as pd
import numpy as np

import export.runs

from concurrent.futures import ThreadPoolExecutor

def load_dataframe(model_path):
    # Now, load the selected simulation from the disk
    with open(model_path, "rt") as model_file:
        # Load the dataframe-as-json from disk, turn it into a dataframe
        data = model_file.read()
        df = json.loads(data)
        df = pd.DataFrame(df)

        # Turn every column into a numpy array
        for column in df.columns:
            first = df[column].iat[0]
            if isinstance(first, list):
                df[column] = [np.array(x) for x in df[column].values]

    return df

def get_datacollector_dataframes(runs_dir, selected_run, selected_model_ids):
    # These are the datacollector dataframes of the different simulations
    datacollector_dataframes = []

    # First, piece together the selected run directory
    selected_run_dir = export.runs.make_selected_run_dir(runs_dir, selected_run)

    # Collect all model paths
    model_paths = [ os.path.join(selected_run_dir, f"{model_id}.json") for model_id in selected_model_ids]

    with ThreadPoolExecutor() as pool:
        datacollector_dataframes = list(pool.map(load_dataframe, model_paths))

    return datacollector_dataframes
