import os
import ujson as json
import pandas as pd
import numpy as np

import export.runs

def get_datacollector_dataframes(runs_dir, selected_run, selected_model_ids):
    # These are the datacollector dataframes of the different simulations
    datacollector_dataframes = []

    # First, piece together the selected run directory
    selected_run_dir = export.runs.make_selected_run_dir(runs_dir, selected_run)

    for selected_model_id in selected_model_ids:
        model_path = os.path.join(selected_run_dir, f"{selected_model_id}.json")
        
        # Now, load the selected simulation from the disk
        with open(model_path, "rt") as model_file:
            # Load the dataframe-as-json from disk, turn it into a dataframe
            df = json.loads(model_file.read())
            df = pd.DataFrame(df)

            # Turn every column into a numpy array
            for column in df.columns:
                first = df[column].iat[0]
                if isinstance(first, list):
                    df[column] = [np.array(x) for x in df[column].values]

            # Add to list of datacollector dataframes
            datacollector_dataframes.append(df)

    return datacollector_dataframes
