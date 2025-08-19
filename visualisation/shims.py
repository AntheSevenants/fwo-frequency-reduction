import numpy as np
import pandas as pd

class Datacollector:
    def __init__(self, df):
        self._df = df

    def get_model_vars_dataframe(self):
            return self._df

class Model:
    def __init__(self, dfs, run_infos, token_infos):
        # All my code is written for a single simulation run
        # Hooray. I now have 100 simulation runs to aggregate
        # Time for some magic.
        df_aggregated = dfs[0].copy()

        for column in df_aggregated.columns:
            #print(column)

            if column in ["fail_reason", "outcomes"]:
                continue

            #print("Original shape:", dfs[0][column].shape)

            # This is where we put the column data for all the different models
            # Once everything has been collected, we will take the mean
            aggregated_data = []

            for df in dfs:
                aggregated_data.append(df[column].to_list())

            aggregated_column = np.array(aggregated_data)

            #print("New shape:", aggregated_column.shape)

            # Take mean across all simuations
            aggregated_column = aggregated_column.mean(axis=0)

            #print("Coerced shape:", aggregated_column.shape)

            # Assign back to the column
            df_aggregated[column] = pd.Series([arr for arr in aggregated_column])            
        
        self.datacollector = Datacollector(df_aggregated)

        self.tokens = token_infos["tokens"].to_list()
        self.frequencies = token_infos["frequencies"].to_list()
        self.percentiles = token_infos["percentiles"].to_list()
        self.ranks = token_infos["ranks"].to_list()

        self.num_tokens = len(self.tokens)

        self.datacollector_step_size = run_infos["datacollector_step_size"]
        self.current_step = run_infos["max_steps"]