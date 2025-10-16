import numpy as np
import pandas as pd

class Datacollector:
    def __init__(self, df):
        self._df = df

    def get_model_vars_dataframe(self):
            return self._df

class Model:
    def __init__(self, dfs, run_infos, token_infos, min_steps):
        # All my code is written for a single simulation run
        # Hooray. I now have 100 simulation runs to aggregate
        # Time for some magic.
        df_aggregated = dfs[0].copy()

        min_steps_computed = int(min_steps) // int(run_infos["datacollector_step_size"]) + 2

        for column in df_aggregated.columns:
            print(column)

            if column in ["fail_reason", "outcomes"]:
                continue

            #print("Original shape:", dfs[0][column].shape)

            # This is where we put the column data for all the different models
            # Once everything has been collected, we will take the mean
            aggregated_data = []

            for df in dfs:
                # Truncate to minimum value if needed
                column_data = df[column].to_list()
                # print(len(column_data))
                column_data = column_data[:min_steps_computed]
                # print(len(column_data))

                aggregated_data.append(column_data)

            aggregated_column = np.array(aggregated_data)

            #print("New shape:", aggregated_column.shape)

            # Take mean across all simuations
            aggregated_column = aggregated_column.mean(axis=0)

            #print("Coerced shape:", aggregated_column.shape)

            # Assign back to the column
            df_aggregated[column] = pd.Series([arr for arr in aggregated_column], index=range(min_steps_computed))

        df_aggregated = df_aggregated.iloc[:min_steps_computed]

        self.datacollector = Datacollector(df_aggregated)

        self.tokens = token_infos["tokens"].to_list()
        self.frequencies = token_infos["frequencies"].to_list()
        self.percentiles = token_infos["percentiles"].to_list()
        self.ranks = token_infos["ranks"].to_list()

        self.num_tokens = len(self.tokens)

        self.datacollector_step_size = run_infos["datacollector_step_size"]
        self.current_step = min_steps
        self.neighbourhood_size = run_infos["neighbourhood_size"]

        if "value_ceil" in run_infos:
            self.value_ceil = run_infos["value_ceil"]

        if "light_serialilsation" in run_infos:
            if not run_infos["light_serialisation"]:
                self.full_vocabulary = dfs[0]["full_vocabulary"]
                #self.exemplar_indices = dfs[0]["exemplar_indices"]