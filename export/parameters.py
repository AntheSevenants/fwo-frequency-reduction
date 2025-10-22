import pandas as pd

# Parameters used by the application. These are not parameters
RESERVED_KEYWORDS = [ "run", "filter" ]

def build_mapping(run_infos):
    parameter_mapping = {}
    constants_mapping = {}

    for column in run_infos:
        if column in [ "run_id", "seed" ]:
            continue
        
        unique_values = run_infos[column].unique()
        unique_values.sort()
        if len(unique_values) == 1:
            constants_mapping[column] = str(unique_values[0])
        else:
            parameter_mapping[column] = [ str(value) for value in unique_values.tolist() ]

    return parameter_mapping, constants_mapping

def find_eligible_models(run_infos, selected_parameters):
    # Create a mask to select the right model
    mask = pd.Series(True, index=run_infos.index)
    
    # Add the selected parameter to the mask
    for column, value in selected_parameters.items():
        if column in RESERVED_KEYWORDS:
            continue
        mask &= (run_infos[column].astype(str) == str(value))
    
    # Filter the data frame
    selected_models = run_infos[mask]

    return selected_models