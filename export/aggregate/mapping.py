import numpy as np


def get_parameter_value_mapping(
    datacollector_dataframes,
    selected_model_ids,
    selected_models,
    aggregate_parameter,
    outcome_parameter,
    step=-1,
):
    # It could be that we're trying to chart lambda against communicative succes
    # Given multiple iterations, we first need to combine the different models,
    # then we can start doing stuff.
    parameter_value_mapping = {}

    for i, df in enumerate(datacollector_dataframes):
        # Get the relevant datapoint (y)
        datapoint = df[outcome_parameter].tolist()[step]

        # Get the parameter setting for this model (x)

        # First, we have to locate the right row (from model id)
        selected_model_id = selected_model_ids[i]
        selected_model_row = selected_models.loc[
            selected_models["run_id"] == int(selected_model_id)
        ]

        if len(selected_model_row) == 0:
            raise ValueError(f"Model {selected_model_id} not found in run infos")
        elif len(selected_model_row) > 1:
            raise ValueError(
                f"Query for model {selected_model_id} in run infos yielded more than one result"
            )
        else:
            selected_model_row = selected_model_row.iloc[0]

        # Now, extract the value for the aggregate parameter for this row
        aggregate_parameter_value = selected_model_row[aggregate_parameter]
        # Turn aggregate parameter value into friendly name
        aggregate_parameter_value = to_friendly_name(aggregate_parameter, aggregate_parameter_value)
        if aggregate_parameter_value not in parameter_value_mapping:
            parameter_value_mapping[aggregate_parameter_value] = []

        parameter_value_mapping[aggregate_parameter_value].append(datapoint)

    return parameter_value_mapping

def to_friendly_name(aggregate_parameter, parameter_value):
    # Convert back into log
    if aggregate_parameter == "exponential_sampling_lambda":
        return pow(10, float(parameter_value))
    else:
        return parameter_value