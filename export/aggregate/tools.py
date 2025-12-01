import numpy as np


def mean(parameter_mapping):
    for aggregate_parameter_value in parameter_mapping:
        parameter_mapping[aggregate_parameter_value] = np.array(
            parameter_mapping[aggregate_parameter_value]
        ).mean()

    return parameter_mapping