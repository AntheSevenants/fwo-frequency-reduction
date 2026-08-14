import model.enums

from dataclasses import dataclass, field
from typing import List, Optional, Any

from model.model_defaults import Parameters


def dict_to_params(params_dict: dict[str, Any]):
    """Convert a batchrunner-generated dict of arguments to the internal parameters representation

    Args:
        params_dict (dict[str, Any]): The generated arguments

    Returns:
        model.model_defaults.Parameters: An instance of the parameters, filled in with defaults if necessary
    """

    return Parameters(**params_dict)
