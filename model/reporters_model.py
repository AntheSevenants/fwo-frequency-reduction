import numpy as np
import model.enums

from dataclasses import dataclass, asdict, field
from typing import Any, Dict, List, Callable, TYPE_CHECKING

if TYPE_CHECKING:
    from model.model import ReductionModel


# Definition of a single model reporter (for model)
@dataclass
class ModelReporter:
    property_name: str
    reporter_types: List[int]
    associated_enum: object | None = None


# What needs to be done with the tracked data?
class ReporterType:
    AS_IS = 0
    PERCENT = 1
    MEAN = 2
    MEDIAN = 3


model_reporters_base = {
    "chosen_constructions": ModelReporter(
        property_name="chosen_constructions", reporter_types=[ReporterType.AS_IS]
    ),
    "communication_results_go": ModelReporter(
        property_name="communication_results",
        reporter_types=[ReporterType.PERCENT],
        associated_enum=model.enums.CommunicationResult,
    ),
    "reduction_outcomes_go": ModelReporter(
        property_name="reduction_outcomes",
        reporter_types=[ReporterType.PERCENT],
        associated_enum=model.enums.CommunicationResult,
    ),
    "reentrance_usage_go": ModelReporter(
        property_name="reentrance_usage",
        reporter_types=[ReporterType.PERCENT],
        associated_enum=model.enums.ReentranceUsage,
    ),
    "decision_entropy_go": ModelReporter(
        property_name="decision_entropy",
        reporter_types=[ReporterType.MEAN, ReporterType.MEDIAN],
    ),
    "confusion_matrix_go": ModelReporter(
        property_name="confusion_matrix", reporter_types=[ReporterType.AS_IS]
    ),
}

# For the reporter keys
reporter_type_translation = {
    ReporterType.AS_IS: "",
    ReporterType.PERCENT: "_share",
    ReporterType.MEAN: "_mean",
    ReporterType.MEDIAN: "_median",
}


def get_model_reporter_key(reporter_name: str, reporter_type: int) -> str:
    """Generates a unique key for a model reporter based on the given parameters.

    Args:
        reporter_name (str): The name of the reporter.
        reporter_type (int): The type of the reporter, will be translated to its corresponding string representation.

    Returns:
        str: The generated key, a concatenation of the reporter name, the string representation of the reporter type,
        and the string representation of the agent type.
    """

    reporter_type_ = reporter_type_translation[reporter_type]

    return f"{reporter_name}{reporter_type_}"


def get_global_reporter_function(property_name: str):
    reporter_function: Callable[["ReductionModel"], np.ndarray] = (
        lambda model: model.tracker.get_global(property_name)
    )

    return reporter_function


def get_percentage_reporter_function(property_name: str, enum_length: int):
    reporter_function: Callable[["ReductionModel"], np.ndarray] = (
        lambda model: model.tracker.get_global_property_percentages(
            property_name, enum_length=enum_length
        )
    )

    return reporter_function


def get_mean_reporter_function(property_name: str):
    reporter_function: Callable[["ReductionModel"], np.ndarray] = (
        lambda model: model.tracker.get_global_property_mean(property_name)
    )

    return reporter_function


def get_median_reporter_function(property_name: str):
    reporter_function: Callable[["ReductionModel"], np.ndarray] = (
        lambda model: model.tracker.get_global_property_median(property_name)
    )

    return reporter_function


def get_model_reporters() -> Dict[str, Callable[["ReductionModel"], np.ndarray | bool]]:
    model_reporters = {}

    # TODO: technically there is a mistake here since I'm overwriting every reporter type with just
    # the model reporter name. this is currently not a problem since I only define one reporter type
    # if I add multiple reporter types I'm gonna have to autogenerate the keys as well
    for model_reporter_name in model_reporters_base:
        model_reporter_config = model_reporters_base[model_reporter_name]
        for reporter_type in model_reporter_config.reporter_types:
            property_name = model_reporter_config.property_name

            reporter_function = lambda model: "ballekes"
            if reporter_type == ReporterType.AS_IS:
                reporter_function = get_global_reporter_function(property_name)
            elif reporter_type == ReporterType.PERCENT:
                if model_reporter_config.associated_enum is None:
                    raise ValueError(
                        "Associated enum cannot be None for PERCENT reporter type"
                    )

                enum_length = len(
                    model.enums.to_dict(model_reporter_config.associated_enum)
                )
                reporter_function = get_percentage_reporter_function(
                    property_name, enum_length
                )
            elif reporter_type == ReporterType.MEAN:
                reporter_function = get_mean_reporter_function(property_name)
            elif reporter_type == ReporterType.MEDIAN:
                reporter_function = get_median_reporter_function(property_name)

            model_reporter_key = get_model_reporter_key(
                model_reporter_name, reporter_type
            )
            model_reporters[model_reporter_key] = reporter_function

    return model_reporters
