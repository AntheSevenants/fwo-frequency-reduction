from dataclasses import dataclass, asdict, field
from typing import List, Callable, Dict, TYPE_CHECKING

if TYPE_CHECKING:
    from model.model import ReductionModel

import numpy as np


# What needs to be done with the tracked data?
class ReporterType:
    SINGULAR = 0
    MEAN = 1
    MEDIAN = 2
    STD = 3


# For which agent types does the computation need to be done?
class AgentType:
    INNOVATOR = 0
    CONSERVATOR = 1
    ALL = 2


# Definition of a single model reporter
@dataclass
class ModelReporter:
    property_name: str
    reporter_types: List[int] = field(
        default_factory=lambda: [
            ReporterType.SINGULAR,
            ReporterType.MEAN,
            ReporterType.MEDIAN,
            ReporterType.STD,
        ]
    )
    index: int | None = None


# Basic properties for which to construct model reporters
model_reporters_base = {
    "means": ModelReporter(property_name="vocabulary.__means__"),
    "sigmas": ModelReporter(property_name="vocabulary.__sigmas__"),
    "ctx_energy_mean": ModelReporter(
        property_name="vocabulary.__ctx_energy_mean_per_ctx__"
    ),
    "ctx_energy_median": ModelReporter(
        property_name="vocabulary.__ctx_energy_median_per_ctx__"
    ),
    "energy_mean": ModelReporter(property_name="vocabulary.__energy_mean__"),
    "energy_median": ModelReporter(property_name="vocabulary.__energy_median__"),
}

# Create a specialised tracking function depending on:
# - property name
# - index
# - innovator profile
# - data operation (no / mean / median)
function_translation: Dict[
    int,
    Callable[[str, int | None, bool | None], Callable[["ReductionModel"], np.ndarray]],
] = {
    ReporterType.SINGULAR: lambda property_name, index, for_innovator: lambda model: model.tracker.get_property_per_agent(
        property_name,
        index=index,
        # for_innovator=for_innovator,
    ),
    ReporterType.MEAN: lambda property_name, index, for_innovator: lambda model: model.tracker.get_property_mean_across_agents(
        property_name,
        index=index,
        # for_innovator=for_innovator,
    ),
    ReporterType.MEDIAN: lambda property_name, index, for_innovator: lambda model: model.tracker.get_property_median_across_agents(
        property_name,
        index=index,
        # for_innovator=for_innovator,
    ),
    ReporterType.STD: lambda property_name, index, for_innovator: lambda model: np.std(
        model.tracker.get_property_per_agent(
            property_name,
            index=index,
            # for_innovator=for_innovator,
        ),
        axis=0,
    ),
}

# For the reporter keys
agent_type_translation = {
    AgentType.INNOVATOR: "_innovator",
    AgentType.CONSERVATOR: "_conservator",
    AgentType.ALL: "",
}

# For the reporter keys
reporter_type_translation = {
    ReporterType.SINGULAR: "_per_agent",
    ReporterType.MEAN: "_mean",
    ReporterType.MEDIAN: "_median",
    ReporterType.STD: "_deviation",
}

# Translate type to an argument value for the tracker methods
agent_type_arg_translation = {
    AgentType.INNOVATOR: True,
    AgentType.CONSERVATOR: False,
    AgentType.ALL: None,
}


def get_model_reporter_key(
    reporter_name: str, reporter_type: int, agent_type: int
) -> str:
    """Generates a unique key for a model reporter based on the given parameters.

    Args:
        reporter_name (str): The name of the reporter.
        reporter_type (int): The type of the reporter, will be translated to its corresponding string representation.
        agent_type (int): The type of the agent, will be be translated to its corresponding string representation.

    Returns:
        str: The generated key, a concatenation of the reporter name, the string representation of the reporter type,
        and the string representation of the agent type.
    """

    reporter_type_ = reporter_type_translation[reporter_type]
    agent_type_ = agent_type_translation[agent_type]

    return f"{reporter_name}{reporter_type_}{agent_type_}"


def get_model_reporter_function(
    reporter_type: int, property_name: str, agent_type: int, property_index: int | None
) -> Callable[["ReductionModel"], np.ndarray]:
    """Builds the appropriate lambda function for a model reporter based on the given parameters.

    Args:
        reporter_type (int): The type of the reporter, which determines what tracker function is called.
        property_name (str): The name of the tracked property to be processed.
        agent_type (int): The type of the agent (innovator / conservator / none).
        property_index (int | None): The index of the property, if necessary. Defaults to None.

    Returns:
        Callable: The model reporter function built from the given parameters.
    """

    agent_type_arg_translation_ = agent_type_arg_translation[agent_type]

    return function_translation[reporter_type](
        property_name, property_index, agent_type_arg_translation_
    )


def get_model_reporters(
    for_all_types: bool = False,
) -> Dict[str, Callable[["ReductionModel"], np.ndarray | bool]]:
    """Build a full Dict of model reporters based on the (hard-coded) model reporter definitions.

    Args:
        for_all_types (bool): Whether to create model reporters for all agent types separately. Defaults to False.

    Returns:
        Dict[str, Callable[[ReductionModel], np.ndarray | bool]]: A Dict with the model reporter names as keys and the correct lambda functions as values.
    """

    model_reporters = {}
    for model_reporter_name in model_reporters_base:
        model_reporter_config = model_reporters_base[model_reporter_name]

        agent_types = [AgentType.INNOVATOR, AgentType.CONSERVATOR, AgentType.ALL]
        if not for_all_types:
            agent_types = [AgentType.ALL]

        for agent_type in agent_types:
            for reporter_type in model_reporter_config.reporter_types:
                # Set the model reporter key
                model_reporter_key = get_model_reporter_key(
                    model_reporter_name, reporter_type, agent_type
                )
                model_reporters[model_reporter_key] = get_model_reporter_function(
                    reporter_type,
                    model_reporter_config.property_name,
                    agent_type,
                    model_reporter_config.index,
                )

    return model_reporters
