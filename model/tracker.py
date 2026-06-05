import mesa
import model.enums
import numpy as np


from typing import Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from model.model import ReductionModel


def get_nested_attr(obj: object, attr_path: str) -> Any:
    """
    Get a nested attribute from an object using dot notation.

    Args:
        obj: The object to get attributes from.
        attr_path: The attribute path in dot notation (e.g., "activation.level").

    Returns:
        The value of the nested attribute.
    """
    attributes = attr_path.split(".")
    current = obj
    try:
        for attr in attributes:
            current = getattr(current, attr)
        return current
    except AttributeError as e:
        raise AttributeError(f"Attribute '{attr_path}' not found") from e


class Tracker:
    """The tracker is the facilitates keeping track of the current model state."""

    def __init__(self, model: "ReductionModel"):
        """Initialise a Tracker by supplying the reduction model.

        Args:
            model (ReductionModel): Reduction model
        """

        # This will break pickling the model forever. Good :-)
        self.model = model

        self.reset()

    # Per step!
    def reset(self):
        self.chosen_constructions = np.zeros(self.model.params.num_constructions)
        self.reduction_outcomes = np.zeros(
            len(model.enums.to_dict(model.enums.CommunicationResult))
        )
        self.communication_results = np.zeros(
            len(model.enums.to_dict(model.enums.CommunicationResult))
        )
        self.reentrance_usage = np.zeros(
            len(model.enums.to_dict(model.enums.ReentranceUsage))
        )
        self.confusion_matrix = np.zeros(
            (self.model.params.num_constructions, self.model.params.num_constructions)
        )
        self.decision_entropy = []

    def register_construction_chosen(self, construction_index: int):
        """Register in the tracker which construction was chosen by an agent.

        Args:
            construction_index (int): The index of the chosen construction
        """

        self.chosen_constructions[construction_index] += 1

    def register_communication_result(self, communication_result: int):
        self.communication_results[communication_result] += 1

    def register_reduction_outcome(self, communication_result: int):
        self.reduction_outcomes[communication_result] += 1

    def register_win_index(self, win_index: int, true_index: int):
        self.confusion_matrix[true_index][win_index] += 1

    def register_reentrance_usage(self, reentrance_usage_index: int):
        self.reentrance_usage[reentrance_usage_index] += 1

    def register_decision_entropy(self, decision_entropy: float):
        self.decision_entropy.append(decision_entropy)

    def get_global(self, property_name):
        return getattr(self, property_name)

    def get_global_property_percentages(self, property_name: str, enum_length: int):
        # Get the value of this property
        counts = getattr(self, property_name)

        # Division by zero check
        if np.sum(counts) == 0:
            return counts

        # Turn into percentages
        shares = counts / counts.sum()

        # For now I'm only returning the shares as-is.
        # The legend is added afterwards anyway
        return shares

    def get_global_property_mean(self, property_name: str):
        value = getattr(self, property_name)

        if len(value) == 0:
            return 0

        return np.mean(value)

    def get_global_property_median(self, property_name: str):
        value = getattr(self, property_name)

        if len(value) == 0:
            return 0

        return np.median(value)

    def get_property_per_agent(self, property_name: str, index: int | None = None):
        """Retrieve a list of property values of each agent. If a property is multi-dimensional, you can ask to retrieve the value of one of the dimensions.

        Args:
            property_name (str): The name of the property that should be retrieved for each agent.
            index (int, optional): The index of the multi-dimensional value that should be retrieved, if multi-dimensional. Returns entire list if None. Defaults to None.

        Returns:
            np.Array: A numpy array containing the requested values.
        """

        agent_property_dist = []

        # Get the property for each agent
        for agent in self.model.agents:
            property_value = get_nested_attr(agent.atts, property_name)
            if index is not None:
                property_value = property_value[index]
            agent_property_dist.append(property_value)

        # Turn into numpy array
        return np.array(agent_property_dist)

    def get_property_mean_across_agents(
        self,
        property_name: str,
        index: Optional[int] = None,
    ):
        """Retrieve the mean of a requested property value across agents. If a property is multi-dimensional, you can ask to take the mean of the values of just one of the dimensions.

        Args:
            property_name (str): The name of the property that should be retrieved for each agent.
            index (int, optional): The index of the multi-dimensional value that should be retrieved, if multi-dimensional. Returns entire list if None. Defaults to None.

        Returns:
            float: A number of the mean of the value
        """

        agent_property_dist = self.get_property_per_agent(property_name, index=index)

        return agent_property_dist.mean(axis=0)

    def get_property_median_across_agents(
        self,
        property_name: str,
        index: Optional[int] = None,
    ):
        """Retrieve the median of a requested property value across agents. If a property is multi-dimensional, you can ask to take the median of the values of just one of the dimensions.

        Args:
            property_name (str): The name of the property that should be retrieved for each agent.
            index (int, optional): The index of the multi-dimensional value that should be retrieved, if multi-dimensional. Returns entire list if None. Defaults to None.

        Returns:
            float: A number of the median of the value
        """

        agent_property_dist = self.get_property_per_agent(property_name, index=index)

        return np.median(agent_property_dist, axis=0)
