import numpy as np

import model.enums
from model.enums import to_dict
from typing import Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from model.model import ReductionModel


class CommunicationResult:
    FAILURE = 0
    SUCCESS = 1
    UNCERTAIN = 2


class CommunicationContext:
    ANY = 0
    NOT_REDUCING_NOT_REENTRANT = 1
    REDUCING = 2
    REENTRANCE = 3


class CommunicativeSuccess:
    def __init__(self, model: "ReductionModel"):
        self.model = model
        self.contexts = list(to_dict(CommunicationContext).values())
        self.results = list(to_dict(CommunicationResult).values())

    def reset(self):
        self.outcomes = np.zeros(
            (len(self.contexts), self.model.params.num_constructions, len(self.results))
        )
        self.reentrance_usage = np.zeros(
            (
                self.model.params.num_constructions,
                len(to_dict(model.enums.ReentranceUsage)),
            ),
        )

    def register_communication_outcome(
        self, communication_result: int, true_index: int, communication_context: int
    ):
        self.outcomes[communication_context, true_index, communication_result] += 1

    def register_reentrance_usage(self, reentrance_usage_index: int, true_index: int):
        self.reentrance_usage[true_index, reentrance_usage_index] += 1

    def get_percentages_per_ctx(self, communication_context: int):
        counts = self.outcomes[communication_context]

        with np.errstate(divide="ignore", invalid="ignore"):
            shares_per_ctx = np.true_divide(counts, np.sum(counts))

        return shares_per_ctx
