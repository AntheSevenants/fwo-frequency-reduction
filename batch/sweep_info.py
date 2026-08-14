from dataclasses import dataclass
from typing import Optional


@dataclass
class SweepInfo:
    """Configuration information for sweeps. Sets the number of steps and datacollector step size."""

    num_steps: int
    datacollector_step_ratio: float
    datacollector_step_size: int = 1

    def __post_init__(self):
        self.datacollector_step_size = round(
            self.datacollector_step_ratio * self.num_steps
        )
