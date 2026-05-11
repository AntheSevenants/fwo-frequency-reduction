import numpy as np

from dataclasses import dataclass, asdict, field


@dataclass
class Base:
    value_floor: int

    # Default case: no reduction whatsoever
    def get_reduced_vector(self, input_vector: np.ndarray):
        return input_vector


class SoftThresholding(Base):
    strength: float = 15

    def get_reduced_vector(self, input_vector: np.ndarray):
        return np.maximum(input_vector - self.strength, self.value_floor)


# other reduction is available :-)
