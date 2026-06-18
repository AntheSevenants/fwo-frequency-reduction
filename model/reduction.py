import numpy as np

from dataclasses import dataclass, asdict, field


@dataclass
class Base:
    # Default case: no reduction whatsoever
    def get_reduced_vector(self, input_vector: np.ndarray):
        return input_vector


@dataclass
class SoftThresholding(Base):
    strength: float = 15

    def get_reduced_vector(self, input_vector: np.ndarray):
        return input_vector - self.strength


# other reduction is available :-)
