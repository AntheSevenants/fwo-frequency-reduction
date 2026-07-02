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


@dataclass
class CompressReduce(Base):
    decay_factor: float = 0.95
    homogenisation_factor: float = 0.1

    def get_reduced_vector(self, input_vector: np.ndarray):
        # We calculate the mean of the vector to get the "centre" of the identity
        centre = np.mean(input_vector)

        # Bring vectors closer together by adding a small amount of the difference with the centre
        output_vector = input_vector + (
            self.homogenisation_factor * (centre - input_vector)
        )

        # Now, also shrink the magnitude of the vector
        output_vector *= self.decay_factor

        # Prevent float bs
        output_vector = np.round(output_vector, 4)

        return output_vector


# other reduction is available :-)
