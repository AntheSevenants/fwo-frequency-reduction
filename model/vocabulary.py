import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from typing import List
from dataclasses import dataclass


@dataclass
class Vocabulary:
    num_words: int
    num_dims: int
    initial_sigma: int

    valid_range: List[int]

    means: np.ndarray

    priors: List[float]

    # Names to use for the words
    display_names: List[str]

    def __post_init__(self):
        self.sigmas: np.ndarray = np.array(
            [[self.initial_sigma] * self.num_dims] * self.num_words
        )
        self.log_priors: np.ndarray = np.log(self.priors)

    def get_construction_vector(
        self, word_index: int, random: np.random.Generator | None = None
    ) -> np.ndarray:
        means = self.means[word_index, :]
        sigmas = self.sigmas[word_index, :]

        # Generate a vector from the word's means and sigmas
        generated_vector = np.array(
            norm.rvs(loc=means, scale=sigmas, random_state=random)
        )
        # Clip to ensure we stay in bounds [min, max]
        generated_vector = np.clip(
            generated_vector, self.valid_range[0], self.valid_range[1]
        )

        return generated_vector

    def calculate_log_likelihood(self, input_vector: np.ndarray) -> np.ndarray:
        # Calculate log-likelihood for each of the n dimensions
        # We use norm.logpdf for numerical stability
        # We compute the likelihood over the entire vocabulary to make things go faster
        log_likelihoods = norm.logpdf(input_vector, self.means, self.sigmas)

        return np.sum(log_likelihoods, axis=1)

    def calculate_log_score(self, input_vector: np.ndarray) -> np.ndarray:
        log_likelihoods = self.calculate_log_likelihood(input_vector)

        return self.log_priors + log_likelihoods

    def update_distribution(
        self,
        index: int,
        vector: np.ndarray,
        weight: float = 1,
        learning_rate: float = 0.2,
    ) -> None:
        self.means[index, :] = (1 - (learning_rate * weight)) * self.means[index, :] + (
            learning_rate * weight * vector
        )

        # Update sigmas dimension by dimension
        errors = np.abs(vector - self.means[index, :])
        self.sigmas[index, :] = (1 - (learning_rate * weight)) * self.sigmas[
            index, :
        ] + (learning_rate * weight * errors)
        self.sigmas[index, :] = np.maximum(
            self.sigmas[index, :], 1
        )  # Prevent sigma from hitting 0

    @property
    def __means__(self) -> np.ndarray:
        return self.means

    @property
    def __sigmas__(self) -> np.ndarray:
        return self.sigmas

    @property
    def __ctx_energy_mean_per_ctx__(self) -> np.ndarray:
        return np.mean(self.__means__, axis=1)

    @property
    def __energy_mean__(self) -> float:
        return float(np.mean(self.__ctx_energy_mean_per_ctx__))

    @property
    def __ctx_energy_median_per_ctx__(self) -> np.ndarray:
        return np.median(self.__means__, axis=1)

    @property
    def __energy_median__(self) -> float:
        return float(np.median(self.__ctx_energy_median_per_ctx__))
