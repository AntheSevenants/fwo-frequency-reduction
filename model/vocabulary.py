import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from typing import List
from dataclasses import dataclass


@dataclass
class Word:
    index: int
    display_name: str

    # Frequency of the word in the total vocabulary?
    prior: float

    # Distribution to organise the word with in memory
    init_means: List[float]
    init_sigmas: List[float]

    def __post_init__(self):
        # Convert to n-dimensional array
        self.means = np.array(self.init_means)
        self.sigmas = np.array(self.init_sigmas)

    def calculate_log_score(self, input_vector: np.ndarray):
        # Calculate log-likelihood for each of the n dimensions
        # We use norm.logpdf for numerical stability
        log_likelihoods = norm.logpdf(input_vector, self.means, self.sigmas)

        # Normalise for the frequency of the word (expressed as the prior)
        return np.log(self.prior) + np.sum(log_likelihoods)

    def update(self, vector: np.ndarray, weight: float = 1, learning_rate: float = 0.2):
        self.means = (1 - (learning_rate * weight)) * self.means + (
            learning_rate * weight * vector
        )

        # Update sigmas dimension by dimension
        errors = np.abs(vector - self.means)
        self.sigmas = (1 - (learning_rate * weight)) * self.sigmas + (
            learning_rate * weight * errors
        )
        self.sigmas = np.maximum(self.sigmas, 0.5)  # Prevent sigma from hitting 0


@dataclass
class Vocabulary:
    num_words: int
    num_dims: int

    valid_range: List[int]

    vectors: np.ndarray
    priors: List[float]

    # Names to use for the words
    display_names: List[str]

    def __post_init__(self):
        sigmas: List[float] = [5.0] * self.num_dims

        self.words: List[Word] = [
            Word(
                i,
                self.display_names[i],
                self.priors[i],
                list(self.vectors[i, :]),
                sigmas,
            )
            for i in range(self.num_words)
        ]

    def get_construction_vector(
        self, word_index: int, random: np.random.Generator | None = None
    ) -> np.ndarray:
        word = self.words[word_index]

        # Generate a vector from the word's means and sigmas
        generated_vector = np.array(
            norm.rvs(loc=word.means, scale=word.sigmas, random_state=random)
        )
        # Clip to ensure we stay in bounds [min, max]
        generated_vector = np.clip(
            generated_vector, self.valid_range[0], self.valid_range[1]
        )

        return generated_vector

    @property
    def __means__(self):
        return np.array([word.means for word in self.words])

    @property
    def __sigmas__(self):
        return np.array([word.sigmas for word in self.words])

    @property
    def __ctx_energy_mean_per_ctx__(self):
        return np.mean(self.__means__, axis=1)

    @property
    def __energy_mean__(self):
        return np.mean(self.__ctx_energy_mean_per_ctx__)

    @property
    def __ctx_energy_median_per_ctx__(self):
        return np.median(self.__means__, axis=1)

    @property
    def __energy_median__(self):
        return np.median(self.__ctx_energy_median_per_ctx__)
