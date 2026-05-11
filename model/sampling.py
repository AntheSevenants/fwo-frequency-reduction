import numpy as np

from typing import List, Tuple
from dataclasses import dataclass, asdict, field


def generate_sample(
    ranks: np.ndarray,
    n_sample: int,
    probabilities: np.ndarray,
    nprandom: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate a sample based on the given probabilities and the sample size

    Args:
        ranks (np.ndarray): The ranks of the full probability distribution
        n_sample (int): The size of the desired sample
        probabilities (np.ndarray): The probs of the full probability distribution
        nprandom (np.random.Generator): The random number generator for reproducibility

    Returns:
        Tuple[np.ndarray, np.ndarray]: Tuple: ranks, sampled probs
    """

    probabilities = probabilities.astype(np.float64)
    probabilities /= probabilities.sum()  # Normalise to sum to 1

    if len(ranks) != n_sample:
        # Sample n_sample items, ensuring distribution
        sampled_indices = nprandom.choice(
            ranks, size=n_sample, replace=False, p=probabilities
        )
        sampled_indices.sort()  # Keep order for clarity

        # Get corresponding cumulative percentiles
        sampled_probabilities = [probabilities[idx - 1] for idx in sampled_indices]
    else:
        sampled_indices = ranks
        sampled_probabilities = probabilities

    # Noramlise again
    sampled_probabilities /= np.array(sampled_probabilities).sum()

    return sampled_indices, sampled_probabilities


@dataclass
class BaseSampling:
    n_large: int = 130000
    n_sample: int = 100


@dataclass
class ZipfianSampling(BaseSampling):
    zipf_param: float = 1.0

    def get_priors(self, nprandom: np.random.Generator):
        # Generate Zipfian probabilities for the "larger" dataset that we will sample from
        ranks = np.arange(1, self.n_large + 1)
        probabilities = 1 / np.power(ranks, self.zipf_param)

        return generate_sample(ranks, self.n_sample, probabilities, nprandom)


@dataclass
class ExponentialSampling(BaseSampling):
    exp_param: float = 0.001

    def get_priors(self, nprandom: np.random.Generator):
        # Generate exponentially decreasing probabilities
        ranks = np.arange(1, self.n_large + 1)
        probabilities = np.exp(np.multiply(-self.exp_param, ranks))

        return generate_sample(ranks, self.n_sample, probabilities, nprandom)


@dataclass
class LinearSampling(BaseSampling):
    intercept: int = 10
    slope: int = 10

    def get_priors(self, nprandom: np.random.Generator):
        # Generate linear probabilities for the "larger" dataset that we will sample from
        ranks = np.arange(1, self.n_large + 1)
        probabilities = self.slope * ranks + self.intercept

        # Reverse
        probabilities = probabilities[::-1]

        return generate_sample(ranks, self.n_sample, probabilities, nprandom)
