from dataclasses import dataclass, field
from typing import Dict, List, Any, Callable, Tuple, Union
from numpy.polynomial import Polynomial

import numpy as np


def compute_slope(data_matrix: np.ndarray) -> np.ndarray:
    """Given a run evolution, or a series of run evolutions, compute the linear slope of that evolution.

    Args:
        data_matrix (np.ndarray): Run evolution or series of run evolutions.

    Raises:
        ValueError: Raised if dimension of input matrix is not 2 or 3

    Returns:
        np.ndarray: List or matrix of the computed slopes
    """

    if len(data_matrix.shape) == 2:
        return np.array(
            [
                Polynomial.fit(range(len(row)), row, deg=1).convert().coef[1]
                for row in data_matrix
            ]
        )
    elif len(data_matrix.shape) == 3:
        return np.array(
            [
                [
                    Polynomial.fit(range(len(col)), col, deg=1).convert().coef[1]
                    for col in row.T
                ]
                for row in data_matrix
            ]
        )
    else:
        raise ValueError("Slope can only be computed for dimensions 2 and 3")


def compute_bool_perc(data_matrix: np.ndarray) -> float:
    """Compute how many runs of a combination have reached consensus. Only the last value in the evolution is counted.

    Args:
        data_matrix (np.ndarray): The input consensus matrix

    Returns:
        float: Percentage of model runs in this combination with a consensus, as a float range.
    """

    convergence_results = data_matrix[:, -1]
    percentage_true = np.sum(convergence_results) / len(convergence_results)

    return percentage_true


@dataclass
class CombinationOperation:
    """A combination operation defines the operation that is applied to the evolving metrics of a simulation run. This can be a mean, median, etc. Usually a combination operation is applied to axis 0, which means across model runs, but keeping the time dimension."""

    name: str
    operation: Callable[[np.ndarray], Any]


class CombinationOperations:
    """Presets for different combination operations."""

    Q1: CombinationOperation = CombinationOperation(
        name="q1", operation=lambda data_matrix: np.percentile(data_matrix, 25, axis=0)
    )
    Q3: CombinationOperation = CombinationOperation(
        name="q3", operation=lambda data_matrix: np.percentile(data_matrix, 75, axis=0)
    )
    MEDIAN: CombinationOperation = CombinationOperation(
        name="median", operation=lambda data_matrix: np.median(data_matrix, axis=0)
    )
    MIN: CombinationOperation = CombinationOperation(
        name="min", operation=lambda data_matrix: np.min(data_matrix, axis=0)
    )
    MAX: CombinationOperation = CombinationOperation(
        name="max", operation=lambda data_matrix: np.max(data_matrix, axis=0)
    )
    MEAN: CombinationOperation = CombinationOperation(
        name="mean", operation=lambda data_matrix: np.mean(data_matrix, axis=0)
    )
    SLOPE: CombinationOperation = CombinationOperation(
        name="slope", operation=compute_slope
    )
    BOOL_PERC: CombinationOperation = CombinationOperation(
        name="bool_perc", operation=compute_bool_perc
    )


def get_combination_metrics(
    data_matrix: np.ndarray, operations: List[CombinationOperation]
) -> Dict[str, Union[List[np.float64], List[List[np.float64]]]]:
    """Given a matrix with different runs and a list of operations, return the requested operation results.

    Args:
        data_matrix (np.ndarray): Input matrix of runs. Every row is a run, every column is a timestep.
        operations (List[CombinationOperation]): List of combination operations to apply to the model run results.

    Returns:
        Dict[str, Union[List[np.float64], List[List[np.float64]]]]: Dict with operation names as keys and operation results as values.
    """

    combined_data_out = {}
    for operation in operations:
        combined_data_out[operation.name] = operation.operation(data_matrix).tolist()

    return combined_data_out
