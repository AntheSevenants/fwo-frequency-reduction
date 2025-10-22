import numpy as np


def soft_thresholding_2(vector, reduction_strength, threshold):
    # Apply L1-based soft thresholding to encourage further sparsity
    vector = np.where(
        vector - reduction_strength >= threshold,
        vector - reduction_strength,
        vector,
    )

    return vector

def reduction_mask(model, vector, reduction_strength, width_ratio=0.5, threshold=1):
    # Compute the center of the reduction mask by selecting a random dimension
    center_index = model.random.randint(0, model.num_dimensions - 1)
    # Compute the reduction mask's length by taking a percentage of the vector length
    width = np.round(model.num_dimensions * width_ratio, decimals=0).astype("int")

    # Start by initialising a reduction mask the length of the vector
    reduction_mask = np.zeros(model.num_dimensions, dtype=np.int64)

    # Go over each affected dimension
    for offset in range(-width, width + 1):
        index = center_index + offset
        # Make sure not to go beyond vector dimensions
        if 0 <= index < model.num_dimensions:
            distance = abs(offset)

            # Linearly decreasing integer reduction, capped at 0
            reduction_value = max(reduction_strength - distance, 0)
            reduction_mask[index] = reduction_value

    reduced_vector = np.maximum(vector - reduction_mask, threshold)

    return reduced_vector


def taper(vec, total_reduce, width=3, rng=None):
    """
    Reduce a total value from a 1D vector with a taper around a random center.

    Parameters
    ----------
    vec : np.ndarray
        Input 1D vector.
    total_reduce : float
        Total amount to remove (distributed over nearby indices).
    width : float
        Controls the spread of the taper (higher = wider, flatter taper).
    rng : np.random.Generator or int, optional
        Random number generator or seed for reproducibility.

    Returns
    -------
    np.ndarray
        New vector after reduction.
    """
    if rng is None:
        rng = np.random.default_rng()
    elif isinstance(rng, int):
        rng = np.random.default_rng(rng)

    n = len(vec)
    center = rng.integers(0, n)

    # Create Gaussian-like taper centered at `center`
    x = np.arange(n)
    weights = np.exp(-0.5 * ((x - center) / width) ** 2)
    weights /= weights.sum()  # normalize

    # Scale weights to the total reduction
    reduction = total_reduce * weights

    # Ensure we don't go below zero
    new_vec = np.maximum(vec - reduction, 0)

    return new_vec


def angle_reduction(vector, distance, negative=False):
    angle = np.arctan2(vector[1], vector[0])

    if negative:
        distance = -distance

    vector[0] = max(0, vector[0] - distance * np.cos(angle))
    vector[1] = max(0, vector[1] - distance * np.sin(angle))

    return vector


def soft_thresholding_dimension(
    model, spoken_token_vector, reduction_strength, threshold
):
    # Choose a random dimension to reduce
    random_index = model.random.randint(0, model.num_dimensions - 1)

    # Remove from that dimension
    spoken_token_vector[random_index] = np.maximum(
        spoken_token_vector[random_index] - reduction_strength, threshold
    )

    return spoken_token_vector


def non_linear(vector, alpha=0.9, step=5):
    vector = multiply_decay(vector, alpha)
    vector = coarse_quantisation(vector, step)

    return vector


def multiply_decay(vector, alpha=0.9):
    # Multiply to introduce non-linearity
    vector = vector * alpha
    # Round and return
    return np.round(vector, 0)


def coarse_quantisation(vector, step=5):
    return np.round(vector / step).astype(int) * step


def bye_max(vector, reduction_strength, threshold):
    max_index = np.argmax(vector)
    new_value = vector[max_index] - reduction_strength

    if new_value > threshold:
        vector[max_index] = new_value

    return vector
