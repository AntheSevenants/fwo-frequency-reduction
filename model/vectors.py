import numpy as np


# TODO: check output of this function
def generate(num_words=100, min_val=70, max_val=95, dims=10):
    target_sum = (max_val - round(max_val - min_val) / 2) * dims

    # Validation
    if target_sum < dims * min_val or target_sum > dims * max_val:
        raise ValueError(
            f"Target sum {target_sum} is impossible with bounds [{min_val}, {max_val}]"
        )

    words = []
    for _ in range(num_words):
        # Start with random values in the allowed range
        vec = np.random.uniform(min_val, max_val, dims)

        # Scale to target sum
        vec = vec * (target_sum / np.sum(vec))

        # Clip to ensure we stay in bounds [min, max]
        vec = np.clip(vec, min_val, max_val)

        # Correct the sum error caused by clipping
        # We do this iteratively to ensure the sum is met without breaking bounds
        for _ in range(5):
            error = target_sum - np.sum(vec)
            if abs(error) < 1e-5:
                break
            vec += error / dims
            vec = np.clip(vec, min_val, max_val)

        words.append(vec)

    return np.array(words)
