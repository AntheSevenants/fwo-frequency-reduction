import numpy as np

def reduction_mask(model, vector, reduction_strength, width_ratio=0.05, threshold=1):
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