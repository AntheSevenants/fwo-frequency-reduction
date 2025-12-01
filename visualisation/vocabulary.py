import numpy as np

def get_vocabulary(model, agent_filter, n, step):
    df = model.datacollector.get_model_vars_dataframe()
    full_vocabulary = df["full_vocabulary"].iloc[step]
    full_indices = df["full_indices"].iloc[step]
    ownership_indices = df["full_vocabulary_owernship"].iloc[step]
    labels = model.tokens[:n]
    
    # Initialise a dictionary to hold the mean vectors for each construction index
    construction_means = {}

    # Iterate over all possible construction indices (0 to n-1)
    for full_index in range(n):
        # Boolean mask for rows matching the construction index and agent
        mask = (full_indices == full_index) & (ownership_indices == agent_filter)

        # Extract the relevant rows
        relevant_rows = full_vocabulary[mask]

        # Compute the mean along the rows (axis=0)
        if relevant_rows.shape[0] > 0:
            mean_vector = np.mean(relevant_rows, axis=0)
            construction_means[full_index] = mean_vector

    # Convert the dictionary to a matrix (stack the mean vectors)
    result_matrix = np.vstack(list(construction_means.values()))

    return result_matrix

def good_matrix_display(arr, precision=2):
    """
    Prints a 2D NumPy array side by side with aligned columns for easy comparison.

    Parameters:
    - arr: 2D NumPy array
    - precision: Number of decimal places to display
    """
    # Convert the array to a string with fixed precision
    fmt = f"{{:.{precision}f}}"
    str_arr = np.array([[fmt.format(val) for val in row] for row in arr])

    # Find the maximum width for each column
    col_widths = [max(len(str_arr[i, j]) for i in range(str_arr.shape[0]))
                  for j in range(str_arr.shape[1])]

    # Print each row with aligned columns
    for row in str_arr:
        line = " | ".join(
            str(row[j]).rjust(col_widths[j]) for j in range(len(row))
        )
        print(line)