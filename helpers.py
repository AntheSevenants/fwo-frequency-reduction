import pandas as pd
import numpy as np

def load_vectors(file_path):
    # Load and skip first row
    df = pd.read_csv(file_path, sep=" ", header=None, skiprows=1)
    
    tokens = df.values[:, 0]
    frequencies = df.values[:, 1]
    percentiles = df.values[:, 2]
    vectors = df.values[:, 3:].astype('float64')
    vectors = np.asmatrix(vectors)

    return vectors, tokens, frequencies, percentiles

def count_non_zeroes(vector):
    return np.nonzero(vector)[1]

def count_zeroes(vector):
    return (vector == 0).nonzero()

def zero_ratio(vector):
    return (count_zeroes(vector) / vector.shape[1])

def compute_mean_non_zero_ratio(model):
    # Turn all vocabulary matrices into a tensor
    # Shape: (num_agents, vocab_size (200), 100)
    vocab_matrices = np.array([agent.vocabulary for agent in model.agents])

    # Compute zero ratios for each row in every agent's vocabulary matrix
    # Shape: (num_agents, vocab_size)
    zero_ratios = (vocab_matrices == 0).sum(axis=2) / vocab_matrices.shape[2]

    # Compute the mean zero ratio over all agents
    # Shape: (vocab_size,)
    return zero_ratios.mean(axis=0)

def compute_communicative_success(model):
    if model.total_turns == 0:
        return 0
    else:
        return model.successful_turns / model.total_turns

def compute_communicative_failure(model):
    if model.total_turns == 0:
        return 0
    else:
        return model.failed_turns / model.total_turns