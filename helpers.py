import pandas as pd
import numpy as np

EPSILON = 0.000001
N_CUTOFF = 10000

def load_vectors(file_path):
    # Load and skip first row
    df = pd.read_csv(file_path, sep=" ", header=None, skiprows=1)
    
    tokens = df.values[:, 0]
    frequencies = df.values[:, 1]
    percentiles = df.values[:, 2]
    vectors = df.values[:, 3:].astype('float64')
    vectors = np.asmatrix(vectors)

    return vectors, tokens, frequencies, percentiles

def load_info(file_path, theoretical=False):
    df = pd.read_csv(file_path, sep="\t")

    frequencies = df["frequency"].to_list()
    if not theoretical:
        percentiles = df["percentile"].to_list()
    else:
        percentiles = df["theoretical_percentile"].to_list()
        percentiles[-1] = 1 # prevent out of bounds
    tokens = df["token"].to_list()
    ranks = df["rank"].to_list()

    return tokens, frequencies, percentiles, ranks

def generate_vectors():
    pass

def generate_zipfian_sample(n_large=130000, n_sample=100, zipf_param=1.1):
    # Generate Zipfian probabilities for the "larger" dataset that we will sample from
    ranks = np.arange(1, n_large + 1)
    probabilities = 1 / np.power(ranks, zipf_param)
    probabilities /= probabilities.sum() # Normalise to sum to 1
    
    # Compute cumulative distribution function
    cumulative_percentiles = np.cumsum(probabilities)
    
    # Sample n_sample items, ensuring Zipfian sampling
    sampled_indices = np.random.choice(ranks, size=n_sample, replace=False, p=probabilities)
    sampled_indices.sort()  # Keep order for clarity
    
    # Get corresponding cumulative percentiles
    sampled_percentiles = [cumulative_percentiles[idx - 1] for idx in sampled_indices]
    
    return list(zip(sampled_indices, sampled_percentiles))

def generate_word_vectors(vocabulary_size=1000, dimensions=300, seed=42):
    np.random.seed(seed)

    # Generate random vectors with uniform distribution in [0,1]
    vectors = np.random.rand(vocabulary_size, dimensions)
    
    # Normalize to unit length to ensure even distribution in space
    # Works as follows:
    # 1. for each row, compute: sqrt(row-element-1^2 + row-element-2^2)
    # 2. divide row elementwise by result, which makes it so the distribution in space is comparable
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    
    # Convert to matrix!
    return np.asmatrix(vectors)

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
    
def compute_tokens_chosen(model):
    return model.tokens_chosen

def compute_confusion_matrix(model):
    return model.confusion_matrix.copy()

def compute_average_vocabulary(model):
    vocabularies = [ agent.vocabulary for agent in model.agents ]
    vocabularies = np.array(vocabularies)
    return np.mean(vocabularies, axis=0)

def compute_average_communicative_success_probability(model):
    communicative_success_probabilities = [ agent.compute_communicative_success_probability() for agent in model.agents ]
    return np.mean(communicative_success_probabilities)
    
def distances_to_probabilities_linear(distances):
    # Add a small value so distance is never truly zero
    distances = distances + EPSILON

    # Step 1: Invert the distances
    inverted_distances = 1 / distances

    # Step 2: Normalize the inverted distances to form a probability distribution
    probabilities = inverted_distances / np.sum(inverted_distances)

    return probabilities
    
def distances_to_probabilities_softmax(distances):
    # Add a small value so distance is never truly zero
    distances = distances + EPSILON

    # Make distances negative
    neg_distances = -distances

    # Apply the softmax function
    exp_distances = np.exp(neg_distances)
    probabilities = exp_distances / np.sum(exp_distances)

    return probabilities