import pandas as pd
import numpy as np
from math import floor, ceil, exp

from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import silhouette_score
from collections import defaultdict
from model.types.neighbourhood import NeighbourhoodTypes
from model.types.sampling import SamplingTypes
from model.reduction import coarse_quantisation

EPSILON = 0.000001
N_CUTOFF = 10000
MAX_FREQUENCY = 310000

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

def generate_sample(ranks, n_sample, probabilities):
    probabilities = probabilities.astype(np.float64)
    probabilities /= probabilities.sum() # Normalise to sum to 1

    # Compute cumulative distribution function
    cumulative_percentiles = np.cumsum(probabilities)

    if len(ranks) != n_sample:
        # Sample n_sample items, ensuring distribution
        sampled_indices = np.random.choice(ranks, size=n_sample, replace=False, p=probabilities)
        sampled_indices.sort()  # Keep order for clarity
    
        # Get corresponding cumulative percentiles
        sampled_percentiles = [cumulative_percentiles[idx - 1] for idx in sampled_indices]
    else:
        sampled_indices = ranks
        sampled_percentiles = cumulative_percentiles
    
    return list(zip(sampled_indices, sampled_percentiles))

def generate_zipfian_sample(n_large=130000, n_sample=100, zipf_param=1.1):
    # Generate Zipfian probabilities for the "larger" dataset that we will sample from
    ranks = np.arange(1, n_large + 1)
    probabilities = 1 / np.power(ranks, zipf_param)

    return generate_sample(ranks, n_sample, probabilities)
    
def generate_exponential_sample(n_large=130000, n_sample=100, exp_param=0.001):
    # Generate exponentially decreasing probabilities
    ranks = np.arange(1, n_large + 1)
    probabilities = np.exp(-exp_param * ranks)

    return generate_sample(ranks, n_sample, probabilities)

def generate_linear_sample(n_large=130000, n_sample=100, intercept=10, slope=10):
    # Generate linear probabilities for the "larger" dataset that we will sample from
    ranks = np.arange(1, n_large + 1)
    probabilities = slope * ranks + intercept

    # Reverse
    probabilities = probabilities[::-1]

    return generate_sample(ranks, n_sample, probabilities)

def generate_zipfian_frequencies(n_large=130000, n_sample=100, zipf_param=1.1):
    sampled = generate_zipfian_sample(n_large, n_sample, zipf_param)
    ranks = [ rank for rank, probability in sampled ]

    frequencies = []
    for rank in ranks:
        frequency = ceil(rank ** (-zipf_param) * MAX_FREQUENCY)
        frequencies.append(frequency)

    return frequencies

def generate_linear_frequencies(intercept, slope, n_sample=100):
    ranks = np.arange(1, n_sample + 1)

    frequencies = []
    for rank in ranks:
        frequency = ceil(intercept + (n_sample - rank) * slope)
        frequencies.append(frequency)

    return frequencies

def generate_exponential_frequencies(n_large=130000, n_sample=100, exp_param=0.1):
    sampled = generate_exponential_sample(n_large=n_large, exp_param=exp_param, n_sample=n_sample)
    ranks = [ rank for rank, probability in sampled ]

    frequencies = []
    for rank in ranks:
        frequency = ceil(exp(-exp_param * (rank - 1)) * MAX_FREQUENCY)
        frequencies.append(frequency)

    return frequencies

def generate_word_vectors(vocabulary_size=1000, dimensions=300, floor=0, ceil=100, seed=42):
    np.random.seed(seed)

    # Generate random vectors with values ranging from floor to 100
    vectors = np.random.randint(floor, ceil, size=(vocabulary_size, dimensions))
    
    return np.asarray(vectors)

def generate_radical_vectors(vocabulary_size=1000, dimensions=300, ceil=100):
    return np.full((vocabulary_size, dimensions), ceil)

def generate_dirk_p2_vectors(max_vocabulary_size=1000, dimensions=300, floor=0, ceil=100, neighbourhood_type=NeighbourhoodTypes.SPATIAL, threshold=5, seed=42, MAX_ATTEMPTS=100, check_quantisation=0):
    np.random.seed(seed)

    # Start with zero representations
    vectors = None

    while True:
        success = False # Were we successful at generating a new vector?
        attempts = 0

        # Try MAX_ATTEMPTS times to generate a vector that is decently far away
        while attempts < MAX_ATTEMPTS:
            # Generate vector between floor and ceil value, of correct number of dimensions
            new_vector = np.random.randint(floor, ceil + 1, dimensions)

            if check_quantisation > 0:
                check_vector = coarse_quantisation(new_vector, check_quantisation)
            else:
                check_vector = new_vector

            # If first vector, we do not have to do any neighbourhood calculations, will always be OK!
            if vectors is not None:
                if neighbourhood_type == NeighbourhoodTypes.SPATIAL:
                    neighbours, weights = get_neighbours(vectors, check_vector, threshold)
                elif neighbourhood_type == NeighbourhoodTypes.LEVENSHTEIN:
                    neighbours, weights = get_neighbours_levenshtein(vectors, check_vector, threshold)
                else:
                    raise ValueError("Unexpected neighbourhood type")
            else:
                neighbours = []

            # If no representation is too close, this is a good vector!
            if len(neighbours) == 0:
                success = True
                if vectors is None:
                    vectors = np.array([ new_vector ])
                else:
                    vectors = np.vstack([vectors, new_vector])

                # If we reached the max number of constructions, set success to false to stop the loop
                if vectors.shape[0] == max_vocabulary_size:
                    success = False

                break

            attempts += 1

        if not success:
            break

    return vectors

def generate_quarter_circle_vectors(radius=100, num_points=50):
    # Angles from 0 to pi/2 (quarter circle)
    angles = np.linspace(0, np.pi / 2, num_points)[:-1]
    x = radius * np.cos(angles)
    y = radius * np.sin(angles)
    vectors = np.vstack((x, y)).T
    return vectors, angles

def add_noise_float(vector, noise_std=0.01):
    # Generate noise factors: 1 + epsilon
    noise_factor = 1 + np.random.normal(0, noise_std, size=vector.shape)
    
    # Ensure that noise_factor is positive.
    # Clip values to a minimum (e.g., 0.001) to avoid very small or negative multipliers.
    noise_factor = np.clip(noise_factor, 0.001, None)
    
    # Optionally, if your model has dimensions that are intentionally zero, do not modify them.
    noise_factor[vector == 0] = 1
    
    # Multiply the original vector by the noise factor.
    return vector * noise_factor

def add_noise(vector, ceiling=100, noise_strength=5, noise_std=1):
    # Generate noise factors: 1 + epsilon
    noise_factor = np.random.normal(noise_strength, noise_std, size=vector.shape)
    
    # Ensure that noise_factor is positive.
    # Clip values to a minimum (e.g., 0.001) to avoid very small or negative multipliers.
    noise_factor = np.clip(noise_factor, 1, ceiling)

    # Round to real  number
    noise_factor = np.round(noise_factor, 0)
    
    # Optionally, if your model has dimensions that are intentionally zero, do not modify them.
    noise_factor[vector == 0] = 1
    
    # Multiply the original vector by the noise factor.
    return vector + noise_factor

def count_non_zeroes(vector):
    return np.nonzero(vector)[1]

def count_zeroes(vector):
    return (vector == 0).nonzero()

def zero_ratio(vector):
    return (count_zeroes(vector) / vector.shape[1])

def compute_mean_non_zero_ratio(model):
    # For each agent, compute the zero ratio among all the exemplars and all the tokens
    # This is a crazy computation, I know, but what it's doing is
    # It looks for each token how many zeroes are in the vectors
    # And because we know how many tokens are in memory, and how many dimensions there are, 
    # we can compute a ratio of how many dimensions of the total are set to zero
    # Shape: (1) x token count
    zero_ratio_per_token_per_agent = np.array([ (agent.vocabulary == 0).sum(axis=2).sum(axis=1) / (model.exemplar_memory_n * model.num_dimensions) for agent in model.agents])

    # Compute the mean zero ratio over all agents
    # Shape: (vocab_size,)
    return zero_ratio_per_token_per_agent.mean(axis=0)

def compute_mean_agent_l1(model):
    mean_l1_per_agent = np.array([ np.nanmean(agent.memory.sum(axis=1)) for agent in model.agents ])

    return mean_l1_per_agent.mean()

def compute_mean_token_l1(model):
    agentwise_memory = np.zeros((model.num_agents, model.num_tokens))

    for agent_index, agent in enumerate(model.agents):
        matrix = agent.memory
        
        # token_index = index of the token
        # indices = the exemplars in the memory corresponding to this token
        for token_index, indices in enumerate(agent.indices_per_token):
            agentwise_memory[agent_index, token_index] = matrix[indices].sum(axis=1).mean()

    return agentwise_memory.mean(axis=0)
    # tokenwise_memory = agentwise_memory.mean(axis=2)
    # return tokenwise_memory
    #agent_token_averages.append(token_averages)

def compute_micro_mean_token_l1(model):
    agentwise_memory = compute_mean_token_l1(model)
    return agentwise_memory.mean(axis=0)

def compute_mean_exemplar_count(model):
    agentwise_count = np.zeros((model.num_agents, model.num_tokens))

    for agent_index, agent in enumerate(model.agents):
        token_counts = np.bincount(agent.indices_in_memory)
        agentwise_count[int(agent_index), :] = token_counts
    
    return agentwise_count.mean(axis=0)

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
    
def compute_reduction_success(model):
    if model.reduced_turns == 0:
        return 0
    else:
        return model.successful_reduced_turns / model.reduced_turns
    
def compute_communicative_success_per_token(model):
    with np.errstate(invalid="ignore"):
        return model.success_per_token / (model.success_per_token + model.failure_per_token)
    
def compute_ratio(model, succes_var, fail_var):
    with np.errstate(invalid="ignore"):
        return getattr(model, succes_var) / (getattr(model, succes_var) + getattr(model, fail_var))
    
def compute_communicative_success_macro_average(model):
    with np.errstate(invalid="ignore"):
        return np.nanmean(model.success_per_token / (model.success_per_token + model.failure_per_token))
    
def compute_token_good_origin(model):
    agentwise_memory = np.zeros((model.num_agents, model.num_tokens))

    for agent_index, agent in enumerate(model.agents):
        good_origin = agent.token_good_origin

        # token_index = index of the token
        # indices = the exemplars in the memory corresponding to this token
        for token_index, indices in enumerate(agent.indices_per_token):
            agentwise_memory[agent_index, token_index] = good_origin[indices].mean()

    return agentwise_memory.mean(axis=0)

def compute_fail_reason(model):
    return model.fail_reason
    
def compute_outcomes(model):
    return model.outcomes

def compute_tokens_chosen(model):
    return model.tokens_chosen

def compute_confusion_matrix(model):
    return model.confusion_matrix.copy()

def compute_average_vocabulary(model):
    vocabularies = [ agent.vocabulary.mean(axis=1) for agent in model.agents ]
    vocabularies = np.array(vocabularies)
    return np.mean(vocabularies, axis=0)

def compute_average_vocabulary_flexible(model):
    agentwise_memory = np.zeros((model.num_agents, model.num_tokens, model.num_dimensions))

    for agent_index, agent in enumerate(model.agents):
        matrix = agent.memory

        # token_index = index of the token
        # indices = the exemplars in the memory corresponding to this token
        for token_index, indices in enumerate(agent.indices_per_token):
            token_index = int(token_index)
            agentwise_memory[agent_index, token_index, :] = matrix[indices].mean(axis=0)

    return agentwise_memory.mean(axis=0)

def compute_full_vocabulary(model):
    full_memory = np.vstack([agent.memory for agent in model.agents])

    return full_memory

def compute_concept_stack(model):
    indices = np.concatenate([agent.indices_in_memory for agent in model.agents])

    return indices

def compute_full_vocabulary_ownership_stack(model):
    agent_indices = np.concatenate([[agent_index] * model.agents[agent_index].memory.shape[0] for agent_index in range(model.num_agents)])

    return agent_indices

def compute_average_communicative_success_probability(model):
    communicative_success_probabilities = [ agent.compute_communicative_success_probability() for agent in model.agents ]
    return np.mean(communicative_success_probabilities)

def compute_mean_communicative_success_per_token(model):
    return inner_ratio_computation(model, "turns_per_word")

def compute_mean_reduction_per_token(model):
    return inner_ratio_computation(model, "reduction_history")

def compute_repairs(model):
    return model.total_repairs

def compute_reentrance_ratio(model):
    if model.total_reductions == 0:
        return 0
    else:
        return model.reversed_reductions / model.total_reductions

def compute_mean_exemplar_age(model):
    agentwise_memory = np.zeros(model.num_agents)

    for agent_index, agent in enumerate(model.agents):
        agentwise_memory[agent_index] = (model.current_step - agent.last_used).mean()

    return agentwise_memory.mean()

def inner_ratio_computation(model, property):
    # Turn all communication memory matrices into a tensor
    # Shape: (num_agents, token count, memory count)
    memory_matrices = np.array([getattr(agent, property) for agent in model.agents])

    # Now, first compute the mean for each token for each agent, then globally
    agentwise_memory = memory_matrices.mean(axis=2)
    tokenwise_memory = agentwise_memory.mean(axis=0)

    return tokenwise_memory

def compute_vector_variation_inner(vocabulary, indices, labels, n):
    records = []

    for i in range(n):
        vectors = []
        for exemplar_index, token_index in enumerate(indices):
            if token_index == i:
                vectors.append(vocabulary[exemplar_index, :])

        vectors = np.vstack(vectors)
        vector_count = vectors.shape[0]

        distances = pdist(vectors, metric='euclidean')
        mean_distance = np.mean(distances)
        max_distance = np.max(distances)
        min_distance = np.min(distances)

        records.append({
            "index": i,
            "token": labels[i],
            "mean": mean_distance,
            "min" : min_distance,
            "max": max_distance,
            "n": vector_count
        })

    return pd.DataFrame.from_records(records)

def compute_vector_variation(model, n=10):
    vocabulary = compute_full_vocabulary(model)
    indices = compute_concept_stack(model)
    labels = model.tokens[:n]

    # Get only those indices which correspond to the top n
    eligible_indices = [ exemplar_index for exemplar_index in range(vocabulary.shape[0]) if indices[exemplar_index] < n ]

    vocabulary = vocabulary[eligible_indices, :]
    indices = indices[eligible_indices]

    return compute_vector_variation_inner(vocabulary, indices, labels, n)

def compute_vector_variation_single_agent(model, agent_index, n=10):
    vocabulary = model.agents[agent_index].memory
    exemplar_to_token_mapping = model.agents[agent_index].indices_in_memory

    labels = model.tokens[:n]

    # Get only those indices which correspond to the top n
    eligible_indices = [ exemplar_index for exemplar_index in range(vocabulary.shape[0]) if exemplar_to_token_mapping[exemplar_index] < n ]

    vocabulary = vocabulary[eligible_indices, :]
    indices = exemplar_to_token_mapping[eligible_indices]

    return compute_vector_variation_inner(vocabulary, indices, labels, n)

def compute_token_separation(model, n=10):
    vocabulary = compute_full_vocabulary(model)
    indices = compute_concept_stack(model)
    # Get only those indices which correspond to the top n
    eligible_exemplar_indices = [ exemplar_index for exemplar_index in range(vocabulary.shape[0]) if indices[exemplar_index] < n ]

    exemplars = vocabulary[eligible_exemplar_indices, :]
    exemplar_indices = indices[eligible_exemplar_indices]
    exemplar_labels = [ model.tokens[index] for index in exemplar_indices ]

    return silhouette_score(exemplars, exemplar_indices)

def compute_token_separation_between_agents(model, token_index, agent_index_1, agent_index_2):
    vocabulary = compute_full_vocabulary(model)
    exemplar_indices = compute_concept_stack(model)
    ownership_indices = compute_full_vocabulary_ownership_stack(model)
    
    exemplar_indices_agent_1 = [ exemplar_index for exemplar_index in range(vocabulary.shape[0]) if exemplar_indices[exemplar_index] == token_index and ownership_indices[exemplar_index] == agent_index_1 ]
    exemplar_indices_agent_2 = [ exemplar_index for exemplar_index in range(vocabulary.shape[0]) if exemplar_indices[exemplar_index] == token_index and ownership_indices[exemplar_index] == agent_index_2 ]

    if len(exemplar_indices_agent_1) + len(exemplar_indices_agent_2) <= 2:
        raise ValueError("Not enough exemplars to compare exemplars")

    vocabulary_agent_1 = vocabulary[exemplar_indices_agent_1, :]
    vocabulary_agent_2 = vocabulary[exemplar_indices_agent_2, :]
    full_vocabulary = np.vstack([ vocabulary_agent_1, vocabulary_agent_2 ])

    agent_ownership_labels = np.concatenate([ [1] * vocabulary_agent_1.shape[0], [2] * vocabulary_agent_2.shape[0] ])

    return silhouette_score(full_vocabulary, agent_ownership_labels)

def add_value_to_row(matrix, row_index, new_value):
    matrix[row_index, :-1] = matrix[row_index, 1:]
    matrix[row_index, -1] = new_value

    return matrix
    
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

def get_neighbours(matrix, target_row, distance_threshold, toroidal_size=None, weighted=False):
    if toroidal_size is None:
        return get_neighbours_linear(matrix, target_row, distance_threshold, weighted)
    else:
        return get_neighbours_toroidal(matrix, target_row, distance_threshold, toroidal_size, weighted)

def get_neighbours_toroidal(matrix, target_row, distance_threshold=2, yx_max=100, weighted=False):
    cx, cy = target_row
    radius_sq = distance_threshold ** 2
    height = yx_max
    # List of centers to check: main + wrapped versions
    centers = [(cx, cy)]

    # Apply toroidal wrapping rules
    if cx - distance_threshold < 0:
        # Left edge wraps to bottom
        centers.append((0 + cy, 0 - cx))
    if cy - distance_threshold < 0:
        # Bottom edge wraps to right
        centers.append((height - cx, height - cy))
        
    # Convert to array for broadcasting
    centers = np.array(centers)

    # Result mask
    inside_mask = np.zeros(matrix.shape[0], dtype=bool)
    weights = np.zeros(matrix.shape[0])

    for wrapped_cx, wrapped_cy in centers:
        dx = matrix[:, 0] - wrapped_cx
        dy = matrix[:, 1] - wrapped_cy
        dist_sq = dx**2 + dy**2
        inside_mask |= dist_sq <= radius_sq

        # Calculate weights as inverse of distance squared
        weights[dist_sq <= radius_sq] = 1 / (dist_sq[dist_sq <= radius_sq] + 1e-10) 

    indices = np.nonzero(inside_mask)[0]
    
    if weighted:
        weights = weights[indices]
        # Normalise the weights
        weights /= np.sum(weights)
    else:
        weights = None

    return indices, weights
    
def get_neighbours_linear(matrix, target_row, distance_threshold, weighted=False):
    # Calculate the Euclidean distance between the target row and all other rows
    distances = np.linalg.norm(matrix - target_row, axis=1)

    # Find the indices of rows within the distance threshold
    neighbour_indices = np.where(distances <= distance_threshold)[0]
    
    # Extract distances of the neighbours
    neighbour_distances = distances[neighbour_indices]

    if weighted:
        if len(neighbour_distances) > 0:
            unnormalized_weights = np.exp(-neighbour_distances)
            weights = unnormalized_weights / np.sum(unnormalized_weights)
        else:
            weights = np.array([])
    else:
        weights = None

    # Exclude the target row itself from the neighbours
    #neighbour_indices = neighbour_indices[neighbour_indices != target_row_index]

    return neighbour_indices, weights

def get_neighbours_nearest(matrix, target_row, n=2, weighted=False):
    distances = np.linalg.norm(matrix - target_row, axis=1)

    # Find the index of the nearest neighbours
    neighbour_indices = np.argsort(distances)[:n]

    if weighted:
        distances = np.sort(distances)[:n]
        
        # Turn distances into weights
        weights = 1 / distances

        # Normalise the weights
        weights /= np.sum(weights)
    else:
        # Disable weights
        weights = None

    return neighbour_indices, weights

def get_neighbours_levenshtein(matrix, target_row, distance_threshold, weighted=False):
    distances = np.sum(np.abs(matrix - target_row), axis=1)
    distances = np.divide(distances, target_row.shape[0])

    # Find the indices of rows within the distance threshold
    neighbour_indices = np.where(distances <= distance_threshold)[0]
    
    # Extract distances of the neighbours
    neighbour_distances = distances[neighbour_indices]

    if weighted:
        if len(neighbour_distances) > 0:
            unnormalized_weights = np.exp(-neighbour_distances)
            weights = unnormalized_weights / np.sum(unnormalized_weights)
        else:
            weights = np.array([])
    else:
        weights = None

    # Exclude the target row itself from the neighbours
    #neighbour_indices = neighbour_indices[neighbour_indices != target_row_index]

    return neighbour_indices, weights

def counts_to_percentages(arr):
    total_sum = np.sum(arr)
    percentages = arr / total_sum
    return percentages

def zipf_exemplars_per_construction(total_memory: int,
                 n_constructions: int,
                 s: float = 0.9,
                 min_per_construction: int = 0):
    """
    Allocate `total_memory` exemplar slots across `n_constructions` following a Zipfian
    distribution with exponent `s` (p_k ∝ 1 / rank^s). Returns a list of integer
    allocations (length n_constructions) that sum to total_memory.

    Arguments:
    - total_memory: total exemplar slots (e.g. 1000)
    - n_constructions: number of constructions (e.g. 50)
    - s: Zipf exponent (default 1.0). s=0 -> uniform.
    - min_per_construction: minimum slots reserved per construction (default 0).
      If >0 we preallocate min_per_construction to each, subtract from total_memory,
      then allocate the remainder by Zipf weights.
    - ranks: optional list of ranks (length n_constructions). If None, ranks = [1..n_constructions].

    Raises:
    - ValueError if inputs are inconsistent (e.g. not enough total_memory to satisfy min_per_construction).
    """
    if total_memory < 0 or n_constructions <= 0:
        raise ValueError("total_memory must be >=0 and n_constructions > 0")
    
    ranks = list(range(1, n_constructions + 1))

    # preallocate minimums
    if min_per_construction < 0:
        raise ValueError("min_per_construction must be >= 0")
    reserved = min_per_construction * n_constructions
    if reserved > total_memory:
        raise ValueError("Not enough total_memory to satisfy min_per_construction for every construction")

    remaining_memory = total_memory - reserved

    # compute weights (Zipf)
    if s == 0.0:
        weights = [1.0 for _ in ranks]
    else:
        weights = [1.0 / (float(r) ** s) for r in ranks]

    total_weight = sum(weights)
    if total_weight == 0:
        # fallback to uniform
        weights = [1.0 for _ in weights]
        total_weight = sum(weights)

    # ideal fractional allocation of remaining_memory according to weights
    ideal = [remaining_memory * (w / total_weight) for w in weights]

    # floor each ideal to get base allocation
    base = [int(floor(x)) for x in ideal]
    allocated = sum(base)
    remainder_slots = int(round(remaining_memory - allocated))  # should be small non-negative integer

    # compute fractional remainders and distribute remainder_slots by largest fractional parts
    fractional = [(i, ideal[i] - base[i]) for i in range(n_constructions)]
    # sort by fractional part descending (ties break by lower rank index implicitly)
    fractional.sort(key=lambda x: x[1], reverse=True)

    extra = [0] * n_constructions
    for j in range(remainder_slots):
        idx = fractional[j][0]
        extra[idx] += 1

    # final allocation = reserved min_per + base + extra
    final = [min_per_construction + base[i] + extra[i] for i in range(n_constructions)]

    # Safety: adjust small rounding errors so sum(final) == total_memory
    diff = total_memory - sum(final)
    if diff > 0:
        # give the remaining diff to highest-weight ranks (rank 1 first)
        ranked_indices = sorted(range(n_constructions), key=lambda i: weights[i], reverse=True)
        for k in range(diff):
            final[ranked_indices[k % n_constructions]] += 1
    elif diff < 0:
        # remove from lowest-weight ranks first (but never below min_per_construction)
        to_remove = -diff
        ranked_indices = sorted(range(n_constructions), key=lambda i: weights[i])  # low to high
        k = 0
        while to_remove > 0 and k < len(ranked_indices):
            i = ranked_indices[k]
            removable = final[i] - min_per_construction
            if removable > 0:
                take = min(removable, to_remove)
                final[i] -= take
                to_remove -= take
            else:
                k += 1
        if to_remove > 0:
            # this should not happen, but fail-safe: raise
            raise RuntimeError("Could not adjust allocation to meet total_memory constraint")

    return final

def linear_exemplars_per_construction(
    total_memory: int,
    n_constructions: int,
    min_per_construction: int = 0,
    intercept: float = 0.0,
    slope: float = 1.0,
):
    if total_memory < 0 or n_constructions <= 0:
        raise ValueError("total_memory must be >=0 and n_constructions > 0")
    if min_per_construction < 0:
        raise ValueError("min_per_construction must be >= 0")

    reserved = min_per_construction * n_constructions
    if reserved > total_memory:
        raise ValueError("Not enough total_memory for min_per_construction")

    remaining_memory = total_memory - reserved
    N = n_constructions
    ranks = list(range(1, N + 1))

    # generalized linear weights
    weights = [max(0.0, (intercept - 1) + slope * (N - r)) for r in ranks]
    total_weight = sum(weights)
    if total_weight <= 0:
        weights = [1.0 for _ in weights]
        total_weight = sum(weights)

    # ideal fractional allocations
    ideal = [remaining_memory * (w / total_weight) for w in weights]

    # floor to integer base allocation
    base = [int(floor(x)) for x in ideal]
    allocated = sum(base)

    # <-- FIX: exact integer remainder (avoid round() on floats)
    remainder_slots = remaining_memory - allocated
    if remainder_slots < 0:
        # defensive: should not happen, but guard
        remainder_slots = 0

    # distribute remaining slots by fractional part (largest-first)
    fractional = [(i, ideal[i] - base[i]) for i in range(N)]
    fractional.sort(key=lambda x: x[1], reverse=True)

    extra = [0] * N
    for j in range(remainder_slots):
        idx = fractional[j % N][0]
        extra[idx] += 1

    final = [min_per_construction + base[i] + extra[i] for i in range(N)]

    # rounding correction to ensure sum(final) == total_memory
    diff = total_memory - sum(final)
    if diff > 0:
        # give remaining diff to highest-weight ranks (rank 1 first)
        ranked_indices = sorted(range(N), key=lambda i: weights[i], reverse=True)
        for k in range(diff):
            final[ranked_indices[k % N]] += 1
    elif diff < 0:
        # remove from lowest-weight ranks first (but never below min_per_construction)
        to_remove = -diff
        ranked_indices = sorted(range(N), key=lambda i: weights[i])  # low to high
        k = 0
        while to_remove > 0 and k < len(ranked_indices):
            i = ranked_indices[k]
            removable = final[i] - min_per_construction
            if removable > 0:
                take = min(removable, to_remove)
                final[i] -= take
                to_remove -= take
            else:
                k += 1
        if to_remove > 0:
            raise RuntimeError("Could not adjust allocation to meet total_memory constraint")

    return final

from math import exp, floor

def exp_exemplars_per_construction(total_memory: int,
                                   n_constructions: int,
                                   lam: float = 0.2,
                                   min_per_construction: int = 0):
    """
    Allocate `total_memory` exemplar slots across `n_constructions` following an
    exponential distribution (p_k ∝ exp(-λ * rank)). Returns a list of integer
    allocations (length n_constructions) that sum to total_memory.

    Arguments:
    - total_memory: total exemplar slots (e.g. 1000)
    - n_constructions: number of constructions (e.g. 50)
    - lam: exponential decay rate λ (default 0.2).
      Higher λ -> faster decay (more skewed toward the first few constructions).
      λ=0 -> uniform distribution.
    - min_per_construction: minimum slots reserved per construction (default 0).
      If >0, these are preallocated to each, and the remainder follows the distribution.

    Raises:
    - ValueError if inputs are inconsistent.
    """
    if total_memory < 0 or n_constructions <= 0:
        raise ValueError("total_memory must be >=0 and n_constructions > 0")

    ranks = list(range(1, n_constructions + 1))

    # preallocate minimums
    if min_per_construction < 0:
        raise ValueError("min_per_construction must be >= 0")
    reserved = min_per_construction * n_constructions
    if reserved > total_memory:
        raise ValueError("Not enough total_memory to satisfy min_per_construction for every construction")

    remaining_memory = total_memory - reserved

    # compute exponential weights
    if lam == 0.0:
        weights = [1.0 for _ in ranks]  # uniform
    else:
        weights = [exp(-lam * (r - 1)) for r in ranks]

    total_weight = sum(weights)
    if total_weight == 0:
        weights = [1.0 for _ in ranks]
        total_weight = sum(weights)

    # ideal fractional allocation
    ideal = [remaining_memory * (w / total_weight) for w in weights]

    # integer allocation via flooring and distributing remainders
    base = [int(floor(x)) for x in ideal]
    allocated = sum(base)
    remainder_slots = int(round(remaining_memory - allocated))

    fractional = [(i, ideal[i] - base[i]) for i in range(n_constructions)]
    fractional.sort(key=lambda x: x[1], reverse=True)

    extra = [0] * n_constructions
    for j in range(remainder_slots):
        idx = fractional[j][0]
        extra[idx] += 1

    final = [min_per_construction + base[i] + extra[i] for i in range(n_constructions)]

    # Correct rounding differences if any
    diff = total_memory - sum(final)
    if diff != 0:
        ranked_indices = sorted(range(n_constructions), key=lambda i: weights[i], reverse=(diff > 0))
        step = 1 if diff > 0 else -1
        for k in range(abs(diff)):
            i = ranked_indices[k % n_constructions]
            if step < 0 and final[i] <= min_per_construction:
                continue
            final[i] += step

    return final
