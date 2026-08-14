import numpy as np


# Disclosure:
# I asked Gemini to generate this function for me and I checked its output
# and added actually meaningful comments
def generate(
    num_words: int = 100,
    num_dims: int = 10,
    min_val: int = 70,
    max_val: int = 100,
    pool_size: int = 50000,
    random_generator: np.random.Generator | None = None,
) -> np.ndarray:
    """Generate a set of vectors that are maximally distinct from each other.

    Args:
        num_words (int, optional): The number of vectors to generate. Defaults to 100.
        num_dims (int, optional): The dimensionality count of the vectors. Defaults to 10.
        min_val (int, optional): The minimum value for each vector. Defaults to 70.
        max_val (int, optional): The maximum value for each vector. Defaults to 100.
        pool_size (int, optional): The size of the pool that we will sample from. A big pool ensures maximum distinctiveness, but will lead to number mostly at the edges of the value range. A small pool ensures varied numbers, but distinctiveness will suffer. Defaults to 50000.
        random_generator (np.random.Generator | None, optional): The random number generator to use. Pass a generator if you want results to be reproducible. Defaults to None (= a fresh generator will be instantiated).

    Returns:
        np.ndarray: The sampled generated vectors
    """

    random = np.random if random_generator is None else random_generator

    # Generate a large pool of vectors to choose from
    pool = np.random.randint(min_val, max_val + 1, size=(pool_size, num_dims))

    # We filter any possible duplicate values
    pool = np.unique(pool, axis=0)

    selected_vectors = []

    # We pick a random vector to start from
    start_index = random.choice(pool.shape[0])

    selected_vectors.append(pool[start_index])

    # We remove the chosen vector from the pool
    pool = np.delete(pool, start_index, axis=0)

    # Subsequent vectors will be picked based on maximum average distance from the
    # current pool. In this way we maximise the distinctiveness of the vectors
    # We repeat iteratively until we have selected all necessary vectors!
    min_squared_distances = np.sum((pool - selected_vectors[0]) ** 2, axis=1)

    for _ in range(1, num_words):
        # Find the vector that is furthest from the starting vector
        max_index = np.argmax(min_squared_distances)
        next_vector = pool[max_index]

        selected_vectors.append(next_vector)
        pool = np.delete(pool, max_index, axis=0)

        # Recompute the chosen vector from the distances
        min_squared_distances = np.delete(min_squared_distances, max_index)

        new_distances = np.sum((pool - next_vector) ** 2, axis=1)
        # The np.minimum call is a failsafe that makes sure we take as our perspective
        # the vector that is closest to all other vectors (this ensures maximum distance)
        min_squared_distances = np.minimum(min_squared_distances, new_distances)

    return np.array(selected_vectors, dtype=np.float64)
