# %%
import mesa
import numpy as np
import pandas as pd

# %%
def load_vectors(file_path):
    # Load and skip first row
    df = pd.read_csv(file_path, sep=" ", header=None, skiprows=1)
    
    tokens = df.values[:, 0]
    frequencies = df.values[:, 1]
    percentiles = df.values[:, 2]
    vectors = df.values[:, 3:].astype('float64')
    vectors = np.asmatrix(vectors)

    return vectors, tokens, frequencies, percentiles

vectors, tokens, frequencies, percentiles = load_vectors("materials/vectors.txt")

# %%
class ReductionAgent(mesa.Agent):
    """A speaker in the model"""

    def __init__(self, model, vocabulary=None):
        # Pass the parameters to the parent class.
        super().__init__(model)

        self.vocabulary = self.model.vectors.copy() # weird behaviour when passed through the agentset
        self.speaking = False
        self.hearing = False

        #print(self.vocabulary.shape)

    def interact(self):
        # Define a dummy outcome for the communication
        communication_successful = False

        # Choose a position (but taking into account the frequency of the vocabulary!)
        random_index = self.model.weighted_random_index()

        # Get the right vector from the vocabulary
        random_vector = self.vocabulary[random_index, :]

        # print(random_vector)
    
        # Get the other information
        token = self.model.tokens[random_index]
        frequency = self.model.frequencies[random_index]
        percentile = self.model.percentiles[random_index]

        # Compute the reduction probability by multiplying the random chance by the chance depending on the percentile
        # TODO perhaps later implement this prior
        reduction_probability = self.model.random.uniform(0, 1) * self.model.token_reduction_prior(percentile)
        non_zero_indices = np.nonzero(random_vector)[1]

        is_reducing = False
        # With prevention for zeroing out vectors leaning on just one dimensions
        if reduction_probability <= self.model.reduction_prior and len(non_zero_indices) > 0:
            is_reducing = True

            # Q: should random dimension be full?
            # A: yes, otherwise there is no reduction going on!
            # Pick a random dimension from the non-zero dimensions
            random_dimension_index = self.model.random.choice(non_zero_indices.tolist())
            # Notice that I'm keeping the matrices two-dimensional because of performance reasons :)
            random_vector[0,random_dimension_index] = 0
        
        other_agent = None
        while other_agent is None:
            other_agent = self.random.choice(self.model.agents)
            if other_agent is self:
                other_agent = None

        other_agent.hearing = True
        heard_index = self.model.find_nearest_neighbour_index(self.vocabulary, random_vector)
        communication_successful = heard_index == random_index

        # If communication is successful, put the reduced vector in the vocabulary
        if communication_successful:
            self.vocabulary[random_index,:] = random_vector[0,:]

# %%
class ReductionModel(mesa.Model):
    """A model of Joan Bybee's *reducing effect*"""

    def __init__(self, num_agents, vectors, tokens, frequencies, percentiles, reduction_prior = 0.5, seed=None):
        super().__init__(seed=seed)

        self.num_agents = num_agents
        self.reduction_prior = reduction_prior
        
        self.vectors = vectors
        self.tokens = tokens
        self.frequencies = frequencies
        self.percentiles = percentiles

        
        self.cumulative_frequencies = np.cumsum(frequencies)
        self.total_frequency = self.cumulative_frequencies[-1]

        # print(vectors.shape)

        # We copy the vectors to the vocabulary of the agent
        ReductionAgent.create_agents(model=self, n=num_agents)

    def weighted_random_index(self):
        r = self.random.uniform(0, self.total_frequency)
        return next(i for i, cumulative_frequency in enumerate(self.cumulative_frequencies) if r < cumulative_frequency)
    
    def token_reduction_prior(self, percentile):
        return 1
    
    def find_nearest_neighbour_index(self, vocabulary, target_vector):
        distances = np.linalg.norm(vocabulary - target_vector, axis=1)
        return np.argmin(distances)

    def step(self):
        self.agents.shuffle_do("interact")

# %%
NUM_AGENTS = 100
model = ReductionModel(NUM_AGENTS, vectors, tokens, frequencies, percentiles, reduction_prior=1)
# model.step()
# for _ in range(10000):
#     model.step()
model.agents[0].interact()


