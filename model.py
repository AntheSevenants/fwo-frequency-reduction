
import mesa
import math
import numpy as np

from agents import ReductionAgent
from helpers import compute_communicative_success, compute_communicative_failure, compute_mean_non_zero_ratio, compute_tokens_chosen, distances_to_probabilities_softmax, distances_to_probabilities_linear, compute_confusion_matrix, compute_average_vocabulary, compute_average_communicative_success_probability, compute_mean_communicative_success_per_token, compute_mean_reduction_per_token, compute_repairs

class ReductionModel(mesa.Model):
    """A model of Joan Bybee's *reducing effect*"""

    def __init__(self, num_agents=50, vectors=[], tokens=[], frequencies=[], percentiles=[], ranks=[], reduction_prior = 0.5, zipfian_token_distribution=True, show_all_words=False, one_shot_nn=True, last_n_turns=-1, exemplar_memory_n=20, disable_reduction=False, seed=None):
        super().__init__(seed=seed)

        self.num_agents = num_agents
        self.reduction_prior = reduction_prior
        self.disable_reduction = disable_reduction
        self.seed = seed
        self.current_step = 0

        #
        # Visualisation stuff
        #

        # The grid is just for visualisation purposes, it doesn't do anything
        self.grid = mesa.space.SingleGrid(10, 10, True)

        # Whether to show all the words in the plot
        self.show_all_words = show_all_words
        # Whether communication uses one-shot matching of vectors
        self.one_shot_nn = one_shot_nn
        
        self.vectors = vectors
        self.tokens = tokens
        self.frequencies = frequencies
        self.percentiles = percentiles
        self.ranks = ranks
        self.num_tokens = len(self.tokens)
        self.num_dimensions = self.vectors.shape[1]
        self.lower_dimension_limit = math.floor(self.num_dimensions / 10)
        self.BASELINE_PROB = 1

        print(f"Lower dimension limit is {self.lower_dimension_limit}")
        
        self.cumulative_frequencies = np.cumsum(frequencies)
        self.total_frequency = self.cumulative_frequencies[-1]
        self.zipfian_token_distribution = zipfian_token_distribution
        self.last_n_turns = exemplar_memory_n
        self.exemplar_memory_n = exemplar_memory_n

        #
        # Communication success
        #
        self.confusion_matrix = np.zeros((self.num_tokens, self.num_tokens))
        self.reset()
        self.turns = []

        self.tokens_chosen = { token: 0 for token in self.tokens }

        # print(vectors.shape)

        # We copy the vectors to the vocabulary of the agent
        agents = ReductionAgent.create_agents(model=self, n=num_agents)
        # Then, we just put the little guys on the grid sequentially
        # I don't care for the order, it means nothing anyway
        for index, a in enumerate(agents):
            # Calculate the position based on the index
            i = index % self.grid.width
            j = index // self.grid.width
            
            self.grid.place_agent(a, (i, j))

        self.datacollector = mesa.DataCollector(
            model_reporters={"words_zero_ratio": compute_mean_non_zero_ratio,
                             "communicative_success": compute_communicative_success,
                             "communicative_failure": compute_communicative_failure,
                             "tokens_chosen": compute_tokens_chosen,
                             "confusion_matrix": compute_confusion_matrix,
                             "average_vocabulary": compute_average_vocabulary,
                             "average_communicative_success_probability": compute_average_communicative_success_probability,
                             "mean_success_per_token": compute_mean_communicative_success_per_token,
                             "mean_reduction_per_token":
                              compute_mean_reduction_per_token,
                             "total_repairs": compute_repairs }
        )
        
    def weighted_random_index(self):
        r = self.random.uniform(0, self.total_frequency)
        return next(i for i, cumulative_frequency in enumerate(self.cumulative_frequencies) if r < cumulative_frequency)
    
    def token_reduction_prior(self, percentile):
        return 1
    
    def do_nearest_neighbour(self, vocabulary, target_vector):
        return np.linalg.norm(vocabulary - target_vector, axis=1)
    
    def find_nearest_neighbour_index(self, vocabulary, target_vector):
        # With the exemplar model, we need to get an "average" representation across exemplars
        # So we average to get a matrix of size: token count x dimension size
        vocabulary_flat = vocabulary.mean(axis=1)

        distances = self.do_nearest_neighbour(vocabulary_flat, target_vector)
        return np.argmin(distances)
    
    def get_nearest_neighbour_distribution(self, vocabulary, target_vector):
        distances = self.do_nearest_neighbour(vocabulary, target_vector)
        print(distances)
        probabilities = distances_to_probabilities_linear(distances)
        print(probabilities)

        return probabilities
    
    def reset(self):
        self.total_repairs = 0
        self.successful_turns = 0
        self.failed_turns = 0
        self.total_turns = 0

    def step(self):
        self.reset()
        self.agents.do("reset")
        #self.random.choice(self.agents).interact()
        self.agents.shuffle_do("interact")
        self.datacollector.collect(self)
        self.current_step += 1