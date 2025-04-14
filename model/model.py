
import mesa
import math
import numpy as np

from model.agents import ReductionAgent
from model.helpers import compute_communicative_success, compute_communicative_failure, compute_mean_non_zero_ratio, compute_tokens_chosen, distances_to_probabilities_softmax, distances_to_probabilities_linear, compute_confusion_matrix, compute_average_vocabulary, compute_average_communicative_success_probability, compute_mean_communicative_success_per_token, compute_mean_reduction_per_token, compute_repairs, compute_mean_agent_l1, compute_mean_token_l1, compute_fail_reason, compute_mean_exemplar_count, compute_average_vocabulary_flexible, compute_communicative_success_per_token, compute_communicative_success_macro_average
from model.types.neighbourhood import NeighbourhoodTypes
from model.types.production import ProductionModels
from model.types.reduction import ReductionModes
from model.types.feedback import FeedbackTypes
from model.types.repair import Repair


class ReductionModel(mesa.Model):
    """A model of Joan Bybee's *reducing effect*"""

    def __init__(self, num_agents=50, vectors=[], tokens=[], frequencies=[], percentiles=[], ranks=[], reduction_prior = 0.5, memory_size=1000, initial_token_count=2, prefill_memory=True, disable_reduction=False, neighbourhood_type=NeighbourhoodTypes.SPATIAL, neighbourhood_size=0.5, production_model=ProductionModels.SINGLE_EXEMPLAR, reduction_mode=ReductionModes.ALWAYS, feedback_type=FeedbackTypes.FEEDBACK, repair=Repair.NO_REPAIR, datacollector_step_size=100, seed=None):
        super().__init__(seed=seed)

        self.num_agents = num_agents
        self.reduction_prior = reduction_prior
        self.disable_reduction = disable_reduction
        self.memory_size = memory_size
        self.initial_token_count = initial_token_count
        self.prefill_memory = prefill_memory
        self.seed = seed
        self.current_step = 0
        self.datacollector_step_size = datacollector_step_size

        self.neighbourhood_size = neighbourhood_size
        self.neighbourhood_type = neighbourhood_type
        self.production_model = production_model
        self.reduction_mode = reduction_mode
        self.feedback_type = feedback_type
        self.repair = repair

        #
        # Visualisation stuff
        #

        # The grid is just for visualisation purposes, it doesn't do anything
        self.grid = mesa.space.SingleGrid(10, 10, True)
        self.vectors = vectors
        self.tokens = tokens
        self.frequencies = frequencies
        self.percentiles = percentiles
        self.ranks = ranks
        self.num_tokens = len(self.tokens)
        self.num_dimensions = self.vectors.shape[1]
        self.lower_dimension_limit = math.floor(self.num_dimensions / 10)

        print(f"Lower dimension limit is {self.lower_dimension_limit}")
        
        self.cumulative_frequencies = np.cumsum(frequencies)
        self.total_frequency = self.cumulative_frequencies[-1]

        #
        # Communication success
        #
        self.confusion_matrix = np.zeros((self.num_tokens, self.num_tokens))
        self.reset(force=True)
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
            model_reporters={"communicative_success": compute_communicative_success,
                             "communicative_failure": compute_communicative_failure,
                             "mean_agent_l1": compute_mean_agent_l1,
                             "mean_token_l1": compute_mean_token_l1,
                             "confusion_matrix": compute_confusion_matrix,
                             "fail_reason": compute_fail_reason,
                             "mean_exemplar_count": compute_mean_exemplar_count,
                             "success_per_token": compute_communicative_success_per_token }
        )

        self.datacollector.collect(self)

    def reset(self, force=False):
        if self.current_step - 1 % self.datacollector_step_size == 0 or force:
            self.total_repairs = 0
            self.successful_turns = 0
            self.failed_turns = 0
            self.success_per_token = np.zeros(self.num_tokens)
            self.failure_per_token = np.zeros(self.num_tokens)
            self.total_turns = 0
            self.fail_reason = { "no_tokens": 0, "wrong_winner": 0, "shared_top": 0 }

    def step(self):
        self.reset()
        self.agents.do("reset")
        self.agents.shuffle_do("interact_do")

        if self.current_step % self.datacollector_step_size == 0:
            self.datacollector.collect(self)
        self.current_step += 1

    def step_unitary(self):
        # Pick speaker and hearer agent
        speaker_agent = self.random.choice(self.agents)
        # Make sure hearer agent does not equal speaker agent
        while True:
            hearer_agent = self.random.choice(self.agents)
            if speaker_agent != hearer_agent:
                break
        event_index = self.weighted_random_index()

        speaker_agent.interact(hearer_agent, event_index)

    def weighted_random_index(self):
        r = self.random.uniform(0, self.total_frequency)
        return next(i for i, cumulative_frequency in enumerate(self.cumulative_frequencies) if r < cumulative_frequency)

    def get_original_vector(self, token_index):
        return self.vectors[token_index, :]
    
    def do_repair(self, spoken_vector, token_index):
        # First, stack both vectors
        original_vector = self.get_original_vector(token_index)
        combined_vector = np.vstack((spoken_vector, original_vector)).mean(axis=0)

        # TODO: decide on better mechanism
        return combined_vector.mean(axis=0)