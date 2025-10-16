
import mesa
import math
import numpy as np

from model.agents import ReductionAgent
from model.helpers import compute_communicative_success, compute_communicative_failure, compute_mean_non_zero_ratio, compute_tokens_chosen, distances_to_probabilities_softmax, distances_to_probabilities_linear, compute_confusion_matrix, compute_average_vocabulary, compute_average_communicative_success_probability, compute_mean_communicative_success_per_token, compute_mean_reduction_per_token, compute_repairs, compute_mean_agent_l1, compute_mean_token_l1, compute_fail_reason, compute_mean_exemplar_count, compute_average_vocabulary_flexible, compute_communicative_success_per_token, compute_communicative_success_macro_average, compute_token_good_origin, compute_mean_exemplar_age, compute_full_vocabulary, compute_concept_stack, compute_full_vocabulary_ownership_stack, compute_outcomes, compute_reduction_success
from model.types.neighbourhood import NeighbourhoodTypes
from model.types.production import ProductionModels
from model.types.reduction import ReductionModes, ReductionMethod
from model.types.feedback import FeedbackTypes
from model.types.who_saves import WhoSaves
from model.types.repair import Repair
from model.helpers import load_vectors, load_info, generate_word_vectors, generate_quarter_circle_vectors

class ReductionModel(mesa.Model):
    """A model of Joan Bybee's *reducing effect*"""

    def __init__(self, num_agents=50, num_dimensions=50, num_tokens=100, reduction_prior = 0.5, memory_size=1000, toroidal=False, value_ceil=100, success_memory_size=20, initial_token_count=2, prefill_memory=True, disable_reduction=False, neighbourhood_type=NeighbourhoodTypes.SPATIAL, neighbourhood_size=0.5, production_model=ProductionModels.SINGLE_EXEMPLAR, reduction_mode=ReductionModes.ALWAYS, reduction_method=ReductionMethod.SOFT_THRESHOLDING, reduction_strength=15, feedback_type=FeedbackTypes.FEEDBACK, repair=Repair.NO_REPAIR, confidence_threshold=0, speaker_confidence_threshold=0, self_check=False, neighbourhood_step_size=0, max_turns=1, jumble_vocabulary=False, zipfian_sampling=True, who_saves=None, exemplar_hearing_equals_use=False, datacollector_step_size=100, light_serialisation=True, dynamic_neighbourhood_size=False, early_stop=False, seed=None):
        super().__init__(seed=seed)

        self.num_agents = num_agents
        self.reduction_prior = reduction_prior
        self.disable_reduction = disable_reduction
        self.memory_size = memory_size
        self.value_ceil = value_ceil
        self.success_memory_size = success_memory_size
        self.initial_token_count = initial_token_count
        self.prefill_memory = prefill_memory
        self.seed = seed
        self.current_step = 0
        self.datacollector_step_size = datacollector_step_size
        self.early_stop = early_stop

        self.neighbourhood_size = neighbourhood_size
        self.neighbourhood_type = neighbourhood_type
        self.production_model = production_model
        self.reduction_mode = reduction_mode
        self.reduction_method = reduction_method
        self.reduction_strength = reduction_strength
        self.feedback_type = feedback_type
        self.repair = repair

        # Confidence treshold
        self.confidence_threshold = confidence_threshold
        self.confidence_judgement = confidence_threshold > 0
        self.speaker_confidence_threshold = speaker_confidence_threshold
        self.speaker_confidence_judgement = speaker_confidence_threshold > 0

        # Marginal inhibition threshold
        self.self_check = self_check

        # Whether to save entire vocabularies to disk (ideally not)
        self.light_serialisation = light_serialisation

        # Neighbourhood step up size
        self.neighbourhood_step_size = neighbourhood_step_size
        self.grow_neighbourhood = neighbourhood_step_size > 0

        # Whether to shuffle vectors at random
        self.jumble_vocabulary = jumble_vocabulary

        # Whether to sample concepts according to the Zipfian distribution
        self.zipfian_sampling = zipfian_sampling

        # Maximum number of turns
        self.max_turns = max_turns

        # Toroidal space
        self.toroidal = toroidal

        # Who saves?
        self.who_saves = who_saves

        # Exemplar hearing equals use
        self.exemplar_hearing_equals_use = exemplar_hearing_equals_use

        #
        # Visualisation stuff
        #

        # The grid is just for visualisation purposes, it doesn't do anything
        self.grid = mesa.space.SingleGrid(10, 10, True)

        num_tokens_override = 100
        tokens, frequencies, percentiles, ranks = load_info(f"vectors/theoretical-percentile-info-{num_tokens_override}.tsv", theoretical=True)

        # Overwrite vectors with my own
        if not self.toroidal:
            vectors = generate_word_vectors(vocabulary_size=len(tokens), dimensions=num_dimensions, floor=reduction_strength, seed=seed)
        else:
            tokens = tokens[0:num_tokens]
            frequencies = frequencies[0:num_tokens]
            percentiles = percentiles[0:num_tokens]
            ranks = ranks[0:num_tokens]

            vectors, angles = generate_quarter_circle_vectors(self.value_ceil - 20, num_points=num_tokens + 1)
            np.random.shuffle(vectors)

        self.vectors = vectors
        self.tokens = tokens
        self.frequencies = frequencies
        self.percentiles = percentiles
        self.ranks = ranks
        self.num_tokens = num_tokens
        self.num_dimensions = num_dimensions
        self.lower_dimension_limit = math.floor(self.num_dimensions / 10)

        self.neighbourhood_size = neighbourhood_size
        if dynamic_neighbourhood_size:
            self.neighbourhood_size = self.num_dimensions * 1.5

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

        self.running = True

        # print(vectors.shape)

        # We copy the vectors to the vocabulary of the agent
        agents = ReductionAgent.create_agents(model=self, n=num_agents)
        # Then, we just put the little guys on the grid sequentially
        # I don't care for the order, it means nothing anyway
        for index, a in enumerate(agents):
            # Calculate the position based on the index
            i = index % self.grid.width
            j = index // self.grid.width

            self.agents[index].no = index
            
            self.grid.place_agent(a, (i, j))

        model_reporters = {
            "communicative_success": compute_communicative_success,
            "communicative_failure": compute_communicative_failure,
            "reduction_success": compute_reduction_success,
            "mean_agent_l1": compute_mean_agent_l1,
            "mean_token_l1": compute_mean_token_l1,
            "confusion_matrix": compute_confusion_matrix,
            "fail_reason": compute_fail_reason,
            "outcomes": compute_outcomes,
            "mean_exemplar_count": compute_mean_exemplar_count,
            "success_per_token": compute_communicative_success_per_token,
            "communicative_success_macro": compute_communicative_success_macro_average,
            "token_good_origin": compute_token_good_origin,
            "mean_exemplar_age": compute_mean_exemplar_age,
        }

        # Include full vocabulary if needed (I hope not)
        if not self.light_serialisation:
            model_reporters = { **model_reporters,
                "average_vocabulary": compute_average_vocabulary_flexible,
                "full_vocabulary": compute_full_vocabulary,
                "full_indices": compute_concept_stack,
                "full_vocabulary_owernship": compute_full_vocabulary_ownership_stack
            }

        self.datacollector = mesa.DataCollector(
            model_reporters=model_reporters
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
            self.fail_reason = { "no_tokens": 0, "wrong_winner": 0, "shared_top": 0, "not_confident": 0 }
            self.outcomes = { "no_tokens": 0, "wrong_winner": 0, "shared_top": 0, "not_confident": 0, "success": 0 }
            self.reduced_turns = 0
            self.successful_reduced_turns = 0

    def step(self):
        self.reset()
        self.agents.do("reset")
        self.agents.shuffle_do("interact_do")

        if self.current_step % self.datacollector_step_size == 0:
            self.datacollector.collect(self)

            if self.early_stop:
                mean_token_l1 = compute_mean_token_l1(self)
                if mean_token_l1[0] == self.reduction_strength * self.num_dimensions:
                    self.running = False

        self.current_step += 1

    def step_unitary(self):
        # Pick speaker and hearer agent
        speaker_agent = self.random.choice(self.agents)
        # Make sure hearer agent does not equal speaker agent
        while True:
            hearer_agent = self.random.choice(self.agents)
            if speaker_agent != hearer_agent:
                break
        if self.zipfian_sampling:
            event_index = self.weighted_random_index()
        else:
            event_index = self.true_random_index()

        speaker_agent.interact(hearer_agent, event_index)

    def weighted_random_index(self):
        r = self.random.uniform(0, self.total_frequency)
        return next(i for i, cumulative_frequency in enumerate(self.cumulative_frequencies) if r < cumulative_frequency)
    
    def linear_random_index(self):
        # We make a fake linear frequency drop-off
        max_frequency = self.num_tokens
        frequencies = [ max_frequency - index for index in range(self.num_tokens) ]
        total_frequency = np.sum(frequencies)
        cumulative_frequencies = np.cumsum(frequencies)

        # Apply the same logic as for the Zipfian drop-off
        r = self.random.uniform(0, total_frequency)
        return next(i for i, cumulative_frequency in enumerate(cumulative_frequencies) if r < cumulative_frequency)

    def true_random_index(self):
        return self.random.randrange(0, self.num_tokens, 1)

    def get_original_vector(self, token_index, override_matrix=None):
        if override_matrix is not None:
            return override_matrix[token_index, :]
        else:
            return self.vectors[token_index, :]
    
    def register_outcome(self, outcome, success=False):
        if not success:
            self.fail_reason[outcome] += 1
        
        self.outcomes[outcome] += 1