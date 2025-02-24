import mesa
import numpy as np

from helpers import add_value_to_row, add_noise

class ReductionAgent(mesa.Agent):
    """A speaker in the model"""

    def __init__(self, model, vocabulary=None):
        # Pass the parameters to the parent class.
        super().__init__(model)

        self.unreduced_vocabulary = self.model.vectors.copy()
        self.vocabulary = self.model.vectors.copy() # weird behaviour when passed through the agentset
        # Turn vocabulary into exemplar memory
        # New shape: token count x exemplar length x dimension count
        self.vocabulary = np.tile(self.vocabulary[:, np.newaxis, :], (1, self.model.exemplar_memory_n, 1))
        self.speaking = False
        self.hearing = False

        self.turns = []
        # Create a success matrix with shape: token count x memory count
        # Fill it with ones (we are optimistic about communication upfront :-) )
        self.turns_per_word = np.full((self.model.num_tokens, self.model.last_n_turns), 1)
        # Also keep a memory of whether reduction was applied or not
        # Same dimensions, and fill it with zeroes (we do not assume any reduction in the past)
        self.reduction_history = np.full((self.model.num_tokens, self.model.last_n_turns), 0)
        # Keep a memory of the last time a form was activated
        self.activation_history = np.zeros((self.model.num_tokens, 1))

        #print(self.vocabulary.shape)

    def reset(self):
        self.speaking = False
        self.hearing = False

    def recover_vector_from_exemplar_memory(self, token_index, exemplar_index):
        return self.vocabulary[token_index, exemplar_index, :].copy()

    def interact(self):
        # Define a dummy outcome for the communication
        communication_successful = False

        # Set this to the speaker
        self.speaking = True

        # Choose a position (but taking into account the frequency of the vocabulary if necessary!)
        if self.model.zipfian_token_distribution:
            random_index = self.model.weighted_random_index()
        else:
            random_index = self.model.random.randrange(0, self.model.num_tokens)

        exemplar_index = self.model.random.randrange(0, self.model.exemplar_memory_n)

        # Get the right vector from the vocabulary
        random_vector = self.recover_vector_from_exemplar_memory(random_index, exemplar_index)

        # print(random_vector)
    
        # Get the other information
        token = self.model.tokens[random_index]
        frequency = self.model.frequencies[random_index]
        percentile = self.model.percentiles[random_index]

        # Save that this token was chosen
        self.model.tokens_chosen[token] += 1

        # Compute the reduction probability by multiplying the random chance by the chance depending on the percentile
        # TODO perhaps later implement this prior
        # Also compute the ratio of communicative success, which we will turn into a probability
        communicative_success_probability = self.compute_communicative_success_probability_token(random_index)
        reduction_success = self.compute_reduction_success(random_index)
        reduction_probability = self.model.random.uniform(0, 1)
        computed_reduction_prior = reduction_success
        non_zero_indices = np.nonzero(random_vector)[0]

        is_reducing = False
        # With prevention for zeroing out vectors leaning on just one dimensions
        if reduction_probability < computed_reduction_prior and len(non_zero_indices) > self.model.lower_dimension_limit and not self.model.disable_reduction:
            is_reducing = True

            # Q: should random dimension be full?
            # A: yes, otherwise there is no reduction going on!
            # Pick a random dimension from the non-zero dimensions
            random_dimension_index = np.random.choice(non_zero_indices.tolist())
            # Notice that I'm keeping the matrices two-dimensional because of performance reasons :)
            random_vector[random_dimension_index] = 0

            # Remove item from non zero indices list
            non_zero_indices = np.delete(non_zero_indices, np.where(non_zero_indices == random_dimension_index))

            # print(random_vector)
        
        other_agent = None
        while other_agent is None:
            other_agent = self.random.choice(self.model.agents)
            if other_agent is self:
                other_agent = None

        other_agent.hearing = True

        # Add perceptual noise
        random_vector_w_noise = add_noise(random_vector)

        attempts = 1
        while attempts <= 3:
            heard_index = self.model.find_nearest_neighbour_index(other_agent.vocabulary, random_vector_w_noise)
            communication_successful = heard_index == random_index

            # Save data for the confusion matrix
            self.model.confusion_matrix[random_index][heard_index] += 1

            # If communication is successful, put the reduced vector in the vocabulary
            if communication_successful:
                self.vector_to_exemplar_memory(random_index, random_vector_w_noise)
                self.model.successful_turns += 1
            else:
                self.model.failed_turns += 1

            self.model.total_turns += 1
            self.model.turns.append(communication_successful)

            # This is used to keep track of the communicative success of the agent
            self.record_turn(random_index, communication_successful)
            self.record_reduction(random_index, is_reducing)

            # Attempt repair
            if not communication_successful:
                # Compute yes zero indices
                total_indices = np.arange(0, self.model.num_dimensions)
                zero_indices = np.setdiff1d(total_indices, non_zero_indices)

                # If no more dimensions left to restore, communication has simply failed
                if len(zero_indices) == 0:
                    break

                # For now, repair with just one dimension, but more radical may be possible later (TODO)
                random_dimension_index = np.random.choice(tuple(zero_indices))

                self.model.total_repairs += 1

                # Reinstate the original value
                random_vector[random_dimension_index] = self.unreduced_vocabulary[random_index, random_dimension_index].copy()

                # Add this index from non zero indices
                non_zero_indices = np.append(non_zero_indices, random_dimension_index)

                # Add noise again
                random_vector_w_noise = add_noise(random_vector)

                # State that we are NOT reducing anymore
                is_reducing = False
            else:
                # Only save activation if communication was successful
                self.record_activation(random_index)
                break
        
            attempts += 1

    def vector_to_exemplar_memory(self, token_index, vector):
        self.vocabulary[token_index, :-1, :] = self.vocabulary[token_index, 1:, :]
        self.vocabulary[token_index, -1, :] = vector

    def record_turn(self, token_index, communication_successful):
        add_value_to_row(self.turns_per_word, token_index, int(communication_successful))

    def record_reduction(self, token_index, is_reducing):
        add_value_to_row(self.reduction_history, token_index, int(is_reducing))

    def record_activation(self, token_index):
        self.activation_history[token_index] = self.model.current_step

    def compute_reduction_success(self, token_index):
        EPSILON = 0.0001
        K = 4
        THETA = 0.5
        LAMBDA = 1
        DELTA = 0.005

        # Find out how long ago this form was last activated for this agent
        activation_delta = self.model.current_step - self.activation_history[token_index]
        decay_factor = np.exp(-DELTA * activation_delta)
        # Find where reductions occurred
        reduction_indices = np.where(self.reduction_history[token_index] == 1)[0]
        # Count successful reductions
        successful_reductions = np.sum(self.turns_per_word[token_index, reduction_indices])
        # Count total reductions
        total_reductions = len(reduction_indices)

        # Calculate the proportion of successful reductions
        proportion_successful = ((successful_reductions) + EPSILON) / (len(reduction_indices) + EPSILON)
        # Decay
        proportion_successful = proportion_successful * decay_factor
        uncertainty = np.sqrt((proportion_successful * (1 - proportion_successful)) + EPSILON / ((total_reductions) + EPSILON))
        p_reduce = (1 / (1 + np.exp(-K * (proportion_successful - THETA - LAMBDA * uncertainty))))

        return p_reduce

    def compute_communicative_success_probability_token(self, token_index):
        # If last n turns disabled, disable communicative memory and just always return 1
        if self.model.last_n_turns < 0:
            return 1
        # Else, compute the ratio of the past few turns

        # Get the last n turns of the agent's memory
        success_ratio = self.turns_per_word[token_index,].mean()
        probability = max(0, 2 * (success_ratio - 0.5))

        return probability

    def compute_communicative_success_probability(self):
        # If last n turns, disable communicative memory and just always return 1
        # If there are no turns to compute from, also return 1
        if self.model.last_n_turns < 0 or len(self.turns) == 0:
            return 1
        # Else, compute the ratio of the past few turns

        # Get the last n turns of the agent's memory
        memory_turns = self.turns[-self.model.last_n_turns:]
        success_ratio = memory_turns.count(True) / len(memory_turns)
        probability = max(0, 2 * (success_ratio - 0.5))

        return probability