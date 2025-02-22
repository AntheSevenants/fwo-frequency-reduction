import mesa
import numpy as np

class ReductionAgent(mesa.Agent):
    """A speaker in the model"""

    def __init__(self, model, vocabulary=None):
        # Pass the parameters to the parent class.
        super().__init__(model)

        self.vocabulary = self.model.vectors.copy() # weird behaviour when passed through the agentset
        self.speaking = False
        self.hearing = False

        self.turns = []
        self.turns_per_word = [ [] for token in self.model.tokens ]

        #print(self.vocabulary.shape)

    def reset(self):
        self.speaking = False
        self.hearing = False

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

        # Get the right vector from the vocabulary
        random_vector = self.vocabulary[random_index, :].copy()

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
        reduction_probability = self.model.random.uniform(0, 1)
        computed_reduction_prior = communicative_success_probability
        non_zero_indices = np.nonzero(random_vector)[1]

        is_reducing = False
        # With prevention for zeroing out vectors leaning on just one dimensions
        if reduction_probability < self.model.reduction_prior and len(non_zero_indices) > 1:
            is_reducing = True

            # Q: should random dimension be full?
            # A: yes, otherwise there is no reduction going on!
            # Pick a random dimension from the non-zero dimensions
            random_dimension_index = np.random.choice(non_zero_indices.tolist())
            # Notice that I'm keeping the matrices two-dimensional because of performance reasons :)
            random_vector[0,random_dimension_index] = 0
        
        other_agent = None
        while other_agent is None:
            other_agent = self.random.choice(self.model.agents)
            if other_agent is self:
                other_agent = None

        other_agent.hearing = True
        heard_index = self.model.find_nearest_neighbour_index(other_agent.vocabulary, random_vector)
        communication_successful = heard_index == random_index

        # Save data for the confusion matrix
        self.model.confusion_matrix[random_index][heard_index] += 1

        # If communication is successful, put the reduced vector in the vocabulary
        if communication_successful:
            self.vocabulary[random_index,:] = random_vector[0,:]
            self.model.successful_turns += 1
        else:
            self.model.failed_turns += 1

        self.model.total_turns += 1
        self.model.turns.append(communication_successful)

        # This is used to keep track of the communicative success of the agent
        self.record_turn(random_index, communication_successful)
        #self.turns.append(communication_successful)

    def record_turn(self, token_index, communication_successful):
        self.turns_per_word[token_index].append(communication_successful)

    def compute_communicative_success_probability_token(self, token_index):
        # If last n turns, disable communicative memory and just always return 1
        # If there are no turns to compute from, also return 1
        if self.model.last_n_turns < 0 or len(self.turns) == 0:
            return 1
        # Else, compute the ratio of the past few turns

        # Get the last n turns of the agent's memory
        memory_turns = self.turns_per_word[token_index][-self.model.last_n_turns:]
        success_ratio = memory_turns.count(True) / len(memory_turns)
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