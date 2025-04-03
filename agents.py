import mesa
import math
import numpy as np

from helpers import add_value_to_row, add_noise, get_neighbours

class ReductionAgent(mesa.Agent):
    """A speaker in the model"""

    def __init__(self, model, vocabulary=None):
        # Pass the parameters to the parent class.
        super().__init__(model)

        self.init_memory()

        self.speaking = False
        self.hearing = False

        self.success_history = {}

        #print(self.vocabulary.shape)

    def init_memory(self):
        # We create a memory with a very large size (won't be used completely)
        self.memory = np.full((self.model.memory_size, self.model.num_dimensions), np.nan)
        self.indices_in_memory = np.full(self.model.memory_size, np.nan)

        i = 0
        for token_index in range(self.model.num_tokens):
            # Get the vector from memory and add noise
            vector = self.model.get_original_vector(token_index)

            # Add each token twice
            for j in range(self.model.initial_token_count):
                noisy_vector = add_noise(vector)

                # Save to memory
                self.memory[i, :] = noisy_vector
                self.indices_in_memory[i] = token_index

                i += 1

        self.current_memory_index = i

    def commit_to_memory(self, vector, concept_index):
        self.memory[self.current_memory_index, :] = vector
        self.indices_in_memory[self.current_memory_index] = concept_index

        self.current_memory_index += 1

    def interact_do(self):
        event_index = self.model.weighted_random_index()
        
        while True:
            hearer_agent = self.random.choice(self.model.agents)
            if self != hearer_agent:
                break

        self.interact(hearer_agent, event_index)

    def interact(self, hearer_agent, event_index):
        # Define a dummy outcome for the communication
        communication_successful = False

        # - - - - - - - - - -
        # P R O D U C T I O N
        # - - - - - - - - - -

        # Set this to the speaker
        self.speaking = True

        # print(f"Event index: {event_index}")

        # Decide upon the form that the speaker will use to communicate about this event
        # TODO: multiple choices here: n nearest neighbour OR fixed size or?
        # TODO: just figuring things out

        # We get the indices of all vectors pertaining to the communicated event
        matching_token_indices = np.where(self.indices_in_memory == event_index)[0].tolist()
        # Then, we pick a random exemplar from this list
        chosen_exemplar_base_index = self.model.random.choice(matching_token_indices)
        chosen_exemplar_vector = self.memory[chosen_exemplar_base_index]
        
        # Now, we make neighbourhood around this exemplar
        speaker_neighbourhood_indices = get_neighbours(self.memory, chosen_exemplar_vector, 0.5)
        # We select only the phonetic representations of the rows of this concept,
        # then stack everything into a single matrix ...
        speaker_selected_rows = self.memory[speaker_neighbourhood_indices]
        speaker_neighbourhood_matrix = np.vstack(speaker_selected_rows)
        # ... and then we turn it into a single representation that we can emit
        spoken_token_vector = np.mean(speaker_neighbourhood_matrix, axis=0)

        # - - - - - - - - -
        # R E D U C T I O N
        # - - - - - - - - -

        # Here we apply a reduction process which is influenced by both the L1 penalty
        # (i.e. encouraging sparsity) and the past communicative success of this event.

        # Compute the L1 penalty (the sum of absolute values)
        l1_penalty = np.sum(np.abs(spoken_token_vector))
        # print(f"L1 penalty: {l1_penalty}")

        # Retrieve historical communicative success for this event token
        # Assume self.success_history is a dict that maps event indices to a success score.
        # Default to 1 if there's no history yet.
        historical_success = self.success_history.get(event_index, 1)

        # Define parameters that weigh the L1 penalty and the historical success
        lambda_param = 1.0  # strength of sparsity effect; adjust as needed
        mu_param = 1.0      # strength of the success factor; adjust as needed

        # Compute the reduction probability. Here a sigmoid function is used to map the combined signal
        reduction_prob = 1 / (1 + np.exp(lambda_param * l1_penalty - mu_param * historical_success))
        # print(f"Reduction probability: {reduction_prob:.3f}")

        # Decide whether to apply reduction based on the computed probability.
        if self.model.random.random() < reduction_prob:
            # Apply L1-based soft thresholding to encourage further sparsity
            threshold = 0.1  # the threshold value can be adjusted
            spoken_token_vector = np.sign(spoken_token_vector) * np.maximum(np.abs(spoken_token_vector) - threshold, 0)
            # print("Reduction applied: Token vector sparsified.")
        else:
            pass
            # print("No reduction applied.")

        # - - - - - - - - -
        # R E C E P T I O N
        # - - - - - - - - -

        # Now, we see what tokens are in the neighbourhood for the hearer in the spoken region
        hearer_neighbourhood_indices = get_neighbours(hearer_agent.memory, spoken_token_vector, 0.5)
        # We check what concepts they are connected to
        hearer_concept_values = hearer_agent.indices_in_memory[hearer_neighbourhood_indices]
        unique, counts = np.unique(hearer_concept_values, return_counts=True)

        # If no tokens were found within the vicinity, communication has failed
        if len(unique) == 0:
            # print("No tokens in this neighbourhood")
            heard_concept_index = None
        elif len(counts) > 1:
            # We check what form is the most represented in the neighbourhood
            sorted_indices = np.argsort(counts)[::-1]
            unique = unique[sorted_indices]
            counts = counts[sorted_indices]
            
            # We need to check if two forms share the top spot
            if counts[0] > counts[1]:
                heard_concept_index = int(unique[0])
            else:
                heard_concept_index = None
        else:
            heard_concept_index = int(unique[0])

        # Communication is successful if the right concept is identified
        communication_successful = event_index == heard_concept_index
        # print(f"Communication successful: {communication_successful}")
        
        # TODO: For now, I'm saving a form if it was successfully recognised by the hearer
        if communication_successful:
            self.commit_to_memory(spoken_token_vector, heard_concept_index)
            
            # Increase the historical success score for this event (or token).
            # This could be a simple counter or a more elaborate moving average.
            self.success_history[event_index] = self.success_history.get(event_index, 0) + 1

            self.model.successful_turns += 1
        else:
            # Optionally, penalize if the communication failed.
            self.success_history[event_index] = max(self.success_history.get(event_index, 0) - 1, 0)

            self.model.failed_turns += 1
    
        if heard_concept_index is not None:
            # Save data for the confusion matrix
            self.model.confusion_matrix[event_index][heard_concept_index] += 1

        self.model.total_turns += 1

    def reset(self):
        self.speaking = False
        self.hearing = False