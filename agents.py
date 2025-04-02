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

    def interact(self, hearer_agent, event_index):
        # Define a dummy outcome for the communication
        communication_successful = False

        # - - - - - - - - - -
        # P R O D U C T I O N
        # - - - - - - - - - -

        # Set this to the speaker
        self.speaking = True

        print(f"Event index: {event_index}")

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
        # R E C E P T I O N
        # - - - - - - - - -

        # Now, we see what tokens are in the neighbourhood for the hearer in the spoken region
        hearer_neighbourhood_indices = get_neighbours(hearer_agent.memory, spoken_token_vector, 0.5)
        # We check what concepts they are connected to
        hearer_concept_values = self.indices_in_memory[hearer_neighbourhood_indices]
        unique, counts = np.unique(hearer_concept_values, return_counts=True)

        # If no tokens were found within the vicinity, communication has failed
        if len(unique) == 0:
            print("No tokens in this neighbourhood")
            heard_concept_index = None
        
        # We check what form is the most represented in the neighbourhood
        sorted_indices = np.argsort(counts)[::-1]
        unique = unique[sorted_indices]
        counts = counts[sorted_indices]

        if len(counts) > 1:
            # We need to check if two forms share the top spot
            if counts[0] > counts[1]:
                heard_concept_index = unique[0]
            else:
                heard_concept_index = None
        else:
            heard_concept_index = unique[0]

        # Communication is successful if the right concept is identified
        communication_successful = event_index == heard_concept_index
        
        # TODO: For now, I'm saving a form if it was successfully recognised by the hearer
        if communication_successful:
            self.commit_to_memory(spoken_token_vector, heard_concept_index)

        print(f"Communication successful: {communication_successful}")

    def reset(self):
        self.speaking = False
        self.hearing = False