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

    def interact(self, other_agent, event_index):
        # Define a dummy outcome for the communication
        communication_successful = False

        # Set this to the speaker
        self.speaking = True

        print(f"Event index: {event_index}")

        # Decide upon the form that the speaker will use to communicate about this event
        # TODO: multiple choices here: n nearest neighbour OR fixed size or?
        # TODO: just figuring things out

        # We get the indices of all vectors pertaining to the communicated event
        matching_token_indices = np.where(self.indices_in_memory == event_index)[0].tolist()
        # Then, we pick a random exemplar from this list
        chosen_exemplar_base = self.model.random.choice(matching_token_indices)
        
        # Now, we make neighbourhood around this exemplar
        neighbourhood = get_neighbours(self.memory, chosen_exemplar_base, 0.5)

        print(neighbourhood)

    def reset(self):
        self.speaking = False
        self.hearing = False