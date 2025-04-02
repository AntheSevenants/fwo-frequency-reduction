import mesa
import math
import numpy as np

from helpers import add_value_to_row, add_noise

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

        for token_index in range(self.model.num_tokens):
            # Get the vector from memory and add noise
            vector = self.model.get_original_vector(token_index)
            noisy_vector = add_noise(vector)

            # Save to memory
            i = token_index
            self.memory[i, :] = noisy_vector
            self.indices_in_memory[i] = token_index

    def reset(self):
        self.speaking = False
        self.hearing = False

    def interact(self):
        pass