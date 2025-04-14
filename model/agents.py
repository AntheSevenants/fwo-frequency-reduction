import mesa
import math
import numpy as np

from model.helpers import add_value_to_row, add_noise, get_neighbours, get_neighbours_nearest
from model.types.neighbourhood import NeighbourhoodTypes
from model.types.production import ProductionModels
from model.types.reduction import ReductionModes
from model.types.feedback import FeedbackTypes
from model.types.repair import Repair

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
        # The model we are building has a maximum size. However, it is not yet maximum size at bootup time.
        # So I'm just keeping the flexible size code. Needs must :-).
        model_memory_size = self.model.initial_token_count * self.model.num_tokens

        if model_memory_size > self.model.memory_size:
            raise ValueError(f"Initial agent memory size is {model_memory_size}, exceeding the memory limit of {self.model.memory_size}")

        self.memory = np.full((model_memory_size, self.model.num_dimensions), np.nan)
        self.indices_in_memory = np.full(model_memory_size, np.nan, dtype=np.int64)
        self.last_used = np.full(self.model.memory_size, 0, dtype=np.int64)
        self.frequency_count = np.full(model_memory_size, 1, dtype=np.int64)

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

        # Now, delete all nans (though there should not be any nans!)
        self.memory = self.memory[~np.isnan(self.memory[:,1])]

        # To prevent certain model anomalies, we prefill the memory
        if self.model.prefill_memory:
            while self.memory.shape[0] < self.model.memory_size:
                random_index = self.model.weighted_random_index()
                random_vector = self.model.get_original_vector(random_index)
                noisy_vector = add_noise(random_vector)

                self.commit_to_memory(noisy_vector, random_index)

    def update_last_used(self, index=None):
        self.last_used[index] = self.model.current_step
    
    def commit_to_memory(self, vector, concept_index):
        # If the memory is full, we need to remove the oldest form eligible for removal
        if self.memory.shape[0] == self.model.memory_size:
            # We look for indices which are associated with tokens that have more than one exemplar in memory
            eligible_indices = [ i for i in range(self.indices_in_memory.size)
                                 if self.frequency_count[self.indices_in_memory[i]] > self.model.initial_token_count ]

            if not eligible_indices:
                # This should rarely happen if you always ensure every token retains at least one exemplar.
                raise Exception(f"Cannot remove an exemplar from memory as no types have frequencies over {self.model.initial_token_count}")
        
            # We choose the index which is the oldest
            remove_index = min(eligible_indices, key=lambda i: self.last_used[i])

            # Now, subtract one from the frequency counts for the associated token
            self.frequency_count[self.indices_in_memory[remove_index]] -= 1

            # And now we can replace the old exemplar with the new one!
            self.memory[remove_index, :] = vector
            self.indices_in_memory[remove_index] = concept_index
            self.update_last_used(remove_index)
            self.frequency_count[concept_index] += 1
        else:
            self.memory = np.vstack([self.memory, vector])
            self.indices_in_memory = np.append(self.indices_in_memory, concept_index)
            self.update_last_used()
            self.frequency_count[concept_index] += 1

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
        
        if self.model.production_model == ProductionModels.SINGLE_EXEMPLAR:
            spoken_token_vector = chosen_exemplar_vector
            # Update last used characteristics for this index
            self.update_last_used(chosen_exemplar_base_index)
        else:    
        # Now, we make neighbourhood around this exemplar
            if self.model.neighbourhood_type == NeighbourhoodTypes.SPATIAL: 
                speaker_neighbourhood_indices, speaker_weights = get_neighbours(self.memory, chosen_exemplar_vector, self.model.neighbourhood_size)
            elif self.model.neighbourhood_type == NeighbourhoodTypes.NEAREST:
                speaker_neighbourhood_indices, speaker_weights = get_neighbours_nearest(self.memory, chosen_exemplar_vector, self.model.neighbourhood_size)
            elif self.model.neighbourhood_type == NeighbourhoodTypes.WEIGHTED_NEAREST:
                speaker_neighbourhood_indices, speaker_weights = get_neighbours_nearest(self.memory, chosen_exemplar_vector, self.model.neighbourhood_size, weighted=True)

            # We select only the phonetic representations of the rows of this concept,
            # then stack everything into a single matrix ...
            speaker_selected_rows = self.memory[speaker_neighbourhood_indices]
            speaker_neighbourhood_matrix = np.vstack(speaker_selected_rows)
            # ... and then we turn it into a single representation that we can emit
            spoken_token_vector = np.average(speaker_neighbourhood_matrix, axis=0, weights=speaker_weights)

            for speaker_neighbourhood_index in speaker_neighbourhood_indices:
                # Update last used characteristics for this index
                self.update_last_used(speaker_neighbourhood_index)

        # - - - - - - - - -
        # R E D U C T I O N
        # - - - - - - - - -

        # I currently don't know whether the whole L1 thing is even necessary for reduction.
        # So I programmed two modes: a success-dependent one and a "dumb" one which just always reduces under a certain threshold

        if self.model.reduction_mode == ReductionModes.ALWAYS:
            reduction_prob = self.model.reduction_prior
        elif self.model.reduction_mode == ReductionModes.SUCCESS_DEPENDENT:
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
            mu_param = 0.1      # strength of the success factor; adjust as needed
            intercept = 7

            # Compute the reduction probability. Here a sigmoid function is used to map the combined signal
            reduction_prob = 1 / (1 + np.exp(-lambda_param * l1_penalty + intercept - mu_param * historical_success))
            # print(f"Reduction probability: {reduction_prob:.3f}")
        else:
            raise ValueError("Reduction mode not recognised")

        # Decide whether to apply reduction based on the computed probability.
        if self.model.random.random() < reduction_prob and not self.model.disable_reduction:
            # Apply L1-based soft thresholding to encourage further sparsity
            threshold = 15  # the threshold value can be adjusted
            spoken_token_vector = np.maximum(spoken_token_vector - threshold, threshold)
            # print("Reduction applied: Token vector sparsified.")
        else:
            pass
            # print("No reduction applied.")

        # - - - - - - - - -
        # R E C E P T I O N
        # - - - - - - - - -

        for attempt in range(1):
            # Now, we see what tokens are in the neighbourhood for the hearer in the spoken region
            if self.model.neighbourhood_type == NeighbourhoodTypes.SPATIAL:
                hearer_neighbourhood_indices, hearer_weights = get_neighbours(hearer_agent.memory, spoken_token_vector, self.model.neighbourhood_size)
            elif self.model.neighbourhood_type == NeighbourhoodTypes.NEAREST:
                hearer_neighbourhood_indices, hearer_weights = get_neighbours_nearest(hearer_agent.memory, spoken_token_vector, self.model.neighbourhood_size)
            elif self.model.neighbourhood_type == NeighbourhoodTypes.WEIGHTED_NEAREST:
                hearer_neighbourhood_indices, hearer_weights = get_neighbours_nearest(hearer_agent.memory, spoken_token_vector, self.model.neighbourhood_size, weighted=True)

            # We check what concepts they are connected to
            hearer_concept_values = hearer_agent.indices_in_memory[hearer_neighbourhood_indices]
            unique, counts = np.unique(hearer_concept_values, return_counts=True)

            # Update last used indices for forms that were activated upon reception
            for hearer_neighbourhood_index in hearer_neighbourhood_indices:
                self.update_last_used(hearer_neighbourhood_index)

            # Set communication to false to begin with
            communication_successful = False
            check_right_form = False

            # If no tokens were found within the vicinity, communication has failed
            if len(unique) == 0:
                # print("No tokens in this neighbourhood")
                heard_concept_index = None
                self.model.fail_reason["no_tokens"] += 1
                break
            elif len(counts) > 1:
                # We check what form is the most represented in the neighbourhood
                sorted_indices = np.argsort(counts)[::-1]
                unique = unique[sorted_indices]
                counts = counts[sorted_indices]
                
                # We need to check if two forms share the top spot
                if counts[0] > counts[1]:
                    heard_concept_index = int(unique[0])
                    check_right_form = True
                else:
                    heard_concept_index = None
                    self.model.fail_reason["shared_top"] += 1

                    if self.model.repair == Repair.REPAIR:
                        # Try again
                        spoken_token_vector = self.model.do_repair(spoken_token_vector, event_index)
                        continue
                    elif self.model.repair == Repair.NO_REPAIR:
                        break
                    else:
                        raise ValueError("Repair option not recognised")
            else:
                heard_concept_index = int(unique[0])
                check_right_form = True
    
            if check_right_form:
                if event_index == heard_concept_index:
                    communication_successful = True
                else:
                    self.model.fail_reason["wrong_winner"] += 1
            break

        # print(f"Communication successful: {communication_successful}")
        
        if communication_successful:
            self.commit_to_memory(spoken_token_vector, heard_concept_index)
            
            # Increase the historical success score for this event (or token).
            # This could be a simple counter or a more elaborate moving average.
            self.success_history[event_index] = self.success_history.get(event_index, 0) + 1

            self.model.successful_turns += 1
        else:
            # If there is no feedback mechanism, also save form when speaker misheard
            if self.model.feedback_type == FeedbackTypes.NO_FEEDBACK and heard_concept_index is not None:
                self.commit_to_memory(spoken_token_vector, heard_concept_index)


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