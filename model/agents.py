import mesa
import math
import warnings
import numpy as np

import model.reduction

from model.helpers import add_value_to_row, add_noise, get_neighbours, get_neighbours_nearest, counts_to_percentages
from model.types.neighbourhood import NeighbourhoodTypes
from model.types.production import ProductionModels
from model.types.reduction import ReductionModes, ReductionMethod
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

        self.memory = np.full((0, self.model.num_dimensions), np.nan)
        self.indices_in_memory = np.full(0, np.nan, dtype=np.int64)
        self.indices_per_token = [ [] for token in range(self.model.num_tokens) ]
        self.last_used = np.full(0, 0, dtype=np.int64)
        self.frequency_count = np.full(self.model.num_tokens, 0, dtype=np.int64)
        self.token_good_origin = np.full(0, np.nan, dtype=np.int64)
        self.num_exemplars_in_memory = 0
        self.success_memory = np.full((self.model.num_tokens, self.model.success_memory_size), np.nan)

        for token_index in range(self.model.num_tokens):
            # Get the vector from memory and add noise
            vector = self.model.get_original_vector(token_index)

            # Add each token twice
            for j in range(self.model.initial_token_count):
                noisy_vector = add_noise(vector, ceiling=self.model.value_ceil)

                # Save the vector to memory
                self.commit_to_memory(noisy_vector, token_index, good_origin=True)

        # Now, delete all nans (though there should not be any nans!)
        self.memory = self.memory[~np.isnan(self.memory[:,1])]

        # To prevent certain model anomalies, we prefill the memory
        if self.model.prefill_memory:
            while self.num_exemplars_in_memory < self.model.memory_size:
                random_index = self.model.weighted_random_index()
                random_vector = self.model.get_original_vector(random_index)
                noisy_vector = add_noise(random_vector)

                self.commit_to_memory(noisy_vector, random_index, good_origin=True)

    def update_last_used(self, index, grow=False):
        if not grow:
            self.last_used[index] = self.model.current_step
        else:
            self.last_used = np.append(self.last_used, self.model.current_step)

    def get_macro_communicative_success(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)

            tokenwise_means = np.nanmean(self.success_memory, axis=1)
            macro_mean = np.nanmean(tokenwise_means, axis=0)
        
            return macro_mean
    
    def commit_to_memory(self, vector, concept_index, good_origin=True):
        # If the memory is full, we need to remove the oldest form eligible for removal
        if self.num_exemplars_in_memory == self.model.memory_size:
            # We look for indices which are associated with tokens that have more than one exemplar in memory
            eligible_indices = []
            for token_index, frequency_count in enumerate(self.frequency_count):
                if frequency_count > 1:
                    eligible_indices += self.indices_per_token[token_index]

            if not eligible_indices:
                # This should rarely happen if you always ensure every token retains at least one exemplar.
                raise Exception(f"Cannot remove an exemplar from memory as no types have frequencies over {self.model.initial_token_count}")
        
            # We choose the index which is the oldest
            remove_index = min(eligible_indices, key=lambda i: self.last_used[i])

            # First, look up which concept this index is currently associated to
            old_concept_index = self.indices_in_memory[remove_index]
            # Then, subtract one from the frequency counts for the associated token
            self.frequency_count[old_concept_index] -= 1

            # We remove the index from the concept to tokens mapping
            self.indices_per_token[old_concept_index].remove(remove_index)

            # Add extra test to be absolutely certain that no token lacks exemplars
            for token_index, exemplar_indices in enumerate(self.indices_per_token):
                if len(exemplar_indices) == 0:
                    raise ValueError(f"Token {token_index} has zero associated exemplars. This should never happen!\n\
Old concept index was {old_concept_index}.\n\
{token_index} has {self.frequency_count[token_index]} registered associated exemplar(s)")

            # And now we can replace the old exemplar with the new one!
            self.memory[remove_index, :] = vector
            self.indices_in_memory[remove_index] = concept_index
            self.indices_per_token[concept_index].append(remove_index)
            self.token_good_origin[remove_index] = int(good_origin)
            self.update_last_used(remove_index)
            self.frequency_count[concept_index] += 1
        # Else, just add to the index
        else:
            # Because we're not removing a form, we don't have an index indicating where to add in the memory
            # (that currently consists of nothing but nans)
            # So we deduce the index from the number of forms currently in memory
            add_index = self.num_exemplars_in_memory

            self.memory = np.vstack([self.memory, vector])
            self.indices_in_memory = np.append(self.indices_in_memory, concept_index)
            self.indices_per_token[concept_index].append(add_index)
            self.token_good_origin = np.append(self.token_good_origin, int(good_origin))
            self.update_last_used(add_index, grow=True)
            self.frequency_count[concept_index] += 1

            self.num_exemplars_in_memory += 1

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
        matching_token_indices = self.indices_per_token[event_index].copy()
        if len(matching_token_indices) == 0:
            print(event_index)
            print(self.frequency_count)
            print(self.indices_per_token)
            raise IndexError("Binkebonke")
        # Then, we pick a random exemplar from this list
        chosen_exemplar_base_index = self.model.random.choice(matching_token_indices)
        # Remove the chosen exemplar from the matching list
        matching_token_indices.remove(chosen_exemplar_base_index)
        # Retrieve the vector from the memory
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

        reduction_strength = self.model.reduction_strength

        if self.model.reduction_mode == ReductionModes.ALWAYS:
            reduction_prob = self.model.reduction_prior
        elif self.model.reduction_mode == ReductionModes.L1_SUCCESS_DEPENDENT:
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
        elif self.model.reduction_mode == ReductionModes.SUCCESS_DEPENDENT_MACRO:
            reduction_prob = self.get_macro_communicative_success()
            # Initialisation issue
            if np.isnan(reduction_prob):
                reduction_prob = 1
        else:
            raise ValueError("Reduction mode not recognised")

        do_reduction = self.model.random.random() < reduction_prob and not self.model.disable_reduction
        # Decide whether to apply reduction based on the computed probability.
        if do_reduction:
            if self.model.reduction_method == ReductionMethod.DIMENSION_SCRAP:
                raise NotImplementedError("Dimension scrapping has not (yet) been reimplemented")
            elif self.model.reduction_method == ReductionMethod.SOFT_THRESHOLDING:
                # Apply L1-based soft thresholding to encourage further sparsity
                threshold = 15  # the threshold value can be adjusted
                spoken_token_vector = np.maximum(spoken_token_vector - reduction_strength, threshold)
            elif self.model.reduction_method == ReductionMethod.GAUSSIAN_MASK:
                spoken_token_vector = model.reduction.reduction_mask(self.model, spoken_token_vector, 15, width_ratio=0.5)
            elif self.model.reduction_method == ReductionMethod.ANGLE:
                spoken_token_vector = model.reduction.angle_reduction(spoken_token_vector, reduction_strength)

            # print("Reduction applied: Token vector sparsified.")
        else:
            pass
            # print("No reduction applied.")

        # - - - - - - - - -
        # R E C E P T I O N
        # - - - - - - - - -

        turns = 1
        neighbourhood_size = self.model.neighbourhood_size
        while turns <= self.model.max_turns:
            toroidal = True # TODO
            if toroidal:
                toroidal_size = self.model.value_ceil
            else:
                toroidal_size = None

            # print(f"Turn: {turns}")
            # Now, we see what tokens are in the neighbourhood for the hearer in the spoken region
            if self.model.neighbourhood_type == NeighbourhoodTypes.SPATIAL:
                hearer_neighbourhood_indices, hearer_weights = get_neighbours(hearer_agent.memory, spoken_token_vector, neighbourhood_size, toroidal_size)
            elif self.model.neighbourhood_type == NeighbourhoodTypes.NEAREST:
                hearer_neighbourhood_indices, hearer_weights = get_neighbours_nearest(hearer_agent.memory, spoken_token_vector, neighbourhood_size)
            elif self.model.neighbourhood_type == NeighbourhoodTypes.WEIGHTED_NEAREST:
                hearer_neighbourhood_indices, hearer_weights = get_neighbours_nearest(hearer_agent.memory, spoken_token_vector, neighbourhood_size, weighted=True)

            # We check what concepts they are connected to
            hearer_concept_values = hearer_agent.indices_in_memory[hearer_neighbourhood_indices]
            unique, counts = np.unique(hearer_concept_values, return_counts=True)

            # Update last used indices for forms that were activated upon reception
            for hearer_neighbourhood_index in hearer_neighbourhood_indices:
                self.update_last_used(hearer_neighbourhood_index)

            # Flag: is the communication successful? The hearer usually doesn't know this
            communication_successful = False
            # Flag: do we need to check whether the heard form is correct? The hearer usually doesn't know this
            check_right_form = False
            # Flag: should the repair mechanism come into play? Counts as a speaking turn ("what?")
            should_repair = False

            # No tokens were found within the vicinity
            if len(unique) == 0:
                heard_concept_index = None

                # If growing the neighbourhood is allowed, do so
                if self.model.grow_neighbourhood:
                    # Grow by the selected step size
                    neighbourhood_size += self.model.neighbourhood_step_size
                    # Then, exit the loop to try another attempt at nearest neighbour etc.
                    # This does not really count as a turn, since the speaker did not speak again
                    # print("Growing neighbourhood")
                    continue
                # Else, communication has failed!
                else:
                    heard_concept_index = None
                    self.model.register_outcome("no_tokens")
            # If multiple counts were found, we need to check a few things
            elif len(counts) > 1:
                # We check what form is the most represented in the neighbourhood
                sorted_indices = np.argsort(counts)[::-1]
                unique = unique[sorted_indices]
                counts = counts[sorted_indices]

                # For the confidence judgement, we turn the counts into percentages
                percentages = counts_to_percentages(counts)
                confident_judgement = percentages[0] >= self.model.confidence_threshold

                # If confidence judgement is turned on, and the judgemnt is not confident
                if self.model.confidence_judgement and not confident_judgement:
                    heard_concept_index = None
                    self.model.register_outcome("not_confident")

                    # Say: "what?"
                    should_repair = True
                    break # leave the if statement so the fail reason remains correct
                # Else, we check if two forms share the top spot
                # If not, all is good!
                elif counts[0] > counts[1]:
                    heard_concept_index = int(unique[0])
                    check_right_form = True
                # If the two top forms share the spot, communication also fails
                else:
                    heard_concept_index = None

                    self.model.register_outcome("shared_top")
                    should_repair = True
            # Only one type of form was heard, so the outcome is clear
            else:
                heard_concept_index = int(unique[0])
                check_right_form = True
                
            # If we got here, it means that there is a unique form that was chosen
            # (and also that we are confident enough if confidence is at play)
            if check_right_form:
                if event_index == heard_concept_index:
                    communication_successful = True
                    self.model.register_outcome("success", success=True)
                else:
                    self.model.register_outcome("wrong_winner")

                if FeedbackTypes.FEEDBACK and not communication_successful:
                    should_repair = True

            # There are multiple types of repair, so I've generalised quite a bit
            if should_repair:
                if self.model.repair == Repair.MEAN:
                    # Make the vector more full and try again
                    # First, stack both vectors
                    original_vector = self.get_original_vector(event_index)
                    combined_vector = np.vstack((spoken_token_vector, original_vector)).mean(axis=0)
                    spoken_token_vector = combined_vector.mean(axis=0)
                    
                    turns += 1
                    continue # forces another attempt
                elif self.model.repair == Repair.NEGATIVE_REDUCTION:
                    # Now, the speaker will neduce negatively
                    spoken_token_vector = np.maximum(spoken_token_vector + reduction_strength, 100)
                    turns += 1
                    continue # forces another attempt
                elif self.model.repair == Repair.NEGATIVE_REDUCTION_ANGLE:
                    spoken_token_vector = model.reduction.angle_reduction(spoken_token_vector, reduction_strength, negative=True)
                    turns += 1
                    continue # forces another attempt
                elif self.model.repair == Repair.PICK_ANOTHER:
                    # Pick another form, without reduction
                    # If no form is available, exit and give up
                    if len(matching_token_indices) > 0:
                        chosen_exemplar_base_index = self.model.random.choice(matching_token_indices)
                    else:
                        break # exit loop

                    # Remove the chosen exemplar from the matching list
                    matching_token_indices.remove(chosen_exemplar_base_index)
                    # Retrieve the vector from the memory
                    chosen_exemplar_vector = self.memory[chosen_exemplar_base_index]
                    turns += 1
                    continue # force another attempt
                elif self.model.repair == Repair.NO_REPAIR:
                    break
                else:
                    raise ValueError("Repair option not recognised")

            # That's all for this loop!
            turns += 1

            break

        # print(f"Communication successful: {communication_successful}")
        
        if communication_successful:
            self.commit_to_memory(spoken_token_vector, heard_concept_index, good_origin=True)
            
            # Increase the historical success score for this event (or token).
            # This could be a simple counter or a more elaborate moving average.
            self.success_history[event_index] = self.success_history.get(event_index, 0) + 1

            self.model.successful_turns += 1
            self.model.success_per_token[event_index] += 1
        else:
            # If there is no feedback mechanism, also save form when speaker misheard
            if self.model.feedback_type == FeedbackTypes.NO_FEEDBACK and heard_concept_index is not None:
                self.commit_to_memory(spoken_token_vector, heard_concept_index, good_origin=False)


            # Optionally, penalize if the communication failed.
            self.success_history[event_index] = max(self.success_history.get(event_index, 0) - 1, 0)

            self.model.failed_turns += 1
            self.model.failure_per_token[event_index] += 1

        # Save communicative success to the success memory
        # TODO: actually, the agent can't *really* know whether communication was successful, right?
        add_value_to_row(self.success_memory, event_index, int(communication_successful))
    
        if heard_concept_index is not None:
            # Save data for the confusion matrix
            self.model.confusion_matrix[event_index][heard_concept_index] += 1

        if do_reduction:
            self.model.reduced_turns += 1

            if communication_successful:
                self.model.successful_reduced_turns += 1

        self.model.total_turns += 1

    def reset(self):
        self.speaking = False
        self.hearing = False