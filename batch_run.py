from model.helpers import load_vectors, generate_word_vectors, generate_zipfian_sample, load_info, compute_average_vocabulary, compute_mean_communicative_success_per_token, generate_quarter_circle_vectors
from model.model import ReductionModel
from model.types.neighbourhood import NeighbourhoodTypes
from model.types.production import ProductionModels
from model.types.reduction import ReductionModes, ReductionMethod
from model.types.feedback import FeedbackTypes
from model.types.repair import Repair

from mesa.batchrunner import batch_run

NUM_AGENTS = 25
NUM_DIMENSIONS = 100
NUM_TOKENS = 100
VALUE_CEIL = 100
# vectors, tokens, frequencies, percentiles = load_vectors(f"materials/vectors-{NUM_DIMENSIONS}.txt")
tokens, frequencies, percentiles, ranks = load_info(f"vectors/theoretical-percentile-info-{NUM_TOKENS}.tsv", theoretical=True)

# Overwrite vectors with my own
vectors = generate_word_vectors(vocabulary_size=len(tokens), dimensions=NUM_DIMENSIONS)

params = { "num_agents": NUM_AGENTS,
           "vectors": vectors,
           "tokens": tokens,
           "frequencies": frequencies,
           "percentiles": percentiles,
           "ranks": ranks,
           "reduction_prior": 0.5,
           "memory_size": 1000,
           "value_ceil": 100,
           "initial_token_count": 1,
           "prefill_memory": True,
           "neighbourhood_type": NeighbourhoodTypes.SPATIAL,
           "neigbourhood_size": 25,
           "neighbourhood_step_size": 25,
           "reduction_strength": 15,
           "production_model": ProductionModels.SINGLE_EXEMPLAR,
           "reduction_mode": ReductionModes.ALWAYS,
           "reduction_method": ReductionMethod.SOFT_THRESHOLDING,
           "feedback_type": FeedbackTypes.NO_FEEDBACK,
           "repair": [ Repair.NO_REPAIR, Repair.NEGATIVE_REDUCTION ],
           "confidence_threshold": [ 0.6, 0.8, 0.95 ],
           "self_check": [ True, False ],
           "max_turns": [ 1, 2, 3 ],
           "datacollector_step_size": 100 }

if __name__ == '__main__':
    results = batch_run(
        ReductionModel,
        parameters=params,
        iterations=5,
        max_steps=100,
        number_processes=8,
        data_collection_period=-1,
        display_progress=True,
    )

    print(results)