from model.helpers import load_vectors, generate_word_vectors, generate_zipfian_sample, load_info, compute_average_vocabulary, compute_mean_communicative_success_per_token, generate_quarter_circle_vectors
from model.model import ReductionModel
from model.types.neighbourhood import NeighbourhoodTypes
from model.types.production import ProductionModels
from model.types.reduction import ReductionModes, ReductionMethod
from model.types.feedback import FeedbackTypes
from model.types.sampling import SamplingTypes
from model.types.repair import Repair
from model.types.who_saves import WhoSaves
from model.types.vector import VectorTypes

from batchrunner import batch_run

from datetime import datetime

import pandas as pd
import numpy as np
import os
import argparse
import random
import math

parser = argparse.ArgumentParser(
    description='batch_run - volt go brr (and crash sometimes)')
parser.add_argument('profile', type=str,
                    help='regular | cone')
parser.add_argument('iterations', type=int, default=1, help='number of iterations, default = 1')
args = parser.parse_args()

NUM_STEPS = 50000

params_template = {
    "num_tokens": [ 100 ],
    "n_large": 100,
    "num_agents": 25,
    "reduction_prior": 0.5,
    "value_ceil": 100,
    "memory_size": 1000,
    "reduction_strength": [ 5 ],
    "neighbourhood_size": 5,
    "num_dimensions": [ 10 ],
    "initial_token_count": 5,
    "prefill_memory": True,
    "neighbourhood_step_size": 0,
    "production_model": ProductionModels.SINGLE_EXEMPLAR,
    "neighbourhood_type": NeighbourhoodTypes.LEVENSHTEIN,
    "reduction_mode": ReductionModes.ALWAYS,
    "feedback_type": [ FeedbackTypes.FEEDBACK, FeedbackTypes.NO_FEEDBACK ],
    "reduction_method": [ ReductionMethod.SOFT_THRESHOLDING ],
    "sampling_type": [ SamplingTypes.ZIPFIAN ],
    "repair": Repair.NO_REPAIR,
    "confidence_threshold": 0,
    "value_floor": 5,
    "who_saves": [ WhoSaves.HEARER ],
    "max_turns": 1,
    "vectors_type": [ VectorTypes.DIRK_P2 ],
    "fixed_memory": True
}

if args.profile == "regular":
    params = { **params_template,
        "self_check": [ False, True ],
        "datacollector_step_size": 1000,
        "zipf_param": 1.0,
    }
elif args.profile == "exponential":
    params = { **params_template,
        "sampling_type": [ SamplingTypes.EXPONENTIAL ],
        "exponential_sampling_lambda": [ math.log(x, 10) for x in np.arange(1, 3 + 0.05, 0.05) ],
        "self_check": [ False, True ],
        "datacollector_step_size": 1000
    }
elif args.profile == "cone":
    NUM_STEPS = 1000

    params = {
        **params_template,
        "num_tokens": 10,
        "num_agents": 5,
        "num_distribution_tokens": 100,
        "value_ceil": 100,
        "reduction_strength": 5,
        "memory_size": 100,
        "initial_token_count": 1,
        "num_dimensions": 2,
        "reduction_method": ReductionMethod.ANGLE,
        "self_check": False,
        "datacollector_step_size": 50,
        "toroidal": True,
        "light_serialisation": False
    }

if __name__ == '__main__':
    now = datetime.now() # current date and time
    date_time = now.strftime("%Y-%m-%d_%H-%M-%S")

    run_folder = f"models/{args.profile}-{date_time}/"
    os.makedirs(run_folder, exist_ok=True)

    # Save frequency, token and percentile information
    # tokens, frequencies, percentiles, ranks = load_info(f"vectors/theoretical-percentile-info-{max(params['num_tokens'])}.tsv", theoretical=True)

    # tokens_df = pd.DataFrame({
    #     "tokens": tokens,
    #     "frequencies": frequencies,
    #     "percentiles": percentiles,
    #     "ranks": ranks
    # })
    # tokens_df.to_csv(f"{run_folder}token_infos.csv", index=False)

    results = batch_run(
        ReductionModel,
        run_folder,
        parameters=params,
        iterations=args.iterations,
        max_steps=NUM_STEPS,
        number_processes=None,
        data_collection_period=100,
        display_progress=True,
    )

    csv_filename = f"{run_folder}run_infos.csv"
    br_df = pd.DataFrame(results)
    br_df.to_csv(csv_filename, index=False)

#     print(results)