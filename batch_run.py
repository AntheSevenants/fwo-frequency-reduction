from model.helpers import load_vectors, generate_word_vectors, generate_zipfian_sample, load_info, compute_average_vocabulary, compute_mean_communicative_success_per_token, generate_quarter_circle_vectors
from model.model import ReductionModel
from model.types.neighbourhood import NeighbourhoodTypes
from model.types.production import ProductionModels
from model.types.reduction import ReductionModes, ReductionMethod
from model.types.feedback import FeedbackTypes
from model.types.repair import Repair
from model.types.who_saves import WhoSaves

from batchrunner import batch_run

from datetime import datetime

import pandas as pd
import os
import argparse
import random

parser = argparse.ArgumentParser(
    description='batch_run - volt go brr (and crash sometimes)')
parser.add_argument('profile', type=str,
                    help='regular | cone')
parser.add_argument('iterations', type=int, default=1, help='number of iterations, default = 1')
args = parser.parse_args()

params_template = {
    "num_agents": 25,
    "reduction_prior": 0.5,
    "initial_token_count": 1,
    "prefill_memory": True,
    "neighbourhood_step_size": 0,
    "reduction_strength": 15,
    "production_model": ProductionModels.SINGLE_EXEMPLAR,
    "neighbourhood_type": NeighbourhoodTypes.SPATIAL,
    "reduction_mode": ReductionModes.ALWAYS,
    "feedback_type": FeedbackTypes.NO_FEEDBACK,
    "repair": Repair.NO_REPAIR,
    "confidence_threshold": 0,
    "who_saves": [ WhoSaves.HEARER ],
    "max_turns": 1
}

if args.profile == "regular":
    NUM_STEPS = 50000

    params = { **params_template,
        "num_tokens": 100,
        "memory_size": [ 100, 1000 ],
        "num_dimensions": 100,
        "neighbourhood_size": [ 150 ],
        "reduction_method": ReductionMethod.SOFT_THRESHOLDING,
        "jumble_vocabulary": [ False, True ],
        "zipfian_sampling": [ True, False ],
        "disable_reduction": [ False, True ],
        "self_check": [ False, True ],
        "datacollector_step_size": 1000
    }
elif args.profile == "cone":
    NUM_STEPS = 1000

    params = {
        **params_template,
        "num_tokens": 10,
        "value_ceil": 250,
        "memory_size": 100,
        "num_dimensions": 2,
        "neighbourhood_size": 25,
        "reduction_method": ReductionMethod.ANGLE,
        "jumble_vocabulary": False,
        "zipfian_sampling": True,
        "disable_reduction": False,
        "self_check": False,
        "datacollector_step_size": 50,
        "toroidal": True,
        "seed": random.randint(0, 99999999),
        "light_serialisation": False
    }

if __name__ == '__main__':
    now = datetime.now() # current date and time
    date_time = now.strftime("%Y-%m-%d_%H-%M-%S")

    run_folder = f"models/{args.profile}-{date_time}/"
    os.makedirs(run_folder, exist_ok=True)

    # Save frequency, token and percentile information
    tokens, frequencies, percentiles, ranks = load_info(f"vectors/theoretical-percentile-info-{params['num_tokens']}.tsv", theoretical=True)

    tokens_df = pd.DataFrame({
        "tokens": tokens,
        "frequencies": frequencies,
        "percentiles": percentiles,
        "ranks": ranks
    })
    tokens_df.to_csv(f"{run_folder}token_infos.csv", index=False)

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