from model.helpers import load_vectors, generate_word_vectors, generate_zipfian_sample, load_info, compute_average_vocabulary, compute_mean_communicative_success_per_token, generate_quarter_circle_vectors
from model.model import ReductionModel
from model.types.neighbourhood import NeighbourhoodTypes
from model.types.production import ProductionModels
from model.types.reduction import ReductionModes, ReductionMethod
from model.types.feedback import FeedbackTypes
from model.types.repair import Repair

from batchrunner import batch_run

from datetime import datetime

import pandas as pd
import os


params = { "num_agents": 25,
           "num_tokens": 100,
           "memory_size": 1000,
           "num_dimensions": 100,
           "reduction_prior": 0.5,
           "initial_token_count": 1,
           "prefill_memory": True,
           "neighbourhood_type": NeighbourhoodTypes.SPATIAL,
           "neighbourhood_size": 25,
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
    now = datetime.now() # current date and time
    date_time = now.strftime("%Y-%m-%d_%H-%M-%S")

    run_folder = f"models/{date_time}/"
    os.makedirs(run_folder, exist_ok=True)

    results = batch_run(
        ReductionModel,
        date_time,
        parameters=params,
        iterations=1,
        max_steps=10000,
        number_processes=None,
        data_collection_period=100,
        display_progress=True,
    )

    csv_filename = f"{run_folder}run_infos.csv"
    br_df = pd.DataFrame(results)
    br_df.to_csv(csv_filename, index=False)

#     print(results)