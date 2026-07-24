import model.enums
import model.sampling
import model.reduction

# TODO
params = {
    "regular": {
        "num_agents": 10,
        "reduction": [
            model.reduction.SoftThresholding(strength=1),
            model.reduction.SoftThresholding(strength=5),
        ],
        "reduction_prob": [0.1, 0.5, 1],
        "value_floor": [1],
        "vector_bounds": [[70, 100]],
        "learning_rate": [0.05, 0.2, 0.5, 0.8],
        "initial_sigma": [5, 10, 15],
        "reentrance": [False, True],
    },
    "quick": {
        "num_agents": 10,
        "reduction": [
            model.reduction.SoftThresholding(strength=1),
        ],
        "reduction_prob": [0, 0.1, 0.5, 1],
        "value_floor": [1],
        "vector_bounds": [[70, 100]],
        "learning_rate": [0.2],
        "initial_sigma": [5],
        "reentrance": [False, True],
        "learning_rate": [0.01, 0.2, 0.5],
    },
    "quick2": {
        "num_agents": 10,
        "reduction": [
            model.reduction.SoftThresholding(strength=1),
        ],
        "reduction_prob": [0.2, 0.5, 0.8],
        "value_floor": [15],
        "vector_bounds": [[70, 100]],
        "learning_rate": [0.2],
        "initial_sigma": [5],
        "reentrance_entropy_floor": [0, 0.025, 0.05, 0.1],
    },
    "quick3": {
        "num_agents": 10,
        "batch_size": 10,
        "reduction": [
            model.reduction.CompressReduce(),
        ],
        "reduction_prob": [0, 0.5],
        "value_floor": [15],
        "vector_bounds": [[70, 100]],
        "learning_rate": [0.2],
        "initial_sigma": [5],
        "reentrance_entropy_floor": [0, 0.05],
        "priors_enabled": [True, False],
        "sampling_type": [
            model.sampling.ZipfianSampling(zipf_param=1),
            model.sampling.ZipfianSampling(zipf_param=0.6),
            model.sampling.ZipfianSampling(zipf_param=0.01),
        ],
        "feedback_type": [
            model.enums.FeedbackTypes.NO_FEEDBACK,
            model.enums.FeedbackTypes.FEEDBACK,
        ],
    },
    "quick4": {
        "num_agents": 10,
        "reduction": [
            model.reduction.CompressReduce(),
        ],
        "reduction_prob": [0.5],
        "value_floor": [15],
        "vector_bounds": [[70, 100]],
        "learning_rate": [0.2],
        "initial_sigma": [5],
        "discriminatory_entropy_floor": [0, 0.01, 0.025, 0.05],
        "priors_enabled": [True, False],
        "sampling_type": [
            model.sampling.ZipfianSampling(zipf_param=1),
        ],
        "feedback_type": [
            model.enums.FeedbackTypes.NO_FEEDBACK,
        ],
    },
}
