import model.enums
import model.sampling
import model.reduction

from batch.profile import BatchProfile

params = {
    "article": BatchProfile(
        parent_params={
            "num_agents": 10,
            "reduction": model.reduction.CompressReduce(),
            "reduction_prob": [0.5],
            "value_floor": [15],
            "vector_bounds": [[70, 100]],
            "learning_rate": [0.2],
            "initial_sigma": [5],
            "priors_enabled": True,
            "sampling_type": model.sampling.ZipfianSampling(zipf_param=1),
            "feedback_type": model.enums.FeedbackTypes.NO_FEEDBACK,
            "batch_size": 10,
        },
        child_params={
            "priors-no-priors": {
                "priors_enabled": [True, False],
            },
            "re-entrance": {
                "reentrance_entropy_floor": [0, 0.05],
            },
            "discriminatory-force": {
                "discriminatory_entropy_floor": [0, 0.01, 0.025, 0.05],
            },
            "no-zipfian": {
                "reentrance_entropy_floor": 0.05,
                "sampling_type": [
                    model.sampling.ZipfianSampling(zipf_param=1),
                    model.sampling.ZipfianSampling(zipf_param=0.6),
                    model.sampling.ZipfianSampling(zipf_param=0.01),
                ],
            },
            "no-zipfian-feedback": {
                "reentrance_entropy_floor": 0.05,
                "sampling_type": [
                    model.sampling.ZipfianSampling(zipf_param=1),
                    model.sampling.ZipfianSampling(zipf_param=0.6),
                    model.sampling.ZipfianSampling(zipf_param=0.01),
                ],
                "feedback_type": model.enums.FeedbackTypes.FEEDBACK,
            },
        },
    )
}
