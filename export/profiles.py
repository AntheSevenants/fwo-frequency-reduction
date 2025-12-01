import model.types.reduction
import model.types.feedback

base_model = {
    "memory_size": 1000,
    "self_check": False,
    "num_dimensions": 10,
    "reduction_method": model.types.reduction.ReductionMethod.SOFT_THRESHOLDING,
    "feedback_type": model.types.feedback.FeedbackTypes.NO_FEEDBACK
}

PROFILES = {
    "base-model": base_model,
    "reentrance-model": {**base_model, "self_check": True},
    "cone-model": {"feedback_type": model.types.feedback.FeedbackTypes.NO_FEEDBACK,
                   "toroidal": True},
    "no-zipfian-model": {**base_model,
                         "exponential_sampling_lambda": 0.0,
                         "self_check": True},
    "no-zipfian-model-feedback": {**base_model,
                                  "feedback_type": model.types.feedback.FeedbackTypes.FEEDBACK,
                                  "exponential_sampling_lambda": 0.0,
                                  "self_check": True}
    # "no-shared-code-model": {**base_model, "self_check": True, "jumble_vocabulary": True},
    # "no-zipfian-model": {**base_model, "self_check": True, "zipfian_sampling": False},
    # "single-exemplar-model": {**base_model, "self_check": True, "memory_size": 100},
    # "no-reduction-model": {**base_model, "self_check": True, "disable_reduction": True}
}
ALL_PROFILE_NAMES = list(PROFILES.keys())