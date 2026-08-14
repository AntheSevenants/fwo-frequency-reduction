def to_dict(cls):
    original_dict = {k: v for k, v in cls.__dict__.items() if not k.startswith("__")}
    reversed_dict = {v: k for k, v in original_dict.items()}

    return reversed_dict


class FeedbackTypes:
    FEEDBACK = 0
    NO_FEEDBACK = 1


class ReductionModes:
    ALWAYS = 0
    L1_SUCCESS_DEPENDENT = 1
    SUCCESS_DEPENDENT_MICRO = 2
    SUCCESS_DEPENDENT_MACRO = 3


class ReductionMethod:
    DIMENSION_SCRAP = 0
    SOFT_THRESHOLDING = 1
    SOFT_THRESHOLDING_DIM = 4
    GAUSSIAN_MASK = 2
    ANGLE = 3
    TAPER = 5
    NON_LINEAR = 6
    ALPHA_ONLY = 9
    BYE_MAX = 7
    SOFT_THRESHOLDING_2 = 8


class Repair:
    MEAN = 0
    NEGATIVE_REDUCTION = 2
    PICK_ANOTHER = 3
    NEGATIVE_REDUCTION_ANGLE = 4
    NO_REPAIR = 1


class SamplingTypes:
    ZIPFIAN = 0
    LINEAR = 1
    FLAT = 2
    EXPONENTIAL = 3


class VectorTypes:
    ORIGINAL = 0
    RADICAL = 1
    DIRK_P2 = 2


class ToUpdate:
    WINNER_ONLY = 0
    DISTRIBUTED = 1


class ReductionOutcomes:
    FAILURE = 0
    SUCCESS = 1


class ReentranceUsage:
    NOT_USED = 0
    USED = 1
