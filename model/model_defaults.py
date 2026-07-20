import model.vectors
import model.vocabulary
import model.reduction
import model.sampling
import model.enums
import numpy as np

from dataclasses import dataclass, asdict, field
from typing import List, Optional, Dict, Type, Any, Tuple

# Mapping of parameter names to their enum classes
PARAMETER_ENUM_MAPPING: Dict[str, Type] = {}


@dataclass
class Parameters:
    # ----
    # Non-default
    # ----
    reduction: model.reduction.Base | model.reduction.SoftThresholding
    vector_bounds: List[int]

    # ----
    # Model housekeeping
    # ----

    num_agents: int = 50
    seed: int | None = None
    # After how many steps do we collect data?
    datacollector_step_size: int = 1

    # ----
    # Vocabulary
    # ----
    num_constructions: int = 100
    num_dims: int = 10
    batch_size: int = 20
    priors_enabled: bool = True

    value_floor: int = 5
    initial_sigma: int = 5
    learning_rate: float = 0.2

    sampling_type: (
        model.sampling.ZipfianSampling
        | model.sampling.ExponentialSampling
        | model.sampling.LinearSampling
    ) = field(default_factory=lambda: model.sampling.ZipfianSampling())

    # Do we only update one form (the winner) or all forms (weighted)
    to_update: int = model.enums.ToUpdate.WINNER_ONLY
    do_update: bool = True

    # ----
    # Reduction
    # ----
    value_floor: int = 5
    reduction_prob: float = 0.5

    # ----
    # Communication
    # ----
    reentrance_entropy_floor: float = 0.0
    feedback_type: int = model.enums.FeedbackTypes.NO_FEEDBACK

    def __post_init__(self):
        # Initialise random number generator
        self.nprandom = np.random.default_rng(self.seed)

        # Populate the construction indices
        self.construction_indices = list(range(self.num_constructions))

        # Get the priors for the chosen sampling type
        true_ranks, self.priors = self.sampling_type.get_priors(self.nprandom)

        # Initialise the vocabulary with randomly generated vectors
        # min_val = minimum value for vector
        # max_val = maximum value for vector, minus sigma
        # TODO add seed!
        vectors = model.vectors.generate(
            self.num_constructions,
            num_dims=self.num_dims,
            min_val=self.vector_bounds[0],
            max_val=self.vector_bounds[1] - self.initial_sigma,
            random_generator=self.nprandom,
        )
        display_names = [f"#{idx}" for idx in self.construction_indices]

        self.vocabulary = model.vocabulary.Vocabulary(
            num_words=self.num_constructions,
            num_dims=self.num_dims,
            initial_sigma=self.initial_sigma,
            means=vectors,
            priors=self.priors.tolist(),
            priors_enabled=self.priors_enabled,
            display_names=display_names,
            # value floor = minimum legal value for vector
            # value max = maximum saved value
            value_floor=self.value_floor,
            valid_range=[1, self.vector_bounds[1]],
        )
