import copy

import model.vocabulary

from dataclasses import dataclass


@dataclass
class Attributes:
    vocabulary: model.vocabulary.Vocabulary

    def __post_init__(self):
        # Make a deepcopy of the vocabulary
        self.vocabulary = copy.deepcopy(self.vocabulary)
