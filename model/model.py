import mesa
import math
import numpy as np

import model.agent
import model.enums
import model.tracker
import model.reporters_agent
import model.reporters_model
import model.model_defaults

from dataclasses import dataclass, asdict
from typing import List, Optional, Callable


class ReductionModel(mesa.Model):
    """A model of the reducing effect"""

    def __init__(self, params: model.model_defaults.Parameters):
        """Initialise the reducing effect model

        Args:
            params (model.model_defaults.Parameters): Parameters object detailing what parameters the simulation should use
        """

        # Load parameters
        self.params = params

        # Load parent class, set random and seed
        super().__init__(rng=self.params.seed)

        # Agents
        agents = model.agent.ReductionAgent.create_agents(
            model=self, n=self.params.num_agents
        )

        # Model data collection
        self.tracker = model.tracker.Tracker(self)

        # Initialise the agent model reporters (no innovator/conservator share in this model)
        model_reporters_agents = model.reporters_agent.get_model_reporters(
            for_all_types=False
        )
        # Initialise the model model reporters
        model_reporters_model = model.reporters_model.get_model_reporters()

        self.datacollector = mesa.DataCollector(
            model_reporters={**model_reporters_agents, **model_reporters_model}
        )
        # self.datacollector.collect(self)

    def step(self):
        """Routine run at every step in the simulation"""

        # Make all agents interact in a random order
        self.agents.shuffle_do("interact_do")

        if self.time % self.params.datacollector_step_size == 0:
            # Collect information about this specific model step
            self.datacollector.collect(self)
            self.tracker.reset()

    def get_random_agent(self, speaker_agent):
        # Choose a random other agent that is not the agent itself
        while True:
            hearer_agent = self.random.choice(self.agents)
            if speaker_agent != hearer_agent:
                break

        return hearer_agent

    def get_random_construction_index(self):
        random_construction_index: int = self.params.nprandom.choice(
            self.params.construction_indices, p=self.params.priors
        )

        return random_construction_index
