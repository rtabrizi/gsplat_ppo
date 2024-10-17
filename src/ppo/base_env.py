import numpy as np
from abc import ABC, abstractmethod

class Env(ABC):
    def __init__(self) -> None:
        self.step_count = 0
        self.num_steps = 100
        super().__init__()
        
    @abstractmethod
    def reset(self):
        """Reset the environment and return the initial observation."""
        raise NotImplementedError

    @abstractmethod
    def step(self, action):
        """
        Perform an action in the environment.
        Returns: next_state, reward, done (boolean)
        """
        raise NotImplementedError

    @abstractmethod
    def get_observation(self):
        """Return the current state/observation of the environment."""
        raise NotImplementedError