import numpy as np
import torch

class EnvInterface:
    def reset(self):
        """Reset the environment and return the initial observation."""
        raise NotImplementedError

    def step(self, action):
        """
        Perform an action in the environment.
        Returns: next_state, reward, done (boolean)
        """
        raise NotImplementedError

    def get_observation(self):
        """Return the current state/observation of the environment."""
        raise NotImplementedError


class TiledInit2DEnv(EnvInterface):
    """
    A test environment where the agent's actions are used to simulate 
    the process of training a neural network.
    """

    def __init__(self, input_dim=3, output_dim=3):
        # Environment state could be the performance of the neural network
        self.state = torch.rand(output_dim)  # Dummy initial state
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.step_count = 0
        self.max_steps = 1  # Simulate a maximum of 100 steps per episode
        self.obs_dim = output_dim
        self.action_dim = input_dim
    
    def reset(self):
        """
        Reset the environment to an initial state.
        """
        self.state = torch.rand(self.output_dim)  # Reset to random initial state
        self.step_count = 0
        return self.get_observation()

    def step(self, action):
        """
        Simulate applying the action (inputs to a neural network) and return 
        the next state, reward, and whether the episode is done.
        
        Args:
            action: The input action that simulates neural network input.
        
        Returns:
            next_state: The new state after applying the action.
            reward: The reward, which could be based on the network performance.
            done: A boolean indicating if the episode has ended.
        """
        self.step_count += 1

        # Simulate the change in state based on the action
        self.state = self.state + action - 0.5 * torch.rand(self.output_dim)

        # Reward is based on how "close" the state gets to a target value (e.g., zero)
        reward = -torch.norm(self.state)  # Negative reward for being far from zero

        # End the episode after a fixed number of steps
        done = self.step_count >= self.max_steps

        return self.get_observation(), reward, done

    def get_observation(self):
        """
        Return the current state (performance metric).
        """
        return self.state

    def render(self):
        """
        Render the current state (just print it for simplicity).
        """
        print(f"Current state: {self.state}")