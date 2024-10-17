import numpy as np
import torch
import torch.nn as nn
from torch import Tensor, optim
import torch.distributions as dist
from abc import ABC, abstractmethod
from typing import Any

class Critic(nn.Module, ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, action: Any, state=None):
        """
        Output the critic's predicted values.

        Parameters:
        - action (Any): The action(s) to be taken.
        - state (Any): The current state 

        Returns:
        - value_estimate (float): The value estimate of the state-action pair.
        """
        pass

class Actor(nn.Module, ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, state: Any):
        """
        Forward through the Actor network
        
        Parameters:
        - state (Any): The current state
        
        Returns:
        - intermediate (Any): The result of the forward pass,
                parameterizing the actor's action distribution
        """
        pass

    @abstractmethod
    def select_action(self, state: Any):
        """
        Runs the actor network to return the action and log_prob of the action.
        
        Parameters:
        - state (Any): The current state
        
        Returns:
        - action (Any): The action to be taken.
        - log_prob (float): The log probability of the action.
        """
        pass

    @abstractmethod
    def log_probs(self):
        """
        Returns log probabilities of the action space.

        Parameters:
        - None

        Returns:
        - log_probs: List(float): the log probabilities of all actions.

        """
        pass

class Policy(nn.Module):
    def __init__(self,
                actor: Actor,
                critic: Critic,
                actor_lr=3e-4,
                critic_lr=1e-3
                ):
        super().__init__()
        self.actor = actor
        self.critic = critic

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)

    def select_action(self, obs):
        """
        Given an observation, use the actor to select an action and return
        the action, log probability, and the value from the critic.

        Parameters:
        - obs (Any): The observation returned by the environment.

        Returns:
        - action (Any): 
        """
        # Sample action from actor
        action, log_prob = self.actor.select_action(obs)  

        # Get value estimate from critic
        value = self.critic(obs)
        return action, log_prob, value

    def evaluate_actions(self, obs, actions):
        """
        Given observations and actions, return log probabilities of actions
        and value estimates. This will be used by PPO to compute the loss.
        """
        # Evaluate actions from actor
        log_probs = self.actor.log_probs(obs, actions) 

        # Get value estimates from critic
        values = self.critic(obs)  
        return values, log_probs

    def optimizer_step(self, actor_loss, critic_loss):
        """
        Perform optimization steps for both the actor and critic.
        """
        # Actor update
        self.actor_optimizer.zero_grad()
        actor_loss.backward(retain_graph=True)  # Retain graph?
        self.actor_optimizer.step()

        # Critic update
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
