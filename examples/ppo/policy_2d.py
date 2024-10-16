import numpy as np
import torch
import torch.nn as nn
from torch import Tensor, optim
import torch.distributions as dist

# choosing lr
# - based on this image, choose a lr
# primitive initialization

class Critic(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = torch.Sequential(
            nn.Linear(1, 1),
            nn.Sigmoid(),
        )

    def forward(self, action: int, state=None):
        return self.network(action).squeeze(-1)
    

class Actor(nn.Module):
    def __init__(self):
        super().__init__()
        self.logits = nn.Parameter(torch.zeros(10))
        self.lrs = [10**-i for i in range(10)]

    def forward(self):
        return nn.Softmax(self.logits, dim=-1)

    def select_action(self):
        log_probs = self.forward()
        actor_dist = dist.Categorical(logits=log_probs)
        lr_idx = actor_dist.sample()
        return self.lrs[lr_idx], actor_dist.log_probs[lr_idx]

    def get_best_lr(self):
        return self.lrs[self.logits.argmax().item()]
    
    def get_best_lr_prob(self):
        return self.forward().max().item()

class Policy(nn.Module):
    def __init__(self, actor_lr=3e-4, critic_lr=1e-3):
        super(Policy, self).__init__()
        self.actor = Actor()
        self.critic = Critic()

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)

    def select_action(self, obs):
        """
        Given an observation, use the actor to select an action and return
        the action, log probability, and the value from the critic.
        """
        action, log_prob = self.actor.select_action(obs)  # Sample action from actor
        value = self.critic(obs)  # Get value estimate from critic
        return action, log_prob, value

    def evaluate_actions(self, obs, actions):
        """
        Given observations and actions, return log probabilities of actions
        and value estimates. This will be used by PPO to compute the loss.
        """
        log_probs = self.actor.evaluate_actions(obs, actions)  # Evaluate actions from actor
        values = self.critic(obs)  # Get value estimates from critic
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