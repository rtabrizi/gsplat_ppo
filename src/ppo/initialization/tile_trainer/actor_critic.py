import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor, optim
import torch.distributions as dist
from src.ppo.base_policy import Actor, Critic
from src.ppo.base_env import Env
from src.modules.unet import UNet

class TileActor(Actor):
    def __init__(
        self, 
        train_env: Env = None, 
        test_env: Env = None, 
        in_size = (128, 128, 3),
        h_dim: int = 64,
        num_tiles_h: int = 4, 
        num_tiles_w: int = 4,
        num_bins: int = 10,
        device="cuda",
    ):
        super().__init__()
        self.h, self.w, self.in_channels = in_size
        self.num_bins = num_bins
        self.tiles_h = num_tiles_h
        self.tiles_w = num_tiles_w
        self.device = device
        
        assert self.h % num_tiles_h == 0, "Height must be divisible by num_tiles_h"
        assert self.w % num_tiles_w == 0, "Width must be divisible by num_tiles_w"
        assert num_tiles_h == num_tiles_w, "Only square tiles supported for now"
        
        maxpool_factor = self.h // num_tiles_h
        self.unet = UNet(in_size=in_size, h_dim=h_dim, num_bins=self.num_bins) #output: (128, 128, num_bins)
        self.layers = [
            self.unet,
            nn.MaxPool2d(maxpool_factor), #(128, 128, 3) -> (n_tiles, n_tiles, 3)
            nn.GELU(),
        ]
        self.network = nn.Sequential(*self.layers).to(device)
        self.train_env = train_env
        self.test_env = test_env
        
        self.set_env(train_mode=True)

    def set_env(self, train_mode: bool):
        assert train_mode or self.test_env is not None, "Test env must be set if train mode is False"
        self.env = self.train_env if train_mode else self.test_env
        
    def forward(self, obs: Tensor):
        obs = obs.to(self.device)
        return self.network(obs)

    def actor_distribution(self, obs: Tensor):
        """
        Returns
        -------
        dist: A flattened Categorical distribution over the bins for each tile
        """
        out = self.forward(obs) # (batch_size, n_tiles, n_tiles, num_bins)
        out = out.view(-1, self.num_bins) # (batch_size * n_tiles * n_tiles, num_bins)
        out_probs = dist.Categorical(out)
        # out_probs = out_probs.view(-1, self.num_tiles_h, self.num_tiles_w, self.num_bins)
        return out_probs
    
    def select_action(self, obs: Tensor):
        actor_dist = self.actor_distribution(obs) # (batch_size * n_tiles_h * n_tiles_w, 1)
        batch_weights_flat = actor_dist.sample() # (batch_size * n_tiles_h * n_tiles_w, 1)
        batch_weights = batch_weights.view(-1, self.num_tiles_h, self.num_tiles_w, self.num_bins) # (batch_size, n_tiles, n_tiles, num_bins)
        log_probs = self.log_prob(actor_dist, batch_weights_flat) # (batch_size, n_tiles, n_tiles, 1)

        return batch_weights, log_probs
        
    def log_prob(self, actor_dist, flat_actions):
        batch_log_probs_flat = actor_dist.log_prob(flat_actions) # (batch_size * n_tiles_h * n_tiles_w, 1)
        batch_log_probs = batch_log_probs_flat.view(-1, self.num_tiles_h, self.num_tiles_w, 1) # (batch_size, n_tiles, n_tiles, 1)
        
        return batch_log_probs
    
    def evaluate_actions(
        self, 
        obs: Tensor,
        actions: Tensor # (batch_size, n_tiles, n_tiles, 1)
    ):
        actor_dist = self.actor_distribution(obs) # (batch_size * n_tiles_h * n_tiles_w, 1)
        actions_flat = actions.view(-1, 1) # (batch_size * n_tiles_h * n_tiles_w, 1)

        batch_log_probs = self.log_prob(actor_dist, actions_flat) # (batch_size, n_tiles, n_tiles, 1)
        
        flat_entropy = actor_dist.entropy() # (batch_size * n_tiles_h * n_tiles_w, 1)
        batch_entropy = flat_entropy.view(-1, self.num_tiles_h, self.num_tiles_w, 1) # (batch_size, n_tiles, n_tiles, 1)
    
        return batch_log_probs, batch_entropy
    

class TileCritic(Critic):
    def __init__(self, train_env: Env, test_env: Env, unet: UNet, input_dim: int = 1024, h_dim: int = 64):
        super().__init__()
        # should map obs (image) to value
        self.actor_encoder = unet.encoder # (batch_size, 8, 8, 4*h_dim)
        self.linear1 = nn.Linear(4 * unet.h_dim, h_dim) # (batch_size, 8, 8, 4*h_dim) --> (batch_size, h_dim)
        self.linear2 = nn.Linear(h_dim, 1) # (batch_size, h_dim) --> (batch_size, 1)
        self.network = nn.Sequential(
            self.actor_encoder,
            nn.GELU(),
            self.linear1,
            self.GELU(),
            self.linear2
        )
        self.train_env = train_env
        self.test_env = test_env
        
        self.set_env(train_mode=True)
    
    def set_env(self, train_mode: bool):
        assert train_mode or self.test_env is not None, "Test env must be set if train mode is False"
        self.env = self.train_env if train_mode else self.test_env

    def forward(self, obs: Tensor):        
        obs = obs.to(torch.int)
        
        # Ensure obs is properly reshaped for the network
        batch_size = obs.shape[0] if len(obs.shape) > 3 else 1
        # obs = obs.view(batch_size, -1)  # Flatten input to (batch_size, features)

        return self.network(obs)