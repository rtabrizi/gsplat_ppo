import numpy as np
import torch
import json
import os
from abc import ABC, abstractmethod
from src.ppo.tile_trainer import TileTrainer
from src.ppo.base_env import Env

import numpy as np
import torch
import json
import warnings
import os
from collections import defaultdict
from examples.image_fitting import SimpleTrainer
from src.ppo.base_env import Env

from src.utils import image_path_to_tensor, dino_preprocess


class Tile2DEnv(Env):
    """
    A test environment where the agent's actions are used to simulate 
    the process of training a neural network.
    """
    def __init__(
        self, 
        lr: float,
        dataset_path: str,
        num_points: int, 
        num_iterations: int,
        # TODO: remove observation_shape?
        observation_shape: tuple, 
        action_shape: tuple,
        num_trials: int = 1,
        device='cuda',
        num_tiles_h: int = 2,
        num_tiles_w: int = 2,
    ):
        # Environment state would be the 2d image
        # action: tile weights
        # 
        self.max_steps = 1
        self.num_points = num_points
        self.num_trials = num_trials
        self.num_iterations = num_iterations
        
        self.lr = lr

        self.device = device
        # self.observation_shape = img.shape
        self.action_shape = action_shape

        self.observation_shape = (1,) # Just img index

        self.num_imgs = 0

        self.imgs = []
        self.img_names = []
        
        for img_name in os.listdir(dataset_path):
            self.num_imgs += 1
            full_path = os.path.join(dataset_path, img_name)
            img = image_path_to_tensor(full_path)
            self.imgs.append(img)
            self.img_names.append(img_name)
            
        print("=" * 100)    
        print(f'num images: {self.num_images}\n num_trials: {self.num_trials}\n num_points: {self.num_points}\n num_iterations: {self.num_iterations}, lr: {self.lr}')
        print("=" * 100)    
        self.imgs = torch.stack(self.imgs)

        current_dir = os.path.dirname(os.path.abspath(__file__))
        dataset_name = dataset_path.split('/')[-1]
        
        self.sample_new_img()

    def sample_new_img(self):
        self.current_img_idx = torch.tensor(np.random.randint(0, self.num_images), device=self.device)

    def reset(self):
        """
        Reset the environment to an initial state.
        """
        # Randomly select new img idx
        self.sample_new_img()
        return self.get_observation()

    def step(self, action: int):
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
        # unsqueeze action if 0-d tensor
        if len(action.shape) == 0:
            action = action.unsqueeze(-1)
            
        # Convert actions to int to use as indices
        actions_int = action.long()
        batch_psnr = self.psnr[self.current_img_idx][actions_int]

        # mse_diff = batch_losses[:, 0] - batch_losses[:, -1]
        # Using PSNR as reward (TODO: normalize for critic?)
        # print(f"batch_losses: {reward_psnr} for action: {action}")

        # torch bool
        done = torch.as_tensor(True, device=self.device, dtype=torch.bool)
        
        # print(f"for action: {action}, reward: {reward}, using lr: {self.lrs[int(action.item())]}")
        return self.get_observation(), batch_psnr, done

    def get_observation(self):
        """
        Return the current state (img_idx)
        """
        return self.current_img_idx
    
    def get_encoded_images(self, img_idx: torch.tensor):
        imgs = self.encoded_images[img_idx]
        if len(imgs.shape) == 4:
            imgs = imgs.squeeze(1)
        if len(imgs.shape) == 3:
            imgs = imgs.squeeze(1)
        return imgs
    
    def get_mean_reward(self, img_idx: torch.tensor):
        return self.psnr_stats['mean'][img_idx]