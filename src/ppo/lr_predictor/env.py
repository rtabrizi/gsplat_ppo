import numpy as np
import torch
import json
import os
from abc import ABC, abstractmethod
from examples.image_fitting import SimpleTrainer
from src.ppo.base_env import Env

class LREnv(Env):
    """
    A test environment where the agent's actions are used to simulate 
    the process of training a neural network.
    """
    def __init__(self, img, num_points: int, iterations: int, lrs: list[float]):
        # Environment state would be the 2d image
        # action: tile weights
        # 
        self.max_steps = 1
        self.img = img
        self.num_points = num_points
        self.iterations = iterations
        self.lrs = [10**-i for i in range(10)]

        # compute losses for each LR
        losses_json_path = f"lr_losses_{self.iterations}_iterations{num_points}_points.json"
        if os.path.exists(losses_json_path):
            with open(losses_json_path, 'r') as f:
                self.lr_losses = json.load(f)
        else: 
            lr_losses = []
            for lr in self.lrs:
                print("*" * 50)
                print(f'currently training with lr={lr}')
                trainer = SimpleTrainer(gt_image=self.img, num_points=num_points)
                losses = trainer.train(
                    iterations=self.iterations,
                    lr=lr,
                    save_imgs=False,
                    model_type='3dgs',
                )
                lr_losses[str(lr)] = losses

            # Write the lr_losses to a JSON file
            output_filename = losses_json_path
            with open(output_filename, 'w') as f:
                json.dump(lr_losses, f)
                
    def reset(self):
        """
        Reset the environment to an initial state.
        """
        
        return self.img

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
        
        trainer = SimpleTrainer(
            gt_image=self.img,
            num_points=self.num_points,
        )
        losses, _ = trainer.train(
            iterations=self.iterations,
            lr=self.lr,
            save_imgs=True,
            model_type='3dgs',
        )
        reward = losses[0] - losses[-1]
        done = True
        
        return self.get_observation(), reward, done

    def get_observation(self):
        """
        Return the current state
        """
        return self.img