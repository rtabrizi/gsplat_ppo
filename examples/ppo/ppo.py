import math
import os
import json
import matplotlib.pyplot as plt
import time
from collections import defaultdict
from tqdm import tqdm
from pathlib import Path
from typing import Literal, Optional
from scipy.signal import butter, filtfilt
import time

import numpy as np
import torch.nn as nn
import torch
import tyro
from PIL import Image
from torch import Tensor, optim

from gsplat import rasterization, rasterization_2dgs
from policy_2d import Policy2D
from rollout_buffer import RolloutBuffer
from image_fitting import SimpleTrainer
import random



def seed_everything(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

seed_everything(42)


def image_path_to_tensor(image_path: Path):
    import torchvision.transforms as transforms

    img = Image.open(image_path)
    transform = transforms.ToTensor()
    img_tensor = transform(img).permute(1, 2, 0)[..., :3]
    return img_tensor 

device = "cuda" if torch.cuda.is_available() else "cpu"
lr_predictor = Policy2D().to(device)

class PPO:
    def __init__(self, policy, env, 
                 clip_epsilon=0.2, gamma=0.99, gae_lambda=0.95, normalize_advantages=False, entropy_coeff=0.0,
                 n_epochs=4, batch_size=64, 
                 **kwargs):
        """
        Initialize the PPO algorithm with hyperparameters and policy.
        """
        self.policy = policy
        self.env = env
        
        self.clip_epsilon = clip_epsilon                    # Threshold to clip the ratio during actor updates
        self.gamma = gamma                                  # Discount factor for future rewards
        self.gae_lambda = gae_lambda                        # GAE parameter (unused)
        self.normalize_advantages = normalize_advantages    # Normalize advantages before updating the policy
        self.n_epochs = n_epochs                            # Number of epochs to update the policy on a collected rollout
        self.entropy_coeff = entropy_coeff                  # Coefficient for entropy regularization (unused)
        self.batch_size = batch_size

        # Optional hyperparameters
        self.max_timesteps = kwargs.get('max_timesteps', 1e6)
        self.log_interval = kwargs.get('log_interval', 1000)
        self.device = kwargs.get('device', 'cuda')

        # Rollout buffer to store experiences
        self.rollout_buffer = RolloutBuffer(
            buffer_size=kwargs.get('buffer_size', 2048),
            observation_space=env.observation_space.shape,
            action_space=env.action_space.shape,
            device=self.device
        )

        # Logging
        self.logger = {"t_so_far": 0, "i_so_far": 0, "actor_losses": []}

    def collect_rollout(self):
        """
        Collect a batch of rollouts from the environment.
        Store the experiences in the rollout buffer.
        """
        self.rollout_buffer.reset()  # Clear buffer before collecting new data
        obs = self.env.reset()
        
        # Collect experience for each step until buffer is full
        while not self.rollout_buffer.is_full():
            with torch.no_grad():
                action, log_prob, value = self.policy.select_action(obs)
            next_obs, reward, done = self.env.step(action)
            
            # Reshape if discrete?
            # actions = actions.reshape(-1, 1)

            # Add the experience to the buffer
            self.rollout_buffer.add(
                obs, 
                torch.tensor(action, device=self.device), 
                torch.tensor(reward, device=self.device), 
                log_prob, value, done
            )
            
            obs = next_obs
            if done:
                obs = self.env.reset()

    def compute_advantages(self, rewards, values, dones):
        """
        Compute advantages using Generalized Advantage Estimation (GAE).
        """
        advantages = []
        gae = 0
        next_value = 0

        # Reverse iterate over rewards and dones
        for t in reversed(range(len(rewards))):
            if dones[t]:
                next_value = 0
                gae = 0
            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.lambda_ * (1 - dones[t]) * gae
            advantages.insert(0, gae)
            next_value = values[t]
        
        return torch.tensor(advantages, device=self.device)

    def compute_loss(self, batch_data):
        """
        Compute the PPO loss for both actor and critic networks.
        """
        states = batch_data['states']
        actions = batch_data['actions']
        old_log_probs = batch_data['log_probs']
        returns = batch_data['returns']

        # Compute advantages
        if self.normalize_advantage and len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-10)  # Normalize advantages

        # Get the new log_probs and value estimates from the policy
        log_probs_new, values_new = self.policy.evaluate_actions(states, actions)

        # ratio between old and new policy
        ratios = torch.exp(log_probs_new - old_log_probs)

        # Actor loss (clipped surrogate objective)
        surr1 = ratios * advantages
        surr2 = torch.clamp(ratios, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
        actor_loss = -torch.min(surr1, surr2).mean()
        
        # TODO: maybe add entropy bonus to the actor loss

        # Critic loss (MSE between values and rewards-to-go)
        critic_loss = nn.MSELoss()(values_new, returns)

        return actor_loss, critic_loss

    def update(self):
        """
        Perform PPO update for the policy using the collected rollout data.
        This method updates the policy over several epochs.
        """
        for _ in range(self.n_epochs):
            # Get batches from the rollout buffer
            for batch_data in self.rollout_buffer.get(batch_size=self.batch_size, shuffle=True):
                actor_loss, critic_loss = self.compute_loss(batch_data)

                # Update the policy (both actor and critic)
                self.policy.optimizer_step(actor_loss, critic_loss)

                # Log the actor loss for monitoring
                # self.logger['actor_losses'].append(actor_loss.detach().cpu().numpy())

    def train(self, total_timesteps):
        """
        Main training loop that runs for the specified number of timesteps.
        """
        t_so_far = 0  # Timesteps tracked so far
        i_so_far = 0  # Number of iterations tracked so far

        while t_so_far < total_timesteps:
            # Collect rollouts and populate the buffer
            self.collect_rollout()

            # Update total timesteps and iterations
            t_so_far += self.rollout_buffer.buffer_size
            i_so_far += 1

            # Log timesteps and iterations
            self.logger['t_so_far'] = t_so_far
            self.logger['i_so_far'] = i_so_far

            # Update the policy using the rollouts
            self.update()

            # Logging summary
            if t_so_far % self.log_interval == 0:
                self._log_summary()


log_iter = 50
height: int = 256
width: int = 256
num_points: int = 100000
save_imgs: bool = True
training_iterations: int = 1000
lr: float = 0.01
model_type: Literal["3dgs", "2dgs"] = "3dgs"

im_path = 'images/adam.jpg'
if not im_path:    
    gt_image = torch.ones((height, width, 3)) * 1.0
    # make top left and bottom right red, blue
    gt_image[: height // 2, : width // 2, :] = torch.tensor([1.0, 0.0, 0.0])
    gt_image[height // 2 :, width // 2 :, :] = torch.tensor([0.0, 0.0, 1.0])
else:
    gt_image = image_path_to_tensor(im_path)

buffer_size = 10
num_epochs = 2
num_updates = int(1e4)
lr_losses = {}

if True:
# if not os.path.exists(f"lr_losses_{training_iterations}.json"):
    print("generating trajectories")
    for lr in lr_predictor.lrs:
        print("*" * 50)
        print(f'currently training with lr={lr}')
        trainer = SimpleTrainer(gt_image=gt_image, num_points=num_points)
        losses = trainer.train(
            iterations=training_iterations,
            lr=lr,
            save_imgs=save_imgs,
            model_type=model_type,
        )
        lr_losses[lr] = losses

    # Write the lr_losses to a JSON file
    output_filename = f"lr_losses_{training_iterations}.json"
    with open(output_filename, 'w') as f:
        json.dump(lr_losses, f)
else:
    with open(f"lr_losses_{training_iterations}.json", 'r') as f:
        lr_losses = json.load(f)

lr_probs = defaultdict(list)
reinforce_losses = []
rewards_history = []

print("training lr predictor")
pbar = tqdm(range(num_updates), desc="best LR prob")

rollout_times = []
update_times = []
log_probs = torch.empty(buffer_size, device=device)
rewards = torch.empty(buffer_size, device=device)
for update in pbar:

    time_start = time.time()
    # Collect rollout
    log_probs = []
    rewards = []

    # Pre-allocate tensors
    log_probs = torch.empty(buffer_size, device=device)
    rewards = torch.empty(buffer_size, device=device)

    for i in range(buffer_size):
        lr, log_prob = lr_predictor.select_action()
        losses = lr_losses[str(lr)]

        final_loss, initial_loss = losses[-1], losses[0]
        loss_reduction = final_loss - initial_loss
        reward = -loss_reduction
        
        log_probs[i] = log_prob
        rewards[i] = reward

    time_rollouts = time.time()
    rollout_times.append(time_rollouts - time_start)

    # Calculate policy gradient loss
    policy_loss = -(log_probs * rewards).mean()
    reinforce_losses.append(policy_loss.item())
    rewards_history.append(rewards.mean().item())

    pbar.set_description(f"best LR {lr_predictor.get_best_lr()} with prob: {lr_predictor.get_best_lr_prob():.4f}")

    # Update the network
    optimizer.zero_grad()
    policy_loss.backward()
    optimizer.step()
    verbose = False

    time_end = time.time()
    update_times.append(time_end - time_start)

    if update % 50:
        print(f"avg rollout time: {np.mean(rollout_times)}, avg update time: {np.mean(update_times)}")
        
    if update % log_iter == 0 and verbose:
        # Optionally, print some statistics
        print("=" * 100)
        print(f"started at loss = {initial_loss}, finished at loss = {final_loss} (difference of {-1 * loss_reduction})for lr idx {lr_idx} (lr {lr})")
        print(f"Update {update}, Avg Reward: {rewards.mean().item()}")
        print(f'best LR: {lr_predictor.get_best_lr()}')
        print(f'softmax: {lr_predictor.forward().exp()}')
        # print(f'best index: {lr_predictor.logits.argmax().item()}')
        print(f'best lr: {lr_predictor.get_best_lr()}')
    for prob, lr in zip(lr_predictor.forward().exp(), lr_predictor.lrs):
        lr_probs[str(lr)].append(prob.detach().cpu().numpy())



updates = list(range(num_updates))
optimal_lr = lr_predictor.get_best_lr()

optimal_lr_probs = lr_probs[str(optimal_lr)]
plt.plot(updates, optimal_lr_probs, label=f'Optimal LR: {optimal_lr}')
plt.xlabel('Updates')
plt.ylabel('Probability')
plt.title('Updates vs Probability of Optimal Learning Rate')
plt.legend()
# plt.show()
plt.savefig(f'lr_probs_{training_iterations}.png')

# Plot REINFORCE losses


def lowpass_filter(data, cutoff=0.1, fs=1.0, order=5):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y

window_size = 100
box_filter = np.ones(window_size) / window_size

filtered_reinforce_losses = np.convolve(reinforce_losses, box_filter, mode='same')

plt.figure()
plt.plot(range(num_updates), reinforce_losses, label='REINFORCE Loss')
plt.xlabel('Updates')
plt.ylabel('Loss')
plt.title('REINFORCE Loss over Updates')
plt.legend()
plt.savefig(f'reinforce_losses_{training_iterations}.png')

plt.plot(range(num_updates), filtered_reinforce_losses, label='filtered REINFORCE Loss')
plt.xlabel('Updates')
plt.ylabel('Loss')
plt.title('REINFORCE Loss over Updates')
plt.legend()
plt.savefig(f'filtered_reinforce_losses_{training_iterations}.png')

# Plot rewards
plt.figure()
plt.plot(range(num_updates), rewards_history, label='Average Reward')
plt.xlabel('Updates')
plt.ylabel('Reward')
plt.title('Average Reward over Updates')
plt.legend()
plt.savefig(f'rewards_{training_iterations}.png')

print("executing training run with optimal lr")
# Launch a training job with the optimal LR for 2000 points
# num_points = 100000
save_img = True
training_iterations = 1000
trainer = SimpleTrainer(gt_image=gt_image, num_points=num_points)
# Assuming we have a function `train` that takes learning rate and number of points as arguments
save_path = f'results/{optimal_lr}_lr_{training_iterations}_iterations_{num_points}_points.png'
trainer.train(iterations=training_iterations, lr=optimal_lr, save_path=save_path)

# Save the training results if save_img is True
if save_img:
    plt.figure()
    plt.plot(range(num_updates), rewards_history[:num_updates], label='Average Reward')
    plt.xlabel('Points')
    plt.ylabel('Reward')
    plt.title('Average Reward over Points')
    plt.legend()
    plt.savefig(f'rewards_{training_iterations}_2000_points.png')

