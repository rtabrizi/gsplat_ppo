import torch

class RolloutBuffer:
    def __init__(self, buffer_size, observation_space, action_space, device="cuda"):
        self.buffer_size = buffer_size
        self.device = device

        # Pre-allocate storage for the buffer on the correct device
        self.obs = torch.zeros((buffer_size, *observation_space), device=device)
        self.actions = torch.zeros((buffer_size, *action_space), device=device)
        self.log_probs = torch.zeros(buffer_size, device=device)
        self.rewards = torch.zeros(buffer_size, device=device)
        self.returns = torch.zeros(buffer_size, device=device) # rtg
        self.values = torch.zeros(buffer_size, device=device)
        self.dones = torch.zeros(buffer_size, dtype=torch.bool, device=device)
        self.ptr = 0  # Pointer to keep track of the current position

    def add(self, state, action, reward, log_prob, value, done):
        """
        Add a new experience to the buffer.
        Make sure that inputs are already on the correct device (cuda or cpu).
        """
        if self.ptr >= self.buffer_size:
            raise IndexError("Buffer overflow: trying to add more samples than buffer can hold")

        self.obs[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.log_probs[self.ptr] = log_prob
        self.values[self.ptr] = value
        self.dones[self.ptr] = done
        self.ptr += 1

    def get(self, batch_size=None, shuffle=False):
        """
        Retrieve the data from the buffer in batches for optimization.
        If batch_size is None, return the entire buffer.
        Otherwise, return the data in batches of the specified size.

        If shuffle is True, shuffle the indices before extracting batches.
        """
        if shuffle:
            indices = torch.randperm(self.ptr, device=self.device)  # Shuffle the indices on the correct device
            return self._get_batches(indices, batch_size)
        else:
            return self._get_batches(torch.arange(self.ptr, device=self.device), batch_size)

    def _get_batches(self, indices, batch_size):
        """
        Retrieve data in batches based on the provided indices.
        """
        if batch_size is None: batch_size = self.buffer_size
        num_batches = len(indices) // batch_size
        for i in range(num_batches):
            batch_indices = indices[i * batch_size: (i + 1) * batch_size]
            yield {
                'obs': self.obs[batch_indices],
                'actions': self.actions[batch_indices],
                'log_probs': self.log_probs[batch_indices],
                'rewards': self.rewards[batch_indices],
                'returns': self.returns[batch_indices],
                'values': self.values[batch_indices],
                'dones': self.dones[batch_indices]
            }
        
        # Handle remaining items if the buffer size isn't an exact multiple of batch_size
        if len(indices) % batch_size != 0:
            batch_indices = indices[num_batches * batch_size:]
            yield {
                'obs': self.obs[batch_indices],
                'actions': self.actions[batch_indices],
                'log_probs': self.log_probs[batch_indices],
                'rewards': self.rewards[batch_indices],
                'returns': self.returns[batch_indices],
                'values': self.values[batch_indices],
                'dones': self.dones[batch_indices]
            }

    def reset(self):
        """
        Reset the buffer for the next iteration.
        """
        self.ptr = 0
        self.obs.zero_()
        self.actions.zero_()
        self.rewards.zero_()
        self.returns.zero_()
        self.log_probs.zero_()
        self.values.zero_()
        self.dones.zero_()

    def is_full(self):
        """
        Check if the buffer has reached its size limit.
        """
        return self.ptr >= self.buffer_size
