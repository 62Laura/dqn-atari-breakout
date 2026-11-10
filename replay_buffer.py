"""Experience replay buffer for DQN (Member 3)."""

import random
from collections import deque

import numpy as np
import torch


class ReplayBuffer:
    """Stores past transitions and hands back random batches of them.

    Training on random old transitions (instead of the last few steps in order)
    breaks the correlation between consecutive frames and makes learning stable.
    """

    def __init__(self, capacity):
        # A deque drops the oldest transition automatically once it is full.
        self.memory = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        # Frames stay as uint8 (0-255) to save memory; 4x84x84 bytes each.
        self.memory.append((
            np.asarray(state, dtype=np.uint8),
            action,
            reward,
            np.asarray(next_state, dtype=np.uint8),
            done,
        ))

    def sample(self, batch_size):
        batch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        # States are returned as raw 0-255 floats; QNetwork divides by 255.
        states = torch.tensor(np.array(states), dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.int64)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.float32)
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.memory)
