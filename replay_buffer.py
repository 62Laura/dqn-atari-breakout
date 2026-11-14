import random
from collections import deque

import numpy as np
import torch


class ReplayBuffer:
    def __init__(self, capacity):
        # oldest transitions are dropped when full
        self.memory = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        # keep frames as uint8 to save memory
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

        # QNetwork divides by 255 itself
        states = torch.tensor(np.array(states), dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.int64)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.float32)
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.memory)
