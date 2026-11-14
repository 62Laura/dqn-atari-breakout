# uv run test_train_step.py

import numpy as np
import torch
import torch.nn as nn

from replay_buffer import ReplayBuffer
from train_step import train_step

N_ACTIONS = 4
BATCH_SIZE = 32


def tiny_network():
    return nn.Sequential(nn.Flatten(), nn.Linear(4 * 84 * 84, N_ACTIONS))


def fill_buffer():
    # buffer holds exactly one batch, so every sample is the same batch
    buffer = ReplayBuffer(capacity=BATCH_SIZE)
    for _ in range(BATCH_SIZE):
        state = np.random.randint(0, 256, (4, 84, 84), dtype=np.uint8)
        next_state = np.random.randint(0, 256, (4, 84, 84), dtype=np.uint8)
        action = np.random.randint(N_ACTIONS)
        reward = float(np.random.choice([0.0, 1.0]))
        done = bool(np.random.rand() < 0.1)
        buffer.add(state, action, reward, next_state, done)
    return buffer


def main():
    torch.manual_seed(0)
    np.random.seed(0)
    buffer = fill_buffer()
    assert len(buffer) == BATCH_SIZE

    online = tiny_network()
    target = tiny_network()
    target.load_state_dict(online.state_dict())
    optimizer = torch.optim.Adam(online.parameters(), lr=1e-4)

    losses = [train_step(online, target, buffer, optimizer, BATCH_SIZE, gamma=0.99)
              for _ in range(100)]
    print(f"first loss {losses[0]:.4f}, last loss {losses[-1]:.4f}")
    assert losses[-1] < losses[0], "loss did not go down"
    print("loss goes down")


if __name__ == "__main__":
    main()
