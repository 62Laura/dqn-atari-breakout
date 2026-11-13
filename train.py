"""Train a DQN agent on Atari Breakout.

Run:  uv run train.py --lr 1e-4 --gamma 0.99 --batch-size 32 \
          --eps-start 1.0 --eps-end 0.05 --eps-decay 100000 --name Laura_exp01
"""
import argparse
import csv
import os
import random
from collections import deque

import ale_py
import gymnasium as gym
import numpy as np
import torch
from q_network import QNetwork
from replay_buffer import ReplayBuffer
from train_step import train_step
from gymnasium.wrappers import AtariPreprocessing, FrameStackObservation

gym.register_envs(ale_py)


ENV_NAME = "ALE/Breakout-v5"
BUFFER_SIZE = 50_000          # transitions kept in memory
LEARNING_STARTS = 5_000       # fill the buffer with this many steps before learning
TRAIN_EVERY = 4               # one learning step every 4 environment steps
TARGET_UPDATE_EVERY = 1_000   # copy online weights into the target network
TOTAL_STEPS = 300_000         # length of one experiment
SEED = 42



def make_env():
    """Breakout with the standard DQN preprocessing: output is 4 x 84 x 84."""
    env = gym.make(ENV_NAME, frameskip=1)  # frame skipping is done by the wrapper below
    env = AtariPreprocessing(env, frame_skip=4, screen_size=84, grayscale_obs=True, noop_max=30)
    env = FrameStackObservation(env, 4)    # stack the last 4 frames so the agent sees motion
    return env


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--eps-start", type=float, default=1.0)
    parser.add_argument("--eps-end", type=float, default=0.05)
    parser.add_argument("--eps-decay", type=int, default=100_000, help="steps to go from eps-start to eps-end")
    parser.add_argument("--name", type=str, default="Laura_exp01", help="MemberName_expNN")
    parser.add_argument("--total-steps", type=int, default=TOTAL_STEPS)
    return parser.parse_args()


def main():
    args = parse_args()

    # Same seed everywhere so runs differ only by their hyperparameters
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    # Outputs go in the member's folder: Laura/exp01_model.pth and Laura/exp01_log.csv
    member, exp = args.name.split("_", 1)
    os.makedirs(member, exist_ok=True)
    model_path = os.path.join(member, f"{exp}_model.pth")
    log_path = os.path.join(member, f"{exp}_log.csv")

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"Using device: {device}")
    env = make_env()
    n_actions = env.action_space.n

    # Online network (learns) and target network (a slowly updated copy, gives stable targets)
    online = QNetwork(n_actions).to(device)
    target = QNetwork(n_actions).to(device)
    target.load_state_dict(online.state_dict())
    optimizer = torch.optim.Adam(online.parameters(), lr=args.lr)
    buffer = ReplayBuffer(BUFFER_SIZE)

    log_file = open(log_path, "w", newline="")
    log = csv.writer(log_file)
    log.writerow(["episode", "total_steps", "episode_length", "reward", "epsilon"])

    state, _ = env.reset(seed=SEED)
    episode, episode_reward, episode_length = 1, 0.0, 0
    recent_rewards = deque(maxlen=10)

    for step in range(1, args.total_steps + 1):
        # 1. choose an action (epsilon-greedy) and play it
        epsilon = epsilon_by_step(step, args.eps_start, args.eps_end, args.eps_decay)
        action = select_action(online, state, epsilon, n_actions)
        next_state, reward, terminated, truncated, _ = env.step(action)

        # 2. store the experience. Reward is clipped to -1/0/+1 (standard DQN trick,
        #    keeps the loss on the same scale). "terminated" tells the Bellman target
        #    that there is no future after the last step.
        buffer.add(state, action, np.sign(reward), next_state, terminated)
        state = next_state
        episode_reward += reward
        episode_length += 1

        # 3. learn from a random batch of old experiences
        if len(buffer) >= LEARNING_STARTS and step % TRAIN_EVERY == 0:
            train_step(online, target, buffer, optimizer, args.batch_size, args.gamma)

        # 4. every N steps copy the online weights into the target network
        if step % TARGET_UPDATE_EVERY == 0:
            target.load_state_dict(online.state_dict())

        # 5. end of episode: log it and start a new one
        if terminated or truncated:
            log.writerow([episode, step, episode_length, episode_reward, round(epsilon, 4)])
            log_file.flush()
            recent_rewards.append(episode_reward)
            if episode % 10 == 0:
                print(f"episode {episode:4d} | step {step:7d} | epsilon {epsilon:.3f} | "
                      f"mean reward (last 10) {np.mean(recent_rewards):.1f}")
            state, _ = env.reset()
            episode += 1
            episode_reward, episode_length = 0.0, 0

    log_file.close()
    torch.save(online.state_dict(), model_path)  # play.py loads this state_dict
    print(f"Saved model to {model_path} and log to {log_path}")



def epsilon_by_step(step, start, end, decay):
    """Epsilon at this step: goes from `start` to `end` over `decay` steps, then stays at `end`."""
    # Decay is finished (or there is nothing to decay): stay at the final value
    if step >= decay:
        return end

    # Still decaying. progress goes from 0.0 (first step) up to 1.0 (end of decay).
    progress = step / decay

    # Straight line from start down to end
    return start + (end - start) * progress


def select_action(online, state, epsilon, n_actions):
    """With probability epsilon return a random action, otherwise the action with the highest Q-value
    from the online network. `state` is a numpy array (4, 84, 84). Returns an int."""
    # Explore: roll a random number between 0 and 1. If it is smaller than epsilon,
    # ignore the network and try a random action.
    if random.random() < epsilon:
        return random.randrange(n_actions)

    # Exploit: ask the network which action it thinks is best.
    device = next(online.parameters()).device           # CPU or GPU, wherever the network lives
    state = torch.as_tensor(state).unsqueeze(0).to(device)  # (4, 84, 84) -> (1, 4, 84, 84): a batch of one
    with torch.no_grad():                               # we are only choosing, not learning
        q_values = online(state)                        # shape (1, n_actions): one score per action
    return int(q_values.argmax(dim=1).item())           # index of the highest score


if __name__ == "__main__":
    main()
