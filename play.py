"""Load a trained DQN model and watch it play ALE/Breakout-v5.

Run from the project root:
    uv run play.py
    uv run play.py --model Ulrich/exp01_model.pth --episodes 3
"""

import argparse
import os

import ale_py
import gymnasium as gym
import numpy as np
import torch
from gymnasium.wrappers import AtariPreprocessing, FrameStackObservation

from q_network import QNetwork


gym.register_envs(ale_py)
ENV_NAME = "ALE/Breakout-v5"


def choose_device(requested):
    """Choose an available device, preferring CUDA and then Apple MPS."""
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but it is not available.")
    if device.type == "mps" and not (
        hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    ):
        raise RuntimeError("MPS was requested, but it is not available.")
    return device


def make_env():
    """Use the same preprocessing as training and enable live rendering."""
    env = gym.make(ENV_NAME, frameskip=1, render_mode="human")
    env = AtariPreprocessing(
        env, frame_skip=4, screen_size=84, grayscale_obs=True, noop_max=30
    )
    env = FrameStackObservation(env, 4)
    return env


def parse_args():
    parser = argparse.ArgumentParser(description="Watch a trained DQN play Breakout.")
    parser.add_argument(
        "--model",
        default="dqn_model.pth",
        help="Path to a saved QNetwork state_dict (default: dqn_model.pth)",
    )
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.episodes < 1:
        raise SystemExit("--episodes must be at least 1.")
    if not os.path.isfile(args.model):
        raise SystemExit(
            f"Model not found: {args.model}\n"
            "Train an experiment first, then pass its model path with --model."
        )

    
    device = choose_device(args.device)
    env = make_env()

    try:
        model = QNetwork(env.action_space.n).to(device)
        state_dict = torch.load(args.model, map_location=device, weights_only=True)
        model.load_state_dict(state_dict)
        model.eval()

        print(f"Loaded {args.model} on {device}. Close the game window to stop.")

        
        for episode in range(1, args.episodes + 1):
            state, _ = env.reset()

            # Remember how many lives the agent has.
            lives = env.unwrapped.ale.lives()
            total_reward = 0.0
            terminated = truncated = False
            need_fire = True

            while not (terminated or truncated):

                # Launch the ball at the beginning
                # and after losing a life.
                if need_fire:
                    action = 1
                    need_fire = False
                else:
                    state_tensor = torch.as_tensor(
                        np.asarray(state),
                        dtype=torch.uint8,
                        device=device
                    ).unsqueeze(0)

                    with torch.no_grad():
                        action = int(
                            model(state_tensor).argmax(dim=1).item()
                        )

                state, reward, terminated, truncated, _ = env.step(action)
                total_reward += float(reward)

                # Check whether the agent lost a life.
                current_lives = env.unwrapped.ale.lives()

                if current_lives < lives:
                    lives = current_lives
                    need_fire = True

            print(
                f"Episode {episode}/{args.episodes}: "
                f"reward = {total_reward:.1f}"
            )


    finally:
        env.close()



if __name__ == "__main__":
    main()
