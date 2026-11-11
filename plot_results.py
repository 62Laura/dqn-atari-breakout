"""Plot the training results of ONE experiment (Member 4).

It reads a *_log.csv file written by train.py and draws 3 graphs:
  1. reward of every episode
  2. length of every episode (number of steps in that episode)
  3. total number of steps done so far

The picture is saved in the plots/ folder.

Run:
  uv run plot_results.py Tonny/exp03_log.csv     plot the experiment you choose
  uv run plot_results.py                         plot the BEST experiment automatically
"""
import argparse
import csv
import glob
import os

import matplotlib
matplotlib.use("Agg")  # draw straight into a picture file, no window needed
import matplotlib.pyplot as plt

# The folder where this script lives (the repo root), so plots/ is always in the same place
ROOT = os.path.dirname(os.path.abspath(__file__))
PLOTS_FOLDER = os.path.join(ROOT, "plots")


def read_log(path):
    """Read a log file. Returns four lists: episodes, total_steps, lengths, rewards."""
    episodes, total_steps, lengths, rewards = [], [], [], []
    with open(path, newline="") as file:
        for row in csv.DictReader(file):
            episodes.append(int(row["episode"]))
            total_steps.append(int(row["total_steps"]))
            lengths.append(int(row["episode_length"]))
            rewards.append(float(row["reward"]))
    return episodes, total_steps, lengths, rewards


def moving_average(values, window):
    """For every point, the average of that point and the (window - 1) points before it.
    This smooths the noisy curve so we can see the trend."""
    averages = []
    for i in range(len(values)):
        start = max(0, i - window + 1)       # at the beginning we have fewer than `window` points
        last_values = values[start:i + 1]
        averages.append(sum(last_values) / len(last_values))
    return averages


def final_score(path, window):
    """How good was this experiment at the end? Average reward of its last `window` episodes."""
    rewards = read_log(path)[3]
    if len(rewards) == 0:
        return None
    last_rewards = rewards[-window:]
    return sum(last_rewards) / len(last_rewards)


def find_best_log(window):
    """Look in every member folder and return the log file with the highest final score."""
    best_path, best_score = None, None
    for path in sorted(glob.glob(os.path.join(ROOT, "*", "*_log.csv"))):
        score = final_score(path, window)
        if score is not None and (best_score is None or score > best_score):
            best_path, best_score = path, score
    return best_path


def make_plot(path, window):
    """Draw the 3 graphs for one log file and save them in plots/."""
    episodes, total_steps, lengths, rewards = read_log(path)
    if len(episodes) == 0:
        raise SystemExit(f"No episodes found in {path}")

    # Name of the experiment, for example "Tonny_exp03"
    member = os.path.basename(os.path.dirname(os.path.abspath(path)))
    experiment = os.path.basename(path).replace("_log.csv", "")
    name = f"{member}_{experiment}"

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(9, 11))
    fig.suptitle(f"DQN Breakout - {name}", fontsize=14)

    # Graph 1: reward per episode (light line = real values, dark line = moving average)
    ax1.plot(episodes, rewards, color="lightblue", label="reward per episode")
    ax1.plot(episodes, moving_average(rewards, window), color="blue",
             label=f"moving average ({window} episodes)")
    ax1.set_title("Reward")
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Reward")
    ax1.legend()

    # Graph 2: episode length (number of steps in each episode)
    ax2.plot(episodes, lengths, color="moccasin", label="length per episode")
    ax2.plot(episodes, moving_average(lengths, window), color="darkorange",
             label=f"moving average ({window} episodes)")
    ax2.set_title("Episode length")
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Steps in the episode")
    ax2.legend()

    # Graph 3: total number of steps done since the start of training
    ax3.plot(episodes, total_steps, color="green")
    ax3.set_title("Total number of steps")
    ax3.set_xlabel("Episode")
    ax3.set_ylabel("Steps since the start")

    fig.tight_layout()
    os.makedirs(PLOTS_FOLDER, exist_ok=True)
    output_path = os.path.join(PLOTS_FOLDER, f"{name}_results.png")
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

    print(f"Experiment: {name}")
    print(f"Episodes: {len(episodes)}, total steps: {total_steps[-1]}")
    print(f"Average reward of the last {window} episodes: {final_score(path, window):.2f}")
    print(f"Plot saved to {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("log_file", nargs="?", default=None,
                        help="path to a *_log.csv file. If you give none, the best experiment is used")
    parser.add_argument("--window", type=int, default=20, help="moving average window (in episodes)")
    args = parser.parse_args()

    if args.log_file is None:
        log_file = find_best_log(args.window)
        if log_file is None:
            raise SystemExit("No *_log.csv files found in the member folders.")
        print(f"Best experiment found automatically: {log_file}")
    else:
        log_file = args.log_file
        if not os.path.isfile(log_file):
            raise SystemExit(f"File not found: {log_file}")

    make_plot(log_file, args.window)


if __name__ == "__main__":
    main()
