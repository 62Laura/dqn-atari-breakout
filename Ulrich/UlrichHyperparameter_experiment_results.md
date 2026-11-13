---
noteId: "7326d2b0df8311f1adc5337a5f7de4e6"
tags: []

---

# Ulrich — DQN Breakout Hyperparameter Experiment Results

## Experimental setup

- **Environment:** `ALE/Breakout-v5`
- **Training budget:** 300,000 environment steps per run
- **Device:** Apple MPS
- **Seed:** 42, as configured in the shared training script
- **Network:** CNN `QNetwork` with four stacked 84 × 84 grayscale frames
- **Metrics:** Calculated from completed episodes recorded in each experiment's CSV log.

The final logged step can be slightly below 300,000 because the training script writes a row only when an episode ends. The training loop still completes the requested 300,000 environment steps before saving the model.

## Results summary

| Exp. | Learning rate | Gamma | Batch size | Epsilon start | Epsilon end | Epsilon decay | Episodes | Overall avg. reward | First 20 avg. | Last 20 avg. | Best reward | Avg. episode length | Last logged step |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 01 | 0.0001 | 0.99 | 32 | 1.0 | 0.05 | 100,000 | 854 | 6.10 | 1.20 | 13.40 | 31 | 350.82 | 299,602 |
| 02 | 0.00005 | 0.99 | 32 | 1.0 | 0.05 | 100,000 | 968 | 4.42 | 1.20 | 9.00 | 26 | 309.34 | 299,440 |
| 03 | 0.00025 | 0.99 | 32 | 1.0 | 0.05 | 100,000 | 719 | 8.50 | 1.20 | 17.35 | 36 | 416.52 | 299,476 |
| 04 | 0.0001 | 0.95 | 32 | 1.0 | 0.05 | 100,000 | 824 | 6.23 | 1.20 | 14.00 | 41 | 364.00 | 299,932 |
| 05 | 0.0001 | 0.995 | 32 | 1.0 | 0.05 | 100,000 | 855 | 6.09 | 1.20 | 13.75 | 45 | 350.59 | 299,754 |
| 06 | 0.0001 | 0.99 | 16 | 1.0 | 0.05 | 100,000 | 948 | 4.67 | 1.20 | 12.45 | 27 | 316.02 | 299,584 |
| 07 | 0.0001 | 0.99 | 64 | 1.0 | 0.05 | 100,000 | 767 | 7.49 | 1.20 | 12.50 | 31 | 391.03 | 299,921 |
| 08 | 0.0001 | 0.99 | 32 | 0.8 | 0.05 | 100,000 | 793 | 7.17 | 1.65 | 16.90 | 41 | 377.90 | 299,674 |
| 09 | 0.0001 | 0.99 | 32 | 1.0 | 0.01 | 100,000 | 763 | 7.35 | 0.80 | 12.90 | 30 | 393.04 | 299,887 |
| 10 | 0.0001 | 0.99 | 32 | 1.0 | 0.05 | 200,000 | 918 | 5.46 | 1.35 | 14.30 | 26 | 326.37 | 299,612 |

## Observed behaviour

### Experiment 01 — Baseline

The baseline achieved an overall average reward of 6.10 and a final-20-episode average of 13.40. The best episode reward was 31. Reward increased from 1.20 in the first 20 episodes, providing a reference for the later one-parameter comparisons.

### Experiment 02 — Lower learning rate (0.00005)

The overall average reward was 4.42 and the final-20 average was 9.00, both below the baseline. Under this training budget, the lower learning rate did not improve the measured performance and may have slowed learning.

### Experiment 03 — Higher learning rate (0.00025)

This was the strongest run by both overall average reward (8.50) and final-20 average reward (17.35). It achieved a best episode reward of 36. The higher learning rate was effective in this run, although repeated seeds would be required to determine whether the improvement is robust.

### Experiment 04 — Lower discount factor (gamma 0.95)

The overall average reward was 6.23 and the final-20 average was 14.00, slightly above the baseline. The best episode reward was 41. This suggests a modest improvement in this run, but the difference is small.

### Experiment 05 — Higher discount factor (gamma 0.995)

The overall average reward was 6.09 and the final-20 average was 13.75, close to the baseline. The best episode reward was 45, the highest individual episode reward among the ten runs. Its average performance, however, did not clearly improve on the baseline.

### Experiment 06 — Smaller batch size (16)

The overall average reward was 4.67 and the final-20 average was 12.45, below the baseline's final-20 average of 13.40. The smaller batch size did not improve the measured performance in this run.

### Experiment 07 — Larger batch size (64)

The overall average reward was 7.49, above the baseline's 6.10, but the final-20 average was 12.50, slightly below the baseline's 13.40. This is a mixed result: better average reward across the full run, but not in the final 20 episodes.

### Experiment 08 — Lower initial epsilon (0.8)

The overall average reward was 7.17 and the final-20 average was 16.90. The first-20 average was 1.65. This was the second-best run by final-20 average reward, close to Experiment 03. Lower initial exploration appears promising in this run, but more seeds are needed to confirm the effect.

### Experiment 09 — Lower final epsilon (0.01)

The overall average reward was 7.35 and the final-20 average was 12.90. Despite the higher overall average, the final-20 average was below the baseline. Reducing the minimum exploration rate further did not improve the final-window result in this run.

### Experiment 10 — Slower epsilon decay (200,000 steps)

The overall average reward was 5.46 and the final-20 average was 14.30. This exceeded the baseline's final-20 average of 13.40, but its overall average was lower. A slower decay may have supported later learning, but did not provide a clear overall improvement in this run.

## Overall comparison and conclusion

Using final-20-episode average reward as the primary comparison metric, Experiment 03 was the strongest run (17.35), followed by Experiment 08 (16.90), Experiment 10 (14.30), Experiment 04 (14.00), Experiment 05 (13.75), and Experiment 01 (13.40).

Experiment 03 also had the highest overall average reward (8.50). Its configuration was learning rate 0.00025, gamma 0.99, batch size 32, epsilon start 1.0, epsilon end 0.05, and epsilon decay 100,000 steps. It is therefore the best-performing configuration among these ten single runs according to the recorded metrics. Experiment 05 achieved the highest individual episode reward (45), but not the highest average performance.

Each configuration was tested once using the project's configured seed. DQN training is stochastic; these results identify the best observed configuration in this experiment set, not proof that it is universally optimal. Repeating configurations with multiple independent seeds would provide stronger evidence.

## Saved artifacts

Each experiment generated a model and a CSV log in the `Ulrich/` directory:

- `exp01_model.pth` and `exp01_log.csv`
- `exp02_model.pth` and `exp02_log.csv`
- `exp03_model.pth` and `exp03_log.csv`
- `exp04_model.pth` and `exp04_log.csv`
- `exp05_model.pth` and `exp05_log.csv`
- `exp06_model.pth` and `exp06_log.csv`
- `exp07_model.pth` and `exp07_log.csv`
- `exp08_model.pth` and `exp08_log.csv`
- `exp09_model.pth` and `exp09_log.csv`
- `exp10_model.pth` and `exp10_log.csv`

The selected best checkpoint from this set is `exp03_model.pth`.
