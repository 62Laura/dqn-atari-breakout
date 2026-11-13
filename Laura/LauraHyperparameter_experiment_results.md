# Hyperparameter experiments: Laura

| Experiment | Learning rate | Gamma | Batch size | Epsilon start | Epsilon end | Epsilon decay | Observed behaviour |
|---|---|---|---|---|---|---|---|
| 1 | 0.0001 | 0.99 | 32 | 1.0 | 0.05 | 100000 | avg reward 1.2 to 14.8 (first to last 20 episodes). Baseline. Reward rose steadily from about 1 to 15, with a best episode of 33. Episodes got longer (150 to about 600 steps). Still improving at the end |
| 2 | 0.001 | 0.99 | 32 | 1.0 | 0.05 | 100000 | avg reward 1.2 to 12.7 (first to last 20 episodes). Learning rate 0.001. Reward rose from about 1 to 13, ending a bit below the baseline (about 15), and it flattened out between 100,000 and 250,000 steps. |
| 3 | 0.0002 | 0.99 | 32 | 1.0 | 0.05 | 100000 | avg reward 1.2 to 13.8 (first to last 20 episodes). Learning rate 0.0002. Reward rose to about 14, with the highest single episode so far (39), but the average dipped late in training. |
| 4 | 0.00025 | 0.99 | 32 | 1.0 | 0.05 | 100000 | avg reward 1.2 to 15.2 (first to last 20 episodes). Learning rate 0.00025. Reward rose from about 1 to 15, with a steady climb to a peak near 20 at about 230,000 steps, then levelling off. Among the best of my runs |
| 6 | 1e-05 | 0.99 | 32 | 1.0 | 0.05 | 100000 | avg reward 1.2 to 1.8 (first to last 20 episodes). Learning rate 0.00001. Barely learned: reward stayed around 1 to 2 for the whole run, with no clear rise. Too small to learn in 300,000 steps. |
