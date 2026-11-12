# Hyperparameter experiments: Laura

| Experiment | Learning rate | Gamma | Batch size | Epsilon start | Epsilon end | Epsilon decay | Observed behaviour |
|---|---|---|---|---|---|---|---|
| 1 | 0.0001 | 0.99 | 32 | 1.0 | 0.05 | 100000 | avg reward 1.2 to 14.8 (first to last 20 episodes). Baseline. Reward rose steadily from about 1 to 15, with a best episode of 33. Episodes got longer (150 to about 600 steps). Still improving at the end |
| 2 | 0.001 | 0.99 | 32 | 1.0 | 0.05 | 100000 | avg reward 1.2 to 12.7 (first to last 20 episodes). Learning rate 0.001. Reward rose from about 1 to 13, ending a bit below the baseline (about 15), and it flattened out between 100,000 and 250,000 steps. |
| 3 | 0.0002 | 0.99 | 32 | 1.0 | 0.05 | 100000 | avg reward 1.2 to 13.8 (first to last 20 episodes). Learning rate 0.0002. Reward rose to about 14, with the highest single episode so far (39), but the average dipped late in training. |
