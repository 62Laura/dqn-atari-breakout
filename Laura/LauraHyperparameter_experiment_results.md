# Hyperparameter experiments: Laura

| Experiment | Learning rate | Gamma | Batch size | Epsilon start | Epsilon end | Epsilon decay | Observed behaviour |
|---|---|---|---|---|---|---|---|
| 1 | 0.0001 | 0.99 | 32 | 1.0 | 0.05 | 100000 | avg reward 1.2 to 14.8 (first to last 20 episodes). Baseline. Reward rose steadily from about 1 to 15, with a best episode of 33. Episodes got longer (150 to about 600 steps). Still improving at the end |
