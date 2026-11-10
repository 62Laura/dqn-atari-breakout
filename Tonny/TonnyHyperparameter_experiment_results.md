# Hyperparameter experiment results: Gentil Tonny Christian Iradukunda

Environment: ALE/Breakout-v5. Every run uses the same number of training steps. Experiment 1 is the baseline; each other run changes one hyperparameter so its effect is clear.

| Experiment | Learning rate | Gamma | Batch size | Epsilon start | Epsilon end | Epsilon decay | Observed behaviour |
|---|---|---|---|---|---|---|---|
| 1 | 1e-4 | 0.99 | 32 | 1.0 | 0.05 | 100000 | |
| 2 | 5e-4 | 0.99 | 32 | 1.0 | 0.05 | 100000 | |
| 3 | 2.5e-5 | 0.99 | 32 | 1.0 | 0.05 | 100000 | |
| 4 | 1e-4 | 0.95 | 32 | 1.0 | 0.05 | 100000 | |
| 5 | 1e-4 | 0.999 | 32 | 1.0 | 0.05 | 100000 | |
| 6 | 1e-4 | 0.99 | 64 | 1.0 | 0.05 | 100000 | |
| 7 | 1e-4 | 0.99 | 16 | 1.0 | 0.05 | 100000 | |
| 8 | 1e-4 | 0.99 | 32 | 1.0 | 0.05 | 50000 | |
| 9 | 1e-4 | 0.99 | 32 | 1.0 | 0.05 | 250000 | |
| 10 | 1e-4 | 0.99 | 32 | 1.0 | 0.01 | 100000 | |
