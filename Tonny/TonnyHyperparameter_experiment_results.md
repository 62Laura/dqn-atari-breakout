# Hyperparameter experiment results: Gentil Tonny Christian Iradukunda

Environment: ALE/Breakout-v5, 300,000 steps per run. Experiment 1 is the baseline, the others change one thing. "Avg reward" is the mean of the first 20 and the last 20 episodes.

| Experiment | Learning rate | Gamma | Batch size | Epsilon start | Epsilon end | Epsilon decay | Observed behaviour |
|---|---|---|---|---|---|---|---|
| 1 | 1e-4 | 0.99 | 32 | 1.0 | 0.05 | 100000 | Baseline. Avg reward 1.2 to 13.4, best episode 31. Episodes went from about 180 to 580 steps. Still slowly improving at the end. |
| 2 | 5e-4 | 0.99 | 32 | 1.0 | 0.05 | 100000 | Avg reward 1.2 to 16.2, best episode 37. Learned faster than the baseline from the start (6.5 vs 3.1 after 100k steps). My best run. |
| 3 | 2.5e-5 | 0.99 | 32 | 1.0 | 0.05 | 100000 | Avg reward 1.2 to 3.9, best episode 11. Learning rate too small: reward barely moved and games stayed short (about 290 steps). Worst run. |
| 4 | 1e-4 | 0.95 | 32 | 1.0 | 0.05 | 100000 | Avg reward 1.2 to 14.0, best episode 41 (highest single game). A bit better than the baseline, so a shorter horizon did not hurt in Breakout. |
| 5 | 1e-4 | 0.999 | 32 | 1.0 | 0.05 | 100000 | Avg reward 1.2 to 12.4, best episode 27. Slightly below the baseline. Looking that far ahead made no real difference. |
| 6 | 1e-4 | 0.99 | 64 | 1.0 | 0.05 | 100000 | Avg reward 1.2 to 12.5, best episode 31. Early learning was faster than the baseline but it ended about the same. Each step was slower to compute. |
| 7 | 1e-4 | 0.99 | 16 | 1.0 | 0.05 | 100000 | Avg reward 1.2 to 11.3, best episode 23. Slowest start of the normal runs (4.1 after 150k steps vs 8.4). Small batches give noisier updates. |
| 8 | 1e-4 | 0.99 | 32 | 1.0 | 0.05 | 50000 | Avg reward 1.1 to 13.3, best episode 29. Less exploration gave a quicker start but the same final result as the baseline. |
| 9 | 1e-4 | 0.99 | 32 | 1.0 | 0.05 | 250000 | Avg reward 1.3 to 12.7, best episode 29. Slow until about 150k steps because it was still acting mostly random, then caught up. Still rising at the end. |
| 10 | 1e-4 | 0.99 | 32 | 1.0 | 0.01 | 100000 | Avg reward 0.8 to 12.9, best episode 30. Longest final games (about 640 steps). Less randomness at the end helped it keep the ball alive. |
