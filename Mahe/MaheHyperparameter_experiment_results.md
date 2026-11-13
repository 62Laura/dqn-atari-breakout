
# Hyperparameter experiment results: MAHE Digne

Environment: ALE/Breakout-v5. Every run uses the same number of training steps (300,000).

## Results

| Experiment | Learning rate | Gamma | Batch size | Epsilon start | Epsilon end | Epsilon decay | Observed behaviour |
|---|---|---|---|---|---|---|---|
| 1 | 1e-4 | 0.99 | 32 | 1.0 | 0.05 | 100000 | Baseline. Average reward rose from 1.2 in the first 20 episodes to 12.4 in the last 20. Epsilon reached 0.05 at step 100,000 and the reward kept improving after that. Learning was steady but noisy from episode to episode. |
| 2 | 5e-4 | 0.99 | 32 | 1.0 | 0.05 | 100000 | Higher learning rate (5 times the baseline). Average reward rose from 1.2 to 13.2 in the last 20 episodes, a little above the baseline (12.4). It needed only 755 episodes for 300,000 steps, so games were longer on average. The gain is small and may be normal run-to-run variation. |
| 3 | 2.5e-5 | 0.99 | 32 | 1.0 | 0.05 | 100000 | Much lower learning rate. Average reward went from 1.2 only to 2.9. This was by far the worst run: the agent learned very slowly and games stayed short (1208 episodes, about 250 steps each). With this learning rate, 300,000 steps were not enough. |
| 4 | 1e-4 | 0.95 | 32 | 1.0 | 0.05 | 100000 | Lower gamma (0.95), so future rewards count less. Average reward rose from 1.2 to 15.8, above the baseline (12.4). Learning was steady and it was one of the better runs, so the shorter reward horizon did not seem to hurt in Breakout. |
| 5 | 1e-4 | 0.999 | 32 | 1.0 | 0.05 | 100000 | Higher gamma (0.999), so the agent cares much more about far-future rewards. Average reward rose from 1.2 to 14.0, slightly above the baseline. It learned about as well as with gamma 0.99, so this change made little difference. |
| 6 | 1e-4 | 0.99 | 64 | 1.0 | 0.05 | 100000 | Larger batch size (64). Average reward rose from 1.2 to 13.8, close to the baseline. It had the fewest episodes (729), so games were longest on average, but each learning step costs more computation. The reward difference from the baseline is small. |
| 7 | 1e-4 | 0.99 | 16 | 1.0 | 0.05 | 100000 | Smaller batch size (16). Average reward rose from 1.2 to 9.6, below the baseline (12.4). It still learned, but less well, with more and shorter episodes (1022). Smaller batches give noisier updates, which may explain this. |
| 8 | 1e-4 | 0.99 | 32 | 1.0 | 0.05 | 50000 | Faster epsilon decay (50,000 steps), so less random exploration. Average reward rose from 1.1 to 18.8, the highest of all my experiments. Reaching epsilon 0.05 earlier probably gave the agent more steps playing with what it had learned. |
| 9 | 1e-4 | 0.99 | 32 | 1.0 | 0.05 | 250000 | Slower epsilon decay (250,000 steps), so random exploration lasted much longer. Average reward rose from 1.3 to 15.1, above the baseline but below experiment 8. Even with exploration lasting most of the run, the agent learned well. |
| 10 | 1e-4 | 0.99 | 32 | 1.0 | 0.01 | 100000 | Lower final epsilon (0.01), so almost no random actions at the end. Average reward rose from 0.8 to 13.3, close to the baseline (12.4). The lower start (0.8 against 1.2) is only random variation. The final epsilon had little effect here. |

## Best experiment

Best experiment: Experiment 8, highest average reward in the last 20 episodes (18.8).
