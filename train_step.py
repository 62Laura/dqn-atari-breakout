"""One DQN learning step (Member 3)."""

import torch
import torch.nn.functional as F


def train_step(online, target, buffer, optimizer, batch_size, gamma):
    """Update the online network on one random batch and return the loss."""
    device = next(online.parameters()).device

    # 1. Sample a random batch of past transitions.
    states, actions, rewards, next_states, dones = buffer.sample(batch_size)
    states = states.to(device)
    actions = actions.to(device)
    rewards = rewards.to(device)
    next_states = next_states.to(device)
    dones = dones.to(device)

    # 2. Q-values the online network gives to the actions we actually took.
    q_values = online(states)                                     # [batch, n_actions]
    q_taken = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)  # [batch]

    # 3. Bellman target from the target network (no gradients flow here).
    #    If the episode ended (done = 1) there is no future reward to add.
    with torch.no_grad():
        max_next_q = target(next_states).max(dim=1).values
        q_target = rewards + gamma * max_next_q * (1 - dones)

    # 4. Huber loss: like MSE for small errors, but less sensitive to big ones.
    loss = F.smooth_l1_loss(q_taken, q_target)

    # 5. Backpropagate and update the online network's weights.
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()
