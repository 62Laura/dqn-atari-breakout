import torch
import torch.nn.functional as F


def train_step(online, target, buffer, optimizer, batch_size, gamma):
    device = next(online.parameters()).device

    states, actions, rewards, next_states, dones = buffer.sample(batch_size)
    states = states.to(device)
    actions = actions.to(device)
    rewards = rewards.to(device)
    next_states = next_states.to(device)
    dones = dones.to(device)

    # Q-values of the actions we took
    q_values = online(states)
    q_taken = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

    # target = r + gamma * max Q_target(s'), no future if done
    with torch.no_grad():
        max_next_q = target(next_states).max(dim=1).values
        q_target = rewards + gamma * max_next_q * (1 - dones)

    # Huber loss
    loss = F.smooth_l1_loss(q_taken, q_target)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()
