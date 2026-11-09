import torch
from torch import nn


class QNetwork(nn.Module):
    """CNN that predicts a Q-value for each available action."""

    def __init__(self, n_actions):
        super().__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )

    def forward(self, state):
        state = state.float() / 255.0
        features = self.conv_layers(state)
        q_values = self.fc_layers(features)
        return q_values