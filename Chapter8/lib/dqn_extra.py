import numpy as np
from ptan.experience import ExperienceReplayBuffer, ExperienceSource, ExperienceFirstLast
import torch
from torch import nn as nn
from torchrl.modules import NoisyLinear
import typing as tt


# replay buffer params
BETA_START = 0.4
BETA_FRAMES = 100000

class NoisyDQN(nn.Module):
    def __init__(self, input_shape: tt.Tuple[int,...],
                 n_actions: int):
        super(NoisyDQN, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        size = self.conv(torch.zeros(1, *input_shape)).size()[-1]
        self.noisy_layers = [
            NoisyLinear(size, 512),
            NoisyLinear(512, n_actions)
        ]

        # only difference is swapping nn.Linear with NoisyLinear
        self.fc = nn.Sequential(
            self.noisy_layers[0],
            nn.ReLU(),
            self.noisy_layers[1],
        )
    
    def forward(self, x: torch.ByteTensor):
        xx = x / 255.0
        return self.fc(self.conv(xx))