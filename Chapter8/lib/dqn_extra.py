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
    
    # this is required: the 𝜖 values are not updated after every optimization step
    # we have to call the reset_noise explicity
    # 𝜖 is not learnable parameter, it is drawn from the normal distribution
    def reset_noise(self):
        for n in self.noisy_layers:
            n.reset_noise()


    # signal-to-noise ratio (SNR) 
    # 𝑅𝑀𝑆(𝜇)/𝑅𝑀𝑆(𝜎), where 𝑅𝑀𝑆 is the root mean square of the corresponding weights
    # for metrics
    @torch.no_grad()
    def noisy_layers_sigma_snr(self) -> tt.List[float]:
        return [
            ((layer.weight_mu ** 2).mean().sqrt() /
             (layer.weight_sigma ** 2).mean().sqrt()).item()
            for layer in self.noisy_layers
        ]