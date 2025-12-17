import numpy as np
from ptan.experience import ExperienceReplayBuffer, ExperienceSource, ExperienceFirstLast
import torch
from torch import nn as nn
from torchrl.modules import NoisyLinear
import typing as tt


# nn.Linear layer with weights calculated as w_i,j = u_i,j + o_i,j * e_i,j
# both u and o are trainable parameters
# e is random noise sampled from the normal distribution
# mu: learnasble mean (deterministic) part of the weights
# sigma: learnable standard deviation (noise scale)
# epsilon: random noise
class NoisyLinear(nn.Linear):
    def __init__(
            self, in_features:int, out_features:int,
            bias:bool = True, device: Optional[DEVIE_TYPING] = None,
            dtype: Optional[torch.dtype] = None, std_init: float = 0.1
    ):
        nn.Module.__init__(self)
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.std_init = std_init

        # create matrix mu u
        self.weight_mu = nn.Parameter(torch.empty((out_features, in_features), device=device, dtype=dtype, requires_grad=True))

        # create matrix sigma o
        self.weight_sigma = nn.Parameter(torch.empty((out_features, in_features), device=device, dtype=dtype, requires_grad=True))

        self.register_buffer("weight_epsilon", torch.empty(out_features, in_features, device=device, dtype = dtype))

        if bias:
            self.bias_mu = nn.Parameter(torch.empty(out_features, device=device, dtype=dtype, requires_grad=True))

            self.bias_sigma = nn.Parameter(torch.empty(out_features, device=device, dtype=dtype, requires_grad=True))

            self.register_buffer("bias_epsilon", torch.empty(out_features, device=device, dtype=dtype))
        else:
            self.bias_mu = None
        self.reset_parameters()
        self.reset_noise()