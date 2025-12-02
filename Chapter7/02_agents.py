import ptan
import torch
from torch import nn
import numpy as np

class DQNNet(nn.Module):
    def __init__(self, actions):
        super(DQNNet, self).__init__()
        self.actions = actions
    
    def forward(self, x):
        # we always produce diagonal tensor of shape
        # (batch_size, actions)
        return torch.eye(x.size()[0], self.actions)