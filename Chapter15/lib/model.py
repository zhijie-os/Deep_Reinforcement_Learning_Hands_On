import ptan
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

HID_SIZE = 128

class ModelA2C(nn.Module):
    def __init__(self, obs_size, act_size):
        super(ModelA2C, self).__init__()

        # shared feature extractor/backbone network
        # extract features (ONCE)
        self.base = nn.Sequential(
            nn.Linear(obs_size, HID_SIZE),
            nn.ReLU(),
        )

        # all following heads use SAME feature
        # return the mean value of the actions
        self.mu = nn.Sequential(
            nn.Linear(HID_SIZE, act_size),
            nn.Tanh(), # squash into -1 ... 1
        )

        # return the variance of the actions
        self.var = nn.Sequential(
            nn.Linear(HID_SIZE, act_size),
            nn.Softplus(), # make the variance positive
        )
        # critic head, returning the value of the state
        self.value = nn.Linear(HID_SIZE, 1) # state value

    def forward(self, x):
        base_out = self.base(x)
        return self.mu(base_out), self.var(base_out), self.value(base_out)
    
class AgentA2C(ptan.agent.BaseAgent):
    def __init__(self, net, device):
        self.net = net
        self.device = device
    
    def __call__(self, states, agent_states):
        states_v = ptan.agent.float32_preprocessor(states)

        states_v = states_v.to(self.device)

        mu_v, var_v, _ = self.net(states_v)
        mu = mu_v.data.cpu().numpy()
        sigma = torch.sqrt(var_v).data.cpu().numpy()
        actions = np.random.normal(mu, sigma)
        actions = np.clip(actions, -1, 1)
        return actions, agent_states