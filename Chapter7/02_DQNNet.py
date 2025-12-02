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

net = DQNNet(actions=3)
print(net(torch.zeros(2, 10)))

selector = ptan.actions.ArgmaxActionSelector()
agent = ptan.agent.DQNAgent(model=net, action_selector=selector)

# given a batch of two observations, each having five values
print(agent(torch.zeros(2, 5)))
# in the output, the agent returns a tuple of two objects
# 1. An array with actions to be executed 
# 2. A list with the agent's internal state; this is used for stateful agents and None for non-stateful agents

selector = ptan.actions.EpsilonGreedyActionSelector(epsilon=1.0)
agent = ptan.agent.DQNAgent(model=net, action_selector=selector)
print(agent(torch.zeros(10, 5))[0])

selector.epsilon = 0.5
print(agent(torch.zeros(10, 5))[0])

selector.epsilon = 0.1
print(agent(torch.zeros(10, 5))[0])