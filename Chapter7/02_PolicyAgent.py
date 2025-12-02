import ptan
import torch
from torch import nn
import numpy as np

class PolicyNet(nn.Module):
    def __init__(self, actions):
        super(PolicyNet, self).__init__()
        self.actions = actions
    
    def forward(self, x):
        # Now we produce the tensor with first two actions
        # having the same logit scores
        shape = (x.size()[0], self.actions)
        res = torch.zeros(shape, dtype = torch.float32)
        res[:, 0] = 1
        res[:, 1] = 1
        return res

net = PolicyNet(actions = 5)
print(net(torch.zeros(6, 10)))

'''
    >>> torch.nn.functional.softmax(torch.tensor([1., 1., 0., 0., 0.]))
    tensor([0.3222, 0.3222, 0.1185, 0.1185, 0.1185])
'''

selector = ptan.actions.ProbabilityActionSelector()
agent = ptan.agent.PolicyAgent(model=net, action_selector=selector, apply_softmax=True)
print(agent(torch.zeros(6, 5))[0])

# softmax operation produces non-zero probabilities for zero logits, so our 
# agent can still select actions with zero logit values