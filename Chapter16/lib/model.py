import ptan
import numpy as np
import torch
import math
import torch.nn as nn
import gymnasium as gym

HID_SIZE = 64


class ModelActor(nn.Module):
    def __init__(self, obs_size: int, act_size: int):
        super(ModelActor, self).__init__()

        self.mu = nn.Sequential(
            nn.Linear(obs_size, HID_SIZE),
            nn.Tanh(),
            nn.Linear(HID_SIZE, HID_SIZE),
            nn.Tanh(),
            nn.Linear(HID_SIZE, act_size),
            nn.Tanh(),
        )
        self.logstd = nn.Parameter(torch.zeros(act_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mu(x)


class ModelCritic(nn.Module):
    def __init__(self, obs_size: int):
        super(ModelCritic, self).__init__()

        self.value = nn.Sequential(
            nn.Linear(obs_size, HID_SIZE),
            nn.ReLU(),
            nn.Linear(HID_SIZE, HID_SIZE),
            nn.ReLU(),
            nn.Linear(HID_SIZE, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.value(x)

class AgentA2C(ptan.agent.BaseAgent):
    def __init__(self, net, device: torch.device):
        self.net = net
        self.device = device

    def __call__(self, states: ptan.agent.States, agent_states: ptan.agent.AgentStates):
        states_v = ptan.agent.float32_preprocessor(states)
        states_v = states_v.to(self.device)

        mu_v = self.net(states_v)
        mu = mu_v.data.cpu().numpy()
        logstd = self.net.logstd.data.cpu().numpy()
        rnd = np.random.normal(size=logstd.shape)
        actions = mu + np.exp(logstd) * rnd
        actions = np.clip(actions, -1, 1)
        return actions, agent_states


def test_net(
        net: ModelActor, env: gym.Env, count: int = 10,
        device: torch.device = torch.device("cpu")
):
    rewards = 0.0
    steps = 0
    for _ in range(count):
        obs, _ = env.reset()
        while True:
            obs_v = ptan.agent.float32_preprocessor([obs])
            obs_v = obs_v.to(device)
            mu_v = net(obs_v)[0]
            action = mu_v.squeeze(dim=0).data.cpu().numpy()
            action = np.clip(action, -1, 1)
            obs, reward, done, is_tr, _ = env.step(action)
            rewards += reward
            steps += 1
            if done or is_tr:
                break
    return rewards / count, steps / count


def calc_logprob(mu_v: torch.Tensor, logstd_v: torch.Tensor, actions_v: torch.Tensor):
    p1 = - ((mu_v - actions_v) ** 2) / (2*torch.exp(logstd_v).clamp(min=1e-3))
    p2 = - torch.log(torch.sqrt(2 * math.pi * torch.exp(logstd_v)))
    return p1 + p2
