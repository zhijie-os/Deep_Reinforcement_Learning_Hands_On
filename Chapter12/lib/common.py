import sys
import time
import numpy as np
import typing as tt

import torch
import torch.nn as nn
from ptan.experience import ExperienceFirstLast


class RewardTracker:
    def __init__(self, writer, stop_reward):
        self.writer = writer
        self.stop_reward = stop_reward

    def __enter__(self):
        self.ts = time.time()
        self.ts_frame = 0
        self.total_rewards = []
        return self

    def __exit__(self, *args):
        self.writer.close()

    def reward(self, reward, frame, epsilon=None):
        self.total_rewards.append(reward)
        speed = (frame - self.ts_frame) / (time.time() - self.ts)
        self.ts_frame = frame
        self.ts = time.time()
        mean_reward = np.mean(self.total_rewards[-100:])
        epsilon_str = "" if epsilon is None else ", eps %.2f" % epsilon
        print("%d: done %d games, mean reward %.3f, speed %.2f f/s%s" % (
            frame, len(self.total_rewards), mean_reward, speed, epsilon_str
        ))
        sys.stdout.flush()
        if epsilon is not None:
            self.writer.add_scalar("epsilon", epsilon, frame)
        self.writer.add_scalar("speed", speed, frame)
        self.writer.add_scalar("reward_100", mean_reward, frame)
        self.writer.add_scalar("reward", reward, frame)
        if mean_reward > self.stop_reward:
            print("Solved in %d frames!" % frame)
            return True
        return False

class AtariA2C(nn.Module):
    def __init__(self, input_shape: tt.Tuple[int, ...], n_actions: int):
        super(AtariA2C, self).__init__()

        # Visual Processing Pipeline
        # shared convoultion body for policy head and value head
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten()
        )

        # size represents the flattened feature dimension after covolutional layers
        size = self.conv(torch.zeros(1, *input_shape)).shape[1]

        self.policy = nn.Sequential(
            nn.Linear(size, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )

        self.value = nn.Sequential(
            nn.Linear(size, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )
    

    def forward(self, x):
        xx = x / 255
        conv_out = self.conv(xx)     # convolutional network forward pass
        return self.policy(conv_out), self.value(conv_out)  # policy and value heads forward pass
    
# takes the batch of environment transitions and return the batch of states, batch of actions taken, and batch of Q-values
def unpack_batch(batch, net, device, gamma, reward_steps):
    states = []
    actions = []
    rewards = []
    not_done_idx = []
    last_states = []

    for idx, exp in enumerate(batch):
        states.append(np.asarray(exp.state))
        actions.append(int(exp.action))
        # exp.reward = r1 + γ*r2 + γ²*r3 + ... + γ^{N-1}*r_N
        rewards.append(exp.reward)  # already contains the discounted reward for REWARD_STEPS
        if exp.last_state is not None:
            not_done_idx.append(idx)
            last_states.append(np.asarray(exp.last_state))
    
    # convert the gathered data into torch tensors and GPU if available
    states_t = torch.FloatTensor(np.asarray(states)).to(device)
    actions_t = torch.LongTensor(actions).to(device)

    
    # handle rewards

    # rewards_np estimate the future rewards after N steps
    rewards_np = np.array(rewards, dtype=np.float32)
    if not_done_idx:
        last_states_t = torch.FloatTensor(np.asarray(last_states)).to(device)
        last_vals_t = net(last_states_t)[1]
        last_vals_np = last_vals_t.data.cpu().numpy()[:, 0]
        last_vals_np *= gamma ** reward_steps
        rewards_np[not_done_idx] += last_vals_np

    ref_vals_t = torch.FloatTensor(rewards_np).to(device)
    return states_t, actions_t, ref_vals_t