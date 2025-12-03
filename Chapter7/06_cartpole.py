import gymnasium as gym
from ptan.experience import ExperienceFirstLast, ExperienceSourceFirstLast
import ptan
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import typing as tt


HIDDEN_SIZE = 128
BATCH_SIZE = 16
TGT_NET_SYNC = 10
GAMMA = 0.9
REPLAY_SIZE = 1000
LR = 1e-3
EPS_DECAY = 0.99


class Net(nn.Module):
    def __init__(self, obs_size: int, hidden_size: int, n_actions: int):
        super(Net, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_actions)
        )

    def forward(self, x):
        return self.net(x.float())

#  takes a sampled batch of ExperienceFirstLast objects and converts them into three tensors: states, actions, and target Q-values.
@torch.no_grad()
def unpack_batch(batch: tt.List[ExperienceFirstLast], net: Net, gamma: float):
    states = []
    actions = []
    rewards = []
    done_masks = []
    last_states = []
    for exp in batch:
        states.append(exp.state)
        actions.append(exp.action)
        rewards.append(exp.reward)
        done_masks.append(exp.last_state is None)
        if exp.last_state is None:
            last_states.append(exp.state)
        else:
            last_states.append(exp.last_state)

    states_v = torch.as_tensor(np.stack(states))
    actions_v = torch.tensor(actions)
    rewards_v = torch.tensor(rewards)
    last_states_v = torch.as_tensor(np.stack(last_states))
    last_state_q_v = net(last_states_v)
    best_last_q_v = torch.max(last_state_q_v, dim=1)[0]
    best_last_q_v[done_masks] = 0.0
    return states_v, actions_v, best_last_q_v * gamma + rewards_v


if __name__ == "__main__":
    env = gym.make("CartPole-v1")
    obs_size = env.observation_space.shape[0]
    n_actions = env.action_space.n

    # create the two layer fed forward network
    net = Net(obs_size, HIDDEN_SIZE, n_actions)
    # create target net
    tgt_net = ptan.agent.TargetNet(net)

    # create an epsilon greedy action with argmax action slection
    selector = ptan.actions.ArgmaxActionSelector()
    selector = ptan.actions.EpsilonGreedyActionSelector(epsilon=1, selector=selector)
    
    # create an agent work on the the network and selects the epsilon argmax action
    agent = ptan.agent.DQNAgent(net, selector)

    # create an experience source
    exp_source = ptan.experience.ExperienceSourceFirstLast(env, agent, gamma=GAMMA)
    # create the experience buffer
    buffer = ptan.experience.ExperienceReplayBuffer(exp_source, buffer_size=REPLAY_SIZE)
    
    # use Adam for optimizer, LR as learning rate 
    optimizer = optim.Adam(net.parameters(), LR)


    step = 0
    episode = 0
    solved = False

    while True:
        step += 1
        # still alive, reward += 1
        buffer.populate(1)

        # get a experience, update values
        for reward, steps in exp_source.pop_rewards_steps():
            episode += 1
            print(f"{step}: episode {episode} done, reward={reward:.2f}, "
                  f"epsilon={selector.epsilon:.2f}")
            solved = reward > 150
        if solved:
            print("Whee!")
            break

        # replay buffer too small, keep updating
        if len(buffer) < 2*BATCH_SIZE:
            continue

        # sample BATCH_SIZE
        batch = buffer.sample(BATCH_SIZE)
        states_v, actions_v, tgt_q_v = unpack_batch(batch, tgt_net.target_model, GAMMA)

        optimizer.zero_grad()
        # get the q value
        q_v = net(states_v)
        # gather together
        q_v = q_v.gather(1, actions_v.unsqueeze(-1)).squeeze(-1)
        loss_v = F.mse_loss(q_v, tgt_q_v)   # calculate loss
        loss_v.backward()   # backprop gradient
        # optimize one step
        optimizer.step()
        # epsilon decay
        selector.epsilon *= EPS_DECAY

        # sync
        if step % TGT_NET_SYNC == 0:
            tgt_net.sync()