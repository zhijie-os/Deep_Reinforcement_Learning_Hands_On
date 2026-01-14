import gymnasium as gym
import ptan
from ptan.experience import ExperienceSourceFirstLast
import numpy as np
import typing as tt
from tensorboardX import SummaryWriter

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

GAMMA = 0.99
LEARNING_RATE = 0.01
EPISODES_TO_TRAIN = 4


class PGN(nn.Module):
    def __init__(self, input_size: int, n_actions: int):
        super(PGN, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

def calc_qvals(rewards: tt.List[float]) -> tt.List[float]:
    res = []
    sum_r = 0.0
    for r in reversed(rewards):
        sum_r *= GAMMA
        sum_r += r
        res.append(sum_r)
    res = list(reversed(res))
    mean_q = np.mean(res)
    return [q - mean_q for q in res]



if __name__ == "__main__":
    env = gym.make("CartPole-v1")
    writer = SummaryWriter(comment="-cartpole-reinforce")

    net = PGN(env.observation_space.shape[0], env.action_space.n)
    print(net)

    print(net)

    agent = ptan.agent.PolicyAgent(
        net, preprocessor=ptan.agent.float32_preprocessor, apply_softmax=True
    )

    exp_source = ExperienceSourceFirstLast(env, agent, gamma=GAMMA)

    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)

    total_rewards = [] # repoting and contain the total rewards for the episodes
    done_episodes = 0 # count of completed episodes

    batch_episodes = 0
    batch_states, batch_actions, batch_qvals = [], [], []
    cur_rewards = []    # local rewards for the episode being currently played

    for step_idx, exp in enumerate(exp_source):
        batch_states.append(exp.state)
        batch_actions.append(int(exp.action))
        cur_rewards.append(exp.reward)

        if exp.last_state is None:   # episode completed
            batch_qvals.extend(calc_qvals(cur_rewards))
            cur_rewards.clear()
            batch_episodes += 1
    
        # reporting the current progress and writing metrics to TensorBoard
        new_rewards = exp_source.pop_total_rewards()
        if new_rewards:
            done_episodes += 1
            reward = new_rewards[0]
            total_rewards.append(reward)
            mean_rewards = float(np.mean(total_rewards[-100:]))
            print(f"{step_idx}: reward: {reward:6.2f}, mean_100: {mean_rewards:6.2f}, "
                  f"episodes: {done_episodes}")
            writer.add_scalar("reward", reward, step_idx)
            writer.add_scalar("reward_100", mean_rewards, step_idx)
            writer.add_scalar("episodes", done_episodes, step_idx)
            if mean_rewards > 450:
                print(f"Solved in {step_idx} steps and {done_episodes} episodes!")
                break   # break when the rewards threshold is reached
        
        # continue to train only when we have enough episodes
        if batch_episodes < EPISODES_TO_TRAIN:
            continue
            
        optimizer.zero_grad()
        states_t = torch.as_tensor(np.asarray(batch_states))
        batch_actions_t = torch.as_tensor(np.asarray(batch_actions))
        batch_qvals_t = torch.as_tensor(np.asarray(batch_qvals))

        logits_t = net(states_t)
        log_prob_t = F.log_softmax(logits_t, dim=1) # Convert to log-probabilities using log_softmax
        batch_idx = range(len(batch_states))
        act_probs_t = log_prob_t[batch_idx, batch_actions_t]
        log_prob_actions_v = batch_qvals_t * act_probs_t
        loss_t = -log_prob_actions_v.mean() # negative for gradient ascent

        loss_t.backward()
        optimizer.step()

        # clear the batch data for on-policy training
        batch_episodes = 0
        batch_states.clear()
        batch_actions.clear()
        batch_qvals.clear()
    writer.close()