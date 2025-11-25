# out model's core is a one-hidden layer NN
# with rectified linear unit (ReLU) and 128 hidden neurons
import numpy as np
import gymnasium as gym
from dataclasses import dataclass
import typing as tt
from torch.utils.tensorboard.writer import SummaryWriter

import torch
import torch.nn as nn
import torch.optim as optim

HIDDEN_SIZE = 128
BATCH_SIZE = 16
PERCENTILE = 70

# off-policy is that the model learns from historical data
# on-policy requires fresh data that the policy we are currently updating

'''
    The core of cross-entropy is to throw away bad episode and train on better ones:
    1. Play N episodes using our current model and environment
    2. Calculate the total reward for each episode and decide on a reward boundary. Usually, we use a percentile of all rewards, such as the 50th or 75th.
    3. Throw away all episodes with a reward below that percentile
    4. Train on the remaining "elite" episodes using observations as the input and issued actions as the desired output
    5. Repeat from step 1
'''

class Net(nn.Module):
    def __init__(self, obs_size, hidden_size, n_actions):
        super(Net, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_actions)
            ) # the output would be probability of actions
    
    def forward(self, x):
        return self.net(x) # the output from the NN is probability distribution over actions
    
@dataclass
class EpisodeStep:
    observation: np.ndarray
    action: int

@dataclass
class Episode: # a sequence of episode step
    reward: float
    steps: tt.List[EpisodeStep]

# generate batches with episodes
def iterate_batches(env: gym.Env, net: Net, batch_size: int) -> tt.Generator[tt.List[Episode], None, None]:
    batch = []
    episode_reward = 0.0
    episode_steps = []
    obs, _ = env.reset()    
    sm = nn.Softmax(dim = 1)    # map the output to distribution of actions

    while True:
        obs_v = torch.tensor(obs, dtype=torch.float32) # observation is converted to tensor
        act_probs_v = sm(net(obs_v.unsqueeze(0)))   # use the observation into the network
        act_probs = act_probs_v.data.numpy()[0] # get the probabilities
    
        action = np.random.choice(len(act_probs), p=act_probs)
        next_obs, reward, is_done, is_trunc, _ = env.step(action)

        episode_reward += float(reward)
        step = EpisodeStep(observation = obs, action=action)
        episode_steps.append(step)

        if is_done or is_trunc:
            e = Episode(reward=episode_reward, steps=episode_steps)
            batch.append(e)
            episode_reward = 0.0
            episode_steps = []
            next_obs, _ = env.reset()
            if len(batch) == batch_size:
                yield batch # used to create a generator function
                batch = []
        obs = next_obs

# from the given batch of episodes and percentile value, it calculates a boundary reward, which is used to filter elite episodes to train on.
def filter_batch(batch: tt.List[Episode], percentile: float) -> \
    tt.Tuple[torch.FloatTensor, torch.LongTensor, float, float]:
    rewards = list(map(lambda s:s.reward, batch))   # get all reward
    reward_bound = float(np.percentile(rewards, percentile))     # find the percentile to eliminate
    reward_mean = float(np.mean(rewards))   # find the mean of the reward

    train_obs: tt.List[np.ndarray] = []
    train_act: tt.List[int] = []
    for episode in batch:
        if episode.reward < reward_bound:   # get rid of unsatisfying reward
            continue
        train_obs.extend(map(lambda step: step.observation, episode.steps)) # get all the observation
        train_act.extend(map(lambda step: step.action, episode.steps))  # get all the actions
    train_obs_v = torch.FloatTensor(np.vstack(train_obs))
    train_act_v = torch.LongTensor(train_act)
    return train_obs_v, train_act_v, reward_bound, reward_mean

if __name__ == "__main__":
    # we create all the required objects: the environment, our NN, the objective function, the optimizer, and the summary writer for TensorBoard.
    # env = gym.make("CartPole-v1")
    env = gym.make("CartPole-v1", render_mode="rgb_array")
    env = gym.wrappers.RecordVideo(env, video_folder="video")   
    assert env.observation_space.shape is not None
    obs_size = env.observation_space.shape[0]
    assert isinstance(env.action_space, gym.spaces.Discrete)
    n_actions = int(env.action_space.n)

    net = Net(obs_size, HIDDEN_SIZE, n_actions)
    print(net)
    objective = nn.CrossEntropyLoss()
    optimizer = optim.Adam(params=net.parameters(), lr=0.01)
    writer = SummaryWriter(comment="-cartpole")

    for iter_no, batch in enumerate(iterate_batches(env, net, BATCH_SIZE)):
        # tensors of observations and taken actions, reward boundary used for filtering and mean reward
        obs_v, acts_v, reward_b, reward_m = filter_batch(batch, PERCENTILE)
        optimizer.zero_grad()   # clean the gradient
        action_score_v = net(obs_v) # get actions scores for observations
        loss_v = objective(action_score_v, acts_v)  # calculate the cross-entropy between the NN output and the actions that the agent took
        # Compare with actions that worked well
        loss_v.backward()
        optimizer.step()
        print("%d: loss=%.3f, reward_mean=%.1f, rw_bound=%.1f" % (iter_no, loss_v.item(), reward_m, reward_b))
        writer.add_scalar("loss", loss_v.item(), iter_no)
        writer.add_scalar("reward_bound", reward_b, iter_no)
        writer.add_scalar("reward_mean", reward_m, iter_no)
        if reward_m > 475:
            print("Solved!")
            break
        writer.close()
