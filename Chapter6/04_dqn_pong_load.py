import gymnasium as gym
from lib import dqn_model
from lib import wrappers

from dataclasses import dataclass
import argparse
import time
import numpy as np
import collections
import typing as tt
import os
import re
import glob

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.tensorboard.writer import SummaryWriter
import ale_py # This import is necessary for environments to register 

DEFAULT_ENV_NAME = "PongNoFrameskip-v4"
MEAN_REWARD_BOUND = 19

GAMMA = 0.99
BATCH_SIZE = 32
REPLAY_SIZE = 10000
LEARNING_RATE = 1e-4
SYNC_TARGET_FRAMES = 1000
REPLAY_START_SIZE = 10000

EPSILON_DECAY_LAST_FRAME = 15000
EPSILON_START = 1.0
EPSILON_FINAL = 0.01

State = np.ndarray
Action = int
BatchTensors = tt.Tuple[
    torch.ByteTensor,           # current state
    torch.LongTensor,           # actions
    torch.Tensor,               # rewards
    torch.BoolTensor,           # done || trunc
    torch.ByteTensor            # next state
]

@dataclass
class Experience:
    state: State
    action: Action
    reward: float
    done_trunc: bool
    new_state: State


class ExperienceBuffer:
    def __init__(self, capacity: int):
        self.buffer = collections.deque(maxlen=capacity)

    def __len__(self):
        return len(self.buffer)

    def append(self, experience: Experience):
        self.buffer.append(experience)

    def sample(self, batch_size: int) -> tt.List[Experience]:
        indices = np.random.choice(len(self), batch_size, replace=False)
        return [self.buffer[idx] for idx in indices]


class Agent:
    def __init__(self, env: gym.Env, exp_buffer: ExperienceBuffer):
        self.env = env
        self.exp_buffer = exp_buffer
        self.state: tt.Optional[np.ndarray] = None
        self._reset()

    def _reset(self):
        self.state, _ = self.env.reset()
        self.total_reward = 0.0

    @torch.no_grad()
    def play_step(self, net: dqn_model.DQN, device: torch.device,
                  epsilon: float = 0.0) -> tt.Optional[float]:
        done_reward = None

        if np.random.random() < epsilon:
            action = self.env.action_space.sample()
        else:
            state_v = torch.as_tensor(self.state).to(device)
            state_v.unsqueeze_(0)
            q_vals_v = net(state_v)
            _, act_v = torch.max(q_vals_v, dim=1)
            action = int(act_v.item())

        # do step in the environment
        new_state, reward, is_done, is_tr, _ = self.env.step(action)
        self.total_reward += reward

        exp = Experience(
            state=self.state, action=action, reward=float(reward),
            done_trunc=is_done or is_tr, new_state=new_state
        )
        self.exp_buffer.append(exp)
        self.state = new_state
        if is_done or is_tr:
            done_reward = self.total_reward
            self._reset()
        return done_reward


def batch_to_tensors(batch: tt.List[Experience], device: torch.device) -> BatchTensors:
    states, actions, rewards, dones, new_state = [], [], [], [], []
    for e in batch:
        states.append(e.state)
        actions.append(e.action)
        rewards.append(e.reward)
        dones.append(e.done_trunc)
        new_state.append(e.new_state)
    states_t = torch.as_tensor(np.asarray(states))
    actions_t = torch.LongTensor(actions)
    rewards_t = torch.FloatTensor(rewards)
    dones_t = torch.BoolTensor(dones)
    new_states_t = torch.as_tensor(np.asarray(new_state))
    return states_t.to(device), actions_t.to(device), rewards_t.to(device), \
           dones_t.to(device),  new_states_t.to(device)


def calc_loss(batch: tt.List[Experience], net: dqn_model.DQN, tgt_net: dqn_model.DQN,
              device: torch.device) -> torch.Tensor:
    states_t, actions_t, rewards_t, dones_t, new_states_t = batch_to_tensors(batch, device)

    state_action_values = net(states_t).gather(
        1, actions_t.unsqueeze(-1)
    ).squeeze(-1)
    with torch.no_grad():
        next_state_values = tgt_net(new_states_t).max(1)[0]
        next_state_values[dones_t] = 0.0
        next_state_values = next_state_values.detach()

    expected_state_action_values = next_state_values * GAMMA + rewards_t
    return nn.MSELoss()(state_action_values, expected_state_action_values)


def find_latest_model(env_name: str) -> str:
    """Find the latest/best model based on filename pattern."""
    pattern = f"{env_name}-best_*.dat"
    files = glob.glob(pattern)
    
    if not files:
        return ""
    
    # Extract reward from filename and find the highest (least negative)
    def extract_reward(filename):
        match = re.search(r'best_(-?\d+)\.dat$', filename)
        if match:
            return int(match.group(1))
        return -float('inf')
    
    # Sort by reward (highest first)
    files.sort(key=extract_reward, reverse=True)
    return files[0]


def find_best_model(env_name: str) -> str:
    """Find the model with the highest reward in filename."""
    return find_latest_model(env_name)  # Same logic since we sort by reward


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dev", default="cpu", help="Device name, default=cpu")
    parser.add_argument("--env", default=DEFAULT_ENV_NAME,
                        help="Name of the environment, default=" + DEFAULT_ENV_NAME)
    parser.add_argument("--load", default="", 
                       help="Load pretrained model from specific file")
    parser.add_argument("--load-latest", action="store_true", default=False,
                       help="Load the latest model (highest reward) automatically")
    parser.add_argument("--load-best", action="store_true", default=False,
                       help="Alias for --load-latest")
    parser.add_argument("--continue-from", type=int, default=0,
                       help="Continue training from this frame index (affects epsilon decay)")
    args = parser.parse_args()
    device = torch.device(args.dev)

    env = wrappers.make_env(args.env)
    net = dqn_model.DQN(env.observation_space.shape, env.action_space.n).to(device)
    tgt_net = dqn_model.DQN(env.observation_space.shape, env.action_space.n).to(device)
    
    # Determine which model to load
    model_to_load = args.load
    
    if not model_to_load and (args.load_latest or args.load_best):
        model_to_load = find_latest_model(args.env)
        if model_to_load:
            print(f"Found latest model: {model_to_load}")
    
    # Load pretrained model if specified
    if model_to_load:
        try:
            if os.path.exists(model_to_load):
                state_dict = torch.load(model_to_load, map_location=device)
                net.load_state_dict(state_dict)
                tgt_net.load_state_dict(state_dict)
                
                # Extract reward from filename for reporting
                match = re.search(r'best_(-?\d+)\.dat$', model_to_load)
                if match:
                    reward = int(match.group(1))
                    print(f"✓ Loaded model: {model_to_load} (reward: {reward})")
                else:
                    print(f"✓ Loaded model: {model_to_load}")
                
                # Also try to extract frame count if present in other naming patterns
                # (e.g., PongNoFrameskip-v4-50000.dat)
                frame_match = re.search(r'-(\d+)\.dat$', model_to_load)
                if frame_match and args.continue_from == 0:
                    # Auto-detect frame count from filename if not manually specified
                    args.continue_from = int(frame_match.group(1))
            else:
                print(f"Warning: Model file {model_to_load} not found, starting from scratch")
                model_to_load = ""
        except Exception as e:
            print(f"Error loading model {model_to_load}: {e}")
            print("Starting from scratch")
            model_to_load = ""
    
    writer = SummaryWriter(comment="-" + args.env + ("-continued" if model_to_load else ""))
    print(net)

    buffer = ExperienceBuffer(REPLAY_SIZE)
    agent = Agent(env, buffer)
    epsilon = EPSILON_START

    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)
    total_rewards = []
    
    # Start from specified frame index (useful when continuing training)
    frame_idx = args.continue_from
    
    # Adjust epsilon based on frame_idx if continuing
    if frame_idx > 0:
        epsilon = max(EPSILON_FINAL, EPSILON_START - frame_idx / EPSILON_DECAY_LAST_FRAME)
        print(f"Continuing from frame {frame_idx}, epsilon={epsilon:.3f}")
    
    ts_frame = frame_idx
    ts = time.time()
    best_m_reward = None

    while True:
        frame_idx += 1
        epsilon = max(EPSILON_FINAL, EPSILON_START - frame_idx / EPSILON_DECAY_LAST_FRAME)

        reward = agent.play_step(net, device, epsilon)
        if reward is not None:
            total_rewards.append(reward)
            speed = (frame_idx - ts_frame) / (time.time() - ts)
            ts_frame = frame_idx
            ts = time.time()
            m_reward = np.mean(total_rewards[-100:])
            print(f"{frame_idx}: done {len(total_rewards)} games, reward {m_reward:.3f}, "
                  f"eps {epsilon:.2f}, speed {speed:.2f} f/s")
            writer.add_scalar("epsilon", epsilon, frame_idx)
            writer.add_scalar("speed", speed, frame_idx)
            writer.add_scalar("reward_100", m_reward, frame_idx)
            writer.add_scalar("reward", reward, frame_idx)
            
            # Save best model when mean reward improves
            if best_m_reward is None or m_reward > best_m_reward:
                # Format filename with reward (negative numbers are fine)
                filename = f"{args.env}-best_{int(m_reward)}.dat"
                torch.save(net.state_dict(), filename)
                if best_m_reward is not None:
                    print(f"✓ Best reward updated {best_m_reward:.3f} -> {m_reward:.3f}")
                    print(f"  Model saved as: {filename}")
                else:
                    print(f"✓ First model saved: {filename}")
                best_m_reward = m_reward
            
            # Also save periodic checkpoints
            if frame_idx % 50000 == 0:
                checkpoint_name = f"{args.env}-checkpoint-{frame_idx}.dat"
                torch.save(net.state_dict(), checkpoint_name)
                print(f"✓ Periodic checkpoint saved: {checkpoint_name}")
            
            if m_reward > MEAN_REWARD_BOUND:
                print(f"🎉 Solved in {frame_idx} frames!")
                # Save final model
                final_name = f"{args.env}-solved-{frame_idx}.dat"
                torch.save(net.state_dict(), final_name)
                print(f"Final model saved: {final_name}")
                break
        
        # Fill buffer before training
        if len(buffer) < REPLAY_START_SIZE:
            continue
        
        # Sync target network
        if frame_idx % SYNC_TARGET_FRAMES == 0:
            tgt_net.load_state_dict(net.state_dict())

        # Training step
        optimizer.zero_grad()
        batch = buffer.sample(BATCH_SIZE)
        loss_t = calc_loss(batch, net, tgt_net, device)
        loss_t.backward()
        optimizer.step()
    
    writer.close()