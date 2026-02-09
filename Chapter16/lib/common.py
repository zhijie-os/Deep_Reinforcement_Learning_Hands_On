import numpy as np
import torch
import torch.distributions as distr
import gymnasium as gym
import typing as tt
import ptan

from lib import model


ENV_IDS = {
    'cheetah': "HalfCheetahBulletEnv-v0",
    'cheetah-mujoco': "HalfCheetah-v4",
    'ant': "AntBulletEnv-v0",
    'ant-mujoco': "Ant-v4",
}

ENV_PARAMS = {
    'cheetah': ('pybullet_envs.gym_locomotion_envs:HalfCheetahBulletEnv', 1000, 3000.0),
    'ant': ('pybullet_envs.gym_locomotion_envs:AntBulletEnv', 1000, 2500.0),
}


def register_env(name: str, mujoco: bool) -> str:
    if mujoco:
        real_id = ENV_IDS[name + "-mujoco"]
    else:
        # register environment in gymnasium registry, not gym's
        real_id = ENV_IDS[name]
        entry, steps, reward = ENV_PARAMS[name]
        gym.register(
            real_id, entry_point=entry,
            max_episode_steps=steps, reward_threshold=reward,
            apply_api_compatibility=True,
            disable_env_checker=True,
        )
    return real_id


def unpack_batch_a2c(
        batch: tt.List[ptan.experience.ExperienceFirstLast],
        net: model.ModelCritic,
        last_val_gamma: float,
        device: torch.device):
    """
    Convert batch into training tensors
    """
    states = []
    actions = []
    rewards = []
    not_done_idx = []
    last_states = []
    for idx, exp in enumerate(batch):
        states.append(exp.state)
        actions.append(exp.action)
        rewards.append(exp.reward)
        if exp.last_state is not None:
            not_done_idx.append(idx)
            last_states.append(exp.last_state)
    states_v = ptan.agent.float32_preprocessor(states).to(device)
    actions_v = torch.FloatTensor(np.asarray(actions)).to(device)

    # handle rewards
    rewards_np = np.array(rewards, dtype=np.float32)
    if not_done_idx:
        last_states_v = ptan.agent.float32_preprocessor(last_states).to(device)
        last_vals_v = net(last_states_v)
        last_vals_np = last_vals_v.data.cpu().numpy()[:, 0]
        rewards_np[not_done_idx] += last_val_gamma * last_vals_np

    ref_vals_v = torch.FloatTensor(rewards_np).to(device)
    return states_v, actions_v, ref_vals_v

