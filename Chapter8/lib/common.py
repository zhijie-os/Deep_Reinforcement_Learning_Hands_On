import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import warnings
import dataclasses
from datetime import timedelta, datetime
import typing as tt

import ptan.ignite as ptan_ignite
from ptan.actions import EpsilonGreedyActionSelector
from ptan.experience import ExperienceFirstLast, \
    ExperienceSourceFirstLast, ExperienceReplayBuffer

from ignite.engine import Engine
from ignite.metrics import RunningAverage
from ignite.contrib.handlers import tensorboard_logger as tb_logger
from ray import tune

import ale_py # This import is necessary for Pong environments to register 


SEED = 123

@dataclasses.dataclass
class Hyperparams:
    env_name: str
    stop_reward: float
    run_name: str
    replay_size: int
    replay_initial: int
    target_net_sync: int
    epsilon_frames: int

    learning_rate: float = 0.0001
    batch_size: int = 32
    gamma: float = 0.99
    epsilon_start: float = 1.0
    epsilon_final: float = 0.1

    tuner_mode: bool = False
    episodes_to_solve:int = 500


GAME_PARAMS = {
    'pong': Hyperparams (
        env_name="PongNoFrameskip-v4",
        stop_reward = 18.0,
        run_name="pong",
        replay_size = 100000,
        replay_initial = 10000,
        target_net_sync = 1000,
        epsilon_frames = 100000,
        epsilon_final = 0.02,
    ),
}

# this function unpacks batch of ExperienceFirstLast into separate arrays
def unpack_batch(batch: tt.List[ExperienceFirstLast]):
    states, actions, rewards, dones, last_states = [], [], [], [], []
    for exp in batch:
        states.append(exp.state)
        actions.append(exp.action)
        rewards.append(exp.reward)
        dones.append(exp.last_state is None)

        if exp.last_state is None:
            lstate = exp.state  # the result will be masked anyway
        else:
            lstate = exp.last_state
        last_states.append(lstate)
    return np.asarray(states), np.array(actions), np.array(rewards, dtype=np.float32), \
        np.array(dones, dtype=np.uint8), np.asarray(last_states)

def calc_loss_dqn(batch: tt.List[ExperienceFirstLast],
                  net: nn.Module,
                  tgt_net: nn.Module,
                  gamma: float,
                  device: torch.device = torch.device("cuda")) -> torch.Tensor:
    states, actions, rewards, dones, last_states = unpack_batch(batch)

    states_v = torch.tensor(states).to(device)
    actions_v = torch.tensor(actions).to(device)
    rewards_v = torch.tensor(rewards).to(device)
    dones_v = torch.tensor(dones).to(device)
    last_states_v = torch.tensor(last_states).to(device)

    actions_v = actions_v.unsqueeze(-1)
    state_action_vals = net(states_v).gather(1, actions_v)
    state_action_vals = state_action_vals.squeeze(-1)

    with torch.no_grad():
        next_state_values = tgt_net(last_states_v).max(1)[0]
        # PyTorch boolean indexing operation (works similarly to NumPy).
        next_state_values[dones_v] = 0.0
        expected_state_action_values = rewards_v + gamma * next_state_values

    return nn.MSELoss()(state_action_vals, expected_state_action_values)

class EpsilonTracker:
    def __init__(self, selector: EpsilonGreedyActionSelector, params: Hyperparams):
        self.selector = selector
        self.params = params
        self.frame(0)   # call the first epsilon udpate

    def frame(self, frame_idx: int):
        eps = self.params.epsilon_start - frame_idx / self.params.epsilon_frames
        self.selector.epsilon = max(self.params.epsilon_final, eps)

#  this takes a replay buffer and yields batches of ExperienceFirstLast objects
def batch_generator(buffer: ExperienceReplayBuffer, initial: int, batch_size: int) \
    -> tt.Generator[tt.List[ExperienceFirstLast], None, None]:
    buffer.populate(initial)
    while True:
        buffer.populate(1)
        # each call to next() resumes this generator, and runs the loop body until it reaches the next yield
        yield buffer.sample(batch_size)

# Ignite is PyTorch's training loop library which gives event-driven monitoring
def setup_ignite(engine: Engine,
                 params: Hyperparams,
                 exp_source: ExperienceSourceFirstLast,
                 run_name: str,
                 extra_metrics: tt.Iterable[str] = (),
                 tuner_reward_episode: int = 100,
                 tuner_reward_min: float = -19):
    # the EndOfEpisodeHanlder emits the Ignite event very time a game episode ends
    handler = ptan_ignite.EndOfEpisodeHandler(
        exp_source, bound_avg_reward=params.stop_reward
    )
    handler.attach(engine)
    # the EpisodeFPSHanlder, tracks the time the episode has taken and 
    # the amount of interactions that we have had with the environment.
    # We calculate frames per seconds (FPS) as a performance metric
    ptan_ignite.EpisodeFPSHanlder().attach(engine)

    @engine.on(ptan_ignite.EpisodeEvents.EPISODE_COMPLETED)
    def episode_completed(trainer: Engine):
        passed = trainer.state.metrics.get("time_passed", 0.0)
        print("Episode %d: reward=%.0f, steps=%s, speed=%.1f f/s, elapsed=%s" %(
            trainer.state.episode,
            trainer.state.episode_reward,
            trainer.state.episode_steps,
            trainer.state.metrics.get("avg_fps", 0.0),
            timedelta(seconds=int(passed))
        ))
    
    @engine.on(ptan_ignite.EpisodeEvents.BOUND_REWARD_RECHAED)
    def game_solve(trainer: Engine):
        passed = trainer.state.metrics['time_passed']
        print("Game solved in %s, after %d episodes and %d iterations!" % (
            timedelta(seconds=int(passed)),
            trainer.state.episode,
            trainer.state.iteration
        ))
        trainer.should_terminate = True
        trainer.state.solved = True

    now = datetime.now().isoformat(timespec="minutes").replace(":", " ")
    logdir = f"runs/{now} - {params.run_game} - {run_name}"
    tb = tb_logger.TensorboardLogger(log_dir=logdir)
    run_avg = RunningAverage(output_transform=lambda v:v["loss"])
    run_avg.attach(engine, "avg_loss")

    metrics = ['reward', 'steps', 'avg_reward']
    handler = tb_logger.OutputHandler(tag="episodes", metric_names=metrics)
    event = ptan_ignite.EpisodeEvents.EPISODE_COMPLETED
    tb.attach(engine, log_handler=handler, event_name=event)

    # write to tensorboard every 100 iterations
    ptan_ignite.PeriodicEvents().attach(engine)
    metrics = ['avg_loss', 'avg_fps']
    metrics.extend(extra_metrics)
    handler = tb_logger.OutputHandler(tag="train", metric_names=metrics,
                                      output_transform=lambda a: a)
    event = ptan_ignite.PeriodEvents.ITERS_100_COMPLETED
    tb.attach(engine, log_handler=handler, event_name=event)


    if params.tuner_mode:
        @engine.on(ptan_ignite.EpisodeEvents.EPISODE_COMPLETED)
        def episode_completed(trainer: Engine):
            avg_reward = trainer.state.metrics.get('avg_reward')
            max_episodes = params.episodes_to_solve * 1.1
            if trainer.state.episode > tuner_reward_episode and \
                    avg_reward < tuner_reward_min:
                trainer.should_terminate = True
                trainer.state.solved = False
            elif trainer.state.episode > max_episodes:
                trainer.should_terminate = True
                trainer.state.solved = False
            if trainer.should_terminate:
                print(f"Episode {trainer.state.episode}, "
                      f"avg_reward {avg_reward:.2f}, terminating")