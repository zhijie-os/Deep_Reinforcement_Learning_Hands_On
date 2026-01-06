

#!/usr/bin/env python3
import gymnasium as gym
import ptan
import ptan.ignite as ptan_ignite
from datetime import datetime, timedelta
import argparse
import random
import warnings
import typing as tt
from dataclasses import dataclass

# multiprocessing support
import torch.multiprocessing as mp

import torch
import torch.optim as optim

from ignite.engine import Engine
from ignite.metrics import RunningAverage
from ignite.contrib.handlers import tensorboard_logger as tb_logger

import ale_py # This import is necessary for environments to register 

from lib import dqn_model, common

NAME = "03_parallel"


# dataclass for an experience
@dataclass
class EpisodeEnded:
    reward: float
    steps: int
    epsilon: float

# play process and will be running in a seperate child process started by the main process
# get experience from the enviornment and push it into the shared queue
def play_func(params: common.Hyperparams, net: dqn_model.DQN,
              dev_name: str, exp_queue: mp.Queue):
    # create the environment
    env = gym.make(params.env_name)
    env = ptan.common.wrappers.wrap_dqn(env)
    device = torch.device(dev_name)

    # create selector, epsilon tracker, agent and Experience buffer
    selector = ptan.actions.EpsilonGreedyActionSelector(epsilon=params.epsilon_start)
    epsilon_tracker = common.EpsilonTracker(selector, params)
    agent = ptan.agent.DQNAgent(net, selector, device=device)

    # This creates an ITERATOR that yields experiences
    exp_source = ptan.experience.ExperienceSourceFirstLast(
        env, agent, gamma=params.gamma, env_seed=common.SEED)

    # run the environment and push experience into the shared queue
    for frame_idx, exp in enumerate(exp_source):    # Each loop = 1 environment step
        epsilon_tracker.frame(frame_idx//2)
        exp_queue.put(exp)  # push experience into the shared queue
        for reward, steps in exp_source.pop_rewards_steps():
            ee = EpisodeEnded(reward=reward, steps=steps, epsilon=selector.epsilon)
            exp_queue.put(ee)


class BatchGenerator:
    def __init__(self, buffer_size: int, exp_queue: mp.Queue,
                 fps_handler: ptan_ignite.EpisodeFPSHandler,
                 initial: int, batch_size: int):
        self.buffer = ptan.experience.ExperienceReplayBuffer(
            experience_source=None, buffer_size=buffer_size)
        self.exp_queue = exp_queue
        # being used for logging to record fps and timestamps
        self.fps_handler = fps_handler
        self.initial = initial
        self.batch_size = batch_size
        self._rewards_steps = []
        self.epsilon = None

    # pop all rewards ending steps for logging
    def pop_rewards_steps(self) -> tt.List[tt.Tuple[float, int]]:
        res = list(self._rewards_steps)
        self._rewards_steps.clear()
        return res

    def __iter__(self):
        while True:
            # drain the experience queue
            while self.exp_queue.qsize() > 0:
                exp = self.exp_queue.get()
                # if it's an episode ended, store the reward and steps
                if isinstance(exp, EpisodeEnded):
                    self._rewards_steps.append((exp.reward, exp.steps))
                    self.epsilon = exp.epsilon
                # else it's a normal experience, add to the buffer
                else:
                    self.buffer._add(exp)
                    self.fps_handler.step()
            # only yield batches when we have enough experience
            if len(self.buffer) < self.initial:
                continue
            # sample and yield a batch
            # the experience in the buffer is being used for training
            yield self.buffer.sample(self.batch_size)


if __name__ == "__main__":
    # get rid of missing metrics warning
    warnings.simplefilter("ignore", category=UserWarning)
    mp.set_start_method('spawn')

    random.seed(common.SEED)
    torch.manual_seed(common.SEED)
    params = common.GAME_PARAMS['pong']
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dev", default="cpu", help="Device to use, default=cpu")
    args = parser.parse_args()
    device = torch.device(args.dev)

    env = gym.make(params.env_name)
    env = ptan.common.wrappers.wrap_dqn(env)

    net = dqn_model.DQN(env.observation_space.shape,
                        env.action_space.n).to(device)

    tgt_net = ptan.agent.TargetNet(net)
    optimizer = optim.Adam(net.parameters(), lr=params.learning_rate)

    # start subprocess and experience queue

    # create shared experience queue between processes
    exp_queue = mp.Queue(maxsize=2)
    proc_args = (params, net, args.dev, exp_queue)

    # create a child process to run play_func
    play_proc = mp.Process(target=play_func, args=proc_args)
    play_proc.start()
    fps_handler = ptan_ignite.EpisodeFPSHandler()
    batch_generator = BatchGenerator(
        params.replay_size, exp_queue, fps_handler,
        params.replay_initial, params.batch_size)

    # do the backward pass and optimization step
    def process_batch(engine, batch):
        optimizer.zero_grad()
        loss_v = common.calc_loss_dqn(batch, net, tgt_net.target_model,
                                      gamma=params.gamma, device=device)
        loss_v.backward()
        optimizer.step()
        if engine.state.iteration % params.target_net_sync == 0:
            tgt_net.sync()
        return {
            "loss": loss_v.item(),
            "epsilon": batch_generator.epsilon,
        }

    engine = Engine(process_batch)
    ptan_ignite.EndOfEpisodeHandler(batch_generator, bound_avg_reward=18.0).attach(engine)
    ptan_ignite.EpisodeFPSHandler().attach(engine)

    @engine.on(ptan_ignite.EpisodeEvents.EPISODE_COMPLETED)
    def episode_completed(trainer: Engine):
        print("Episode %d: reward=%s, steps=%s, speed=%.3f frames/s, elapsed=%s" % (
            trainer.state.episode, trainer.state.episode_reward,
            trainer.state.episode_steps, trainer.state.metrics.get('avg_fps', 0),
            timedelta(seconds=trainer.state.metrics.get('time_passed', 0))))
        trainer.should_terminate = trainer.state.episode > 700

    @engine.on(ptan_ignite.EpisodeEvents.BOUND_REWARD_REACHED)
    def game_solved(trainer: Engine):
        print("Game solved in %s, after %d episodes and %d iterations!" % (
            timedelta(seconds=trainer.state.metrics['time_passed']),
            trainer.state.episode, trainer.state.iteration))
        trainer.should_terminate = True

    logdir = f"runs/{datetime.now().isoformat(timespec='minutes').replace(':', '-')}-{params.run_name}-{NAME}"
    tb = tb_logger.TensorboardLogger(log_dir=logdir)
    RunningAverage(output_transform=lambda v: v['loss']).attach(engine, "avg_loss")

    episode_handler = tb_logger.OutputHandler(tag="episodes", metric_names=['reward', 'steps', 'avg_reward'])
    tb.attach(engine, log_handler=episode_handler, event_name=ptan_ignite.EpisodeEvents.EPISODE_COMPLETED)

    # write to tensorboard every 100 iterations
    ptan_ignite.PeriodicEvents().attach(engine)
    handler = tb_logger.OutputHandler(tag="train", metric_names=['avg_loss', 'avg_fps'],
                                      output_transform=lambda a: a)
    tb.attach(engine, log_handler=handler, event_name=ptan_ignite.PeriodEvents.ITERS_100_COMPLETED)

    try:
        # start the engine
        engine.run(batch_generator)
    finally:
        play_proc.kill()
        play_proc.join()