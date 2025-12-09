import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import warnings
import dataclasses
import ptan
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
        epsilon_frames = 100000, # decay in every 100 thousands frame
        epsilon_final = 0.02, # final epsilon value, the lowest exploration rate would 0.2
    ),
}

# this function unpacks batch of ExperienceFirstLast into separate arrays
def unpack_batch(batch: tt.List[ExperienceFirstLast]):
    # state: observation from the environment
    # action: integer action taken by the agent
    # reward: immediate reward if the step_count = 1; if the step_count is larger, we use discontinued sum of rewards
    # last state: If the transition corresponds to the final step in the environment, then the field is None; otherwise, it contains the last observation in the experience chain
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
    dones_v = torch.tensor(dones).to(device).bool()
    last_states_v = torch.tensor(last_states).to(device)

    # using the current net to get Q-values for all actions
    actions_v = actions_v.unsqueeze(-1) # make actions_v shape (batch_size, 1)
    state_action_vals = net(states_v).gather(1, actions_v)  # get Q-values for taken actions only
    state_action_vals = state_action_vals.squeeze(-1) # shape (batch_size,)

    # use the target network to get the max Q-value for the next state
    with torch.no_grad():
        next_state_values = tgt_net(last_states_v).max(1)[0]
        # PyTorch boolean indexing operation (works similarly to NumPy).
        next_state_values[dones_v] = 0.0
        expected_state_action_values = rewards_v + gamma * next_state_values

    # use MSE loss to minimize the difference between current Q-values(from current net) and expected Q-values (from target net)
    return nn.MSELoss()(state_action_vals, expected_state_action_values)

def calc_loss_double_dqn(batch: tt.List[ptan.experience.ExperienceFirstLast],
    net: nn.Module, tgt_net: nn.Module, gamma: float, device: torch.device):
    states, actions, rewards, dones, last_states = unpack_batch(batch)

    states_v = torch.tensor(states).to(device)
    actions_v = torch.tensor(actions).to(device)
    rewards_v = torch.tensor(rewards).to(device)
    dones_v = torch.tensor(dones).to(device).bool()
    last_states_v = torch.tensor(last_states).to(device)


    # using the current net to get Q-values for all actions
    actions_v = actions_v.unsqueeze(-1) # make actions_v shape (batch_size, 1)
    state_action_vals = net(states_v).gather(1, actions_v)  # get Q-values for taken actions only
    state_action_vals = state_action_vals.squeeze(-1) # shape (batch_size,)

    while torch.no_grad():
        next_states_v = torch.as_tensor(last_states).to(device)
        next_state_acts = net(next_states_v).max(1)[1] # using the current net to choose the action
        next_state_acts = next_state_acts.unsqueeze(-1)
        # using the target net to get the Q-value for the current net chosen action
        next_state_vals = tgt_net(next_states_v).gather(1, next_state_acts).squeeze(-1)
        next_state_vals[dones_v] = 0.0
        expected_state_action_values = rewards_v + gamma * next_state_vals.detach()
    
    return nn.MSELoss()(state_action_vals, expected_state_action_values)    # return the MSE loss value

# This function monitors the average Q-value estimates during training as a diagnostic metric
@torch.no_grad()
def calc_values_of_states(states: np.ndarray, net: nn.Module, device: torch.device):
    mean_vals = []
    for batch in np.array_split(states, 64):
        states_v = torch.tensor(batch).to(device)
        action_values_v = net(states_v)
        best_action_values_v = action_values_v.max(1)[0]
        mean_vals.append(best_action_values_v.mean().item())
    return np.mean(mean_vals)

class EpsilonTracker:
    def __init__(self, selector: EpsilonGreedyActionSelector, params: Hyperparams):
        self.selector = selector
        self.params = params
        self.frame(0)   # call the first epsilon udpate

    # eps decay some amount very epsilon_frames until it reachese epsilon_final
    def frame(self, frame_idx: int):
        eps = self.params.epsilon_start - frame_idx / self.params.epsilon_frames
        self.selector.epsilon = max(self.params.epsilon_final, eps)

#  this takes a replay buffer and yields batches of ExperienceFirstLast objects
def batch_generator(buffer: ExperienceReplayBuffer, initial: int, batch_size: int) \
    -> tt.Generator[tt.List[ExperienceFirstLast], None, None]:
    # initialize the buffer with some experiences first (initialis the number of experience needed)
    buffer.populate(initial)
    while True:
        # get one more experience to keep the buffer fresh
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
    # the EndOfEpisodeHandler emits the Ignite event very time a game episode ends
    handler = ptan_ignite.EndOfEpisodeHandler(
        exp_source, bound_avg_reward=params.stop_reward
    )
    handler.attach(engine)
    # the EpisodeFPSHandler, tracks the time the episode has taken and 
    # the amount of interactions that we have had with the environment.
    # We calculate frames per seconds (FPS) as a performance metric
    ptan_ignite.EpisodeFPSHandler().attach(engine)

    # ignite engine where the episode completed event is handled
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
    
    # game solved event handler
    @engine.on(ptan_ignite.EpisodeEvents.BOUND_REWARD_REACHED)
    def game_solve(trainer: Engine):
        passed = trainer.state.metrics['time_passed']
        print("Game solved in %s, after %d episodes and %d iterations!" % (
            timedelta(seconds=int(passed)),
            trainer.state.episode,
            trainer.state.iteration
        ))
        trainer.should_terminate = True
        trainer.state.solved = True

    # get the time
    now = datetime.now().isoformat(timespec="minutes").replace(":", " ")
    logdir = f"runs/{now} - {params.run_name} - {run_name}"
    tb = tb_logger.TensorboardLogger(log_dir=logdir)
    run_avg = RunningAverage(output_transform=lambda v:v["loss"])
    run_avg.attach(engine, "avg_loss")
    # get the avergae loss

    # attach the metrics to tensorboard logger
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

# hyperparams tuner
# the hyperparams dataclass defines the parameters needed for training

# TrainFunc is a type hint that defines the signature of the training functions used in hyperparameter tuning.
TrainFunc = tt.Callable[ # this is a callable function
    # takes three arguments 1. Hyperparams 2. torch.device 3. dict - the extra config dictionary
    [Hyperparams, torch.device, dict],
    tt.Optional[int]
    # either an integer or None is returned 
]

# BASE_SPACE defines the default search space for hyperparameter tuning
# it uses Ray Tune's search space definitions to specify ranges and distributions for hyperparameters
BASE_SPACE = {
    "learning_rate": tune.loguniform(1e-5, 1e-4),
    "gamma": tune.choice([0.9, 0.92, 0.95, 0.98, 0.99, 0.995]),
}

def tune_params(
        base_params: Hyperparams, train_func: TrainFunc, device: torch.device,
        samples: int = 10, extra_space: tt.Optional[tt.Dict[str, tt.Any]] = None,
):
    """
    Perform hyperparameters tune.
    :param train_func: Train function, has to return "episodes" key with metric
    :param device: torch device
    :param samples: count of samples to perform
    :param extra_space: additional search space
    """
    search_space = dict(BASE_SPACE)
    if extra_space is not None:
        search_space.update(extra_space)
    config = tune.TuneConfig(num_samples=samples)

    def objective(config: dict, device: torch.device) -> dict:
        keys = dataclasses.asdict(base_params).keys()
        upd = {"tuner_mode": True}
        for k, v in config.items():
            if k in keys:
                upd[k] = v
        params = dataclasses.replace(base_params, **upd)
        res = train_func(params, device, config)
        return {"episodes": res if res is not None else 10**6}

    obj = tune.with_parameters(objective, device=device)
    if device.type == "cuda":
        obj = tune.with_resources(obj, {"gpu": 1})
    tuner = tune.Tuner(obj, param_space=search_space, tune_config=config)
    results = tuner.fit()
    best = results.get_best_result(metric="episodes", mode="min")
    print(best.config)
    print(best.metrics)


def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dev", default="cpu", help="Device to use, default=cpu")
    parser.add_argument(
        "--params", choices=('common', 'best'), default="best",
        help="Params to use for training or tuning, default=best"
    )
    parser.add_argument(
        "--tune", type=int, help="Steps of params tune")
    return parser


def train_or_tune(
        args: argparse.Namespace,
        train_func: TrainFunc,
        best_params: Hyperparams,
        extra_params: tt.Optional[dict] = None,
        extra_space: tt.Optional[dict] = None,
):
    random.seed(SEED)
    torch.manual_seed(SEED)
    device = torch.device(args.dev)

    if args.params == "common":
        params = GAME_PARAMS['pong']
    else:
        params = best_params

    if extra_params is None:
        extra_params = {}
    if args.tune is None:
        train_func(params, device, extra_params)
    else:
        tune_params(params, train_func, device, samples=args.tune,
                    extra_space=extra_space)