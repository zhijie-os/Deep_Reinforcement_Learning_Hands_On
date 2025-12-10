import gymnasium as gym
import ptan
import typing as tt

import torch
import torch.optim as optim

from ignite.engine import Engine

from lib import dqn_model, common
import ale_py # This import is necessary for environments to register 

NAME = "02_n_steps"
DEFAULT_STEPS = 3

BEST_PONG = common.Hyperparams(
    env_name="PongNoFrameskip-v4",
    stop_reward=18.0,
    run_name="pong",
    replay_size=100000,
    replay_initial=10000,
    target_net_sync=1000,
    epsilon_frames=100000,
    epsilon_final=0.02,
    # learning_rate=9.932831968547505e-05,
    learning_rate=7.82e-05,
    gamma=0.98,
    episodes_to_solve=340,
)

n_steps = DEFAULT_STEPS
def train(params: common.Hyperparams,
        device: torch.device,
        _:dict) -> tt.Optional[int]:
    
    # make the parameterized environment
    env = gym.make(params.env_name)
    # wrap the environment for into ptan compatible env
    env = ptan.common.wrappers.wrap_dqn(env)

    # create the network, target network
    net = dqn_model.DQN(env.observation_space.shape, env.action_space.n).to(device)
    tgt_net = ptan.agent.TargetNet(net)

    # create epsilon greedy action selector 
    selector = ptan.actions.EpsilonGreedyActionSelector(epsilon=params.epsilon_start)
    # set epsilon tracker to update epsilon value during training
    epsilon_tracker = common.EpsilonTracker(selector, params)
    
    # create the DQN agent which wraps the network and the action selector
    agent = ptan.agent.DQNAgent(net, selector, device=device)

    # create experience source and replay buffer
    # this runs the environment and produces experience objects
    exp_source = ptan.experience.ExperienceSourceFirstLast(
        env, agent, gamma=params.gamma, env_seed=common.SEED, steps_count = n_steps) # enable the steps_count
    # create replay buffer
    # this stores those experience objects, keep history and return random minibatches for training
    buffer = ptan.experience.ExperienceReplayBuffer(
        exp_source, buffer_size=params.replay_size)
    # create the optmizer
    optimizer = optim.Adam(net.parameters(), lr=params.learning_rate)

    # this is a step for a batch of data
    def process_batch(engine, batch):
        optimizer.zero_grad() # stop gradient
        # calculate the loss
        loss_v = common.calc_loss_dqn(batch, net, tgt_net.target_model, gamma=params.gamma**n_steps, device=device)

        # apply the gradient
        loss_v.backward()
        optimizer.step() # update the network parameters
        epsilon_tracker.frame(engine.state.iteration)   # update the epsilon value
        # for every target_net_sync iterations, sync the target network with the current network
        if engine.state.iteration % params.target_net_sync == 0:
            tgt_net.sync()
        
        # return loss and epsilon
        return {
            "loss": loss_v.item(),
            "epsilon":selector.epsilon
        }
    
    # get the ignite engine
    engine = Engine(process_batch)

    # setup ignite engine with common handlers
    common.setup_ignite(engine, params, exp_source, NAME)
    # run the training loop with batch generator
    r = engine.run(common.batch_generator(buffer, params.replay_initial, params.batch_size))
    # the engine runs a given "process_function" over each batch of dataset, emitting events as it goes


    if r.solved:
        return r.episode
    
if __name__ == "__main__":
    args = common.argparser().parse_args()
    common.train_or_tune(args, train, BEST_PONG)
