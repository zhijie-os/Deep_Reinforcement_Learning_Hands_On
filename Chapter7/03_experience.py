# experience source classes take the agent instance and environment 
# and provide you with step-by-step data from trajectories

'''
    1. support for multiple environment being communicated at the same time
    2. a trajectory can be preprocessed and presented in a convienent from further training
    3. support for vectorized environments from gymnasium
'''

'''
1. ExperienceSource: Using the agent and the set of environments, it produces n-step subtrajectories with all intermediate steps

2. ExperienceSourceFirstLast: This is same as ExperienceSource, but instead of 
full subtrajectory (with all steps), it keeps only the first and the last steps,
with proper reward accumulation in between. The can save a lot of memory in the 
case of n-step DQN on advantage actor-critic (A2C) rollouts

3. ExperienceSourceRollouts: This follows the asynchronous advantage actor-critic(A3C)
rollouts scheme described in Mnih's paper about Atari games (we will discuss this topic in Chapter12)

'''

from lib import *
if __name__ == "__main__":
    env = ToyEnv()
    s, _ = env.reset()
    print(f"env.reset() -> {s}")
    s = env.step(1)
    print(f"env.step(1) -> {s}")
    s = env.step(2)
    print(f"env.step(2) -> {s}")

    for _ in range(10):
        r = env.step(0)
        print(r)

    agent = DullAgent(action=1)
    print("agent:", agent([1, 2])[0])

    env = ToyEnv()
    agent = DullAgent(action=1)
    exp_source = ptan.experience.ExperienceSource(
        env=env, agent=agent, steps_count=2)
    for idx, exp in enumerate(exp_source):
        if idx > 15:
            break
        print(exp)

    exp_source = ptan.experience.ExperienceSource(
        env=env, agent=agent, steps_count=4)
    print(next(iter(exp_source)))

    '''
    1. reset() was called in the environment to get the initial state.
    2. The agent was asked to select the action to execute from the state returned.
    3. The step() method was executed to get the reward and the next state.
    4. This next state was passed to the agent for the next action.
    5. Information about the transition from one state to the next state was returned.
    6. If the environment returns the end-of-episode flag, we emit the rest of the trajectory and reset the
    environment to start over.
    7. The process continues (from step 3) during the iteration over the experience source.
    '''

    # when you pass a list of environments, they are used in round-robin fashion.
    exp_source = ptan.experience.ExperienceSource(env=[ToyEnv(), ToyEnv()], agent=agent, steps_count=2)
    for idx, exp in enumerate(exp_source):
        if idx > 4:
            break
        print(exp)

    print("ExperienceSourceFirstLast")

    # returns state, action, reward, last_state
    exp_source = ptan.experience.ExperienceSourceFirstLast(
        env, agent, gamma=1.0, steps_count=1)
    for idx, exp in enumerate(exp_source):
        print(exp)
        if idx > 10:
            break