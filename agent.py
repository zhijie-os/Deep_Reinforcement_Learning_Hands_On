import random
from environment import Environment
class Agent:
    def __init__(self):
        self.total_reward = 0
    
    def step(self, env: Environment):
        current_obs = env.get_observation()
        actions = env.get_actions()
        reward = env.action()
        self.total_reward += reward
    
if __name__ == "__main__":
    env = Environment()
    agent = Agent()
    while not env.is_done():
        agent.step(env)
    print("Total reward got: %.4f", agent.total_reward)
