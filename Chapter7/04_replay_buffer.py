'''
    1 ExperienceReplayBuffer: A simple replay buffer of a predefined size with uniform sampling.

    2 PrioReplayBufferNaive: A simple, but not very efficient, prioritized replay buffer implementation.
    The complexity of sampling is O(n), which might become an issue with large buffers. This version
    has the advantage over the optimized class, having much easier code. For medium-sized buffers the
    performance is still acceptable, so we will use it in some examples.

    3 PrioritizedReplayBuffer: Uses segment trees for sampling, which makes the code cryptic, but with
    O(log(n)) sampling complexity.
'''


from lib import *

if __name__ == "__main__":
    env = ToyEnv()
    agent = DullAgent(action=1)
    exp_source = ptan.experience.ExperienceSourceFirstLast(env, agent, gamma=1.0, steps_count=1)
    # create the replay buffer with size to 100
    buffer = ptan.experience.ExperienceReplayBuffer(exp_source, buffer_size = 100)

    for step in range(6):
        buffer.populate(1)

        if len(buffer) < 5:
            continue

        batch = buffer.sample(4)
        print("Train time, %d batch samples:" % len(batch))
        for s in batch:
            print(s)