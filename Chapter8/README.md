# DQN extensions

1. N-step DQN: improve convergence speed and stability with a simple unrolling of the Bellman equation

2. Double DQN: deal with DQN overstimation of the values of the actions
   1. Q(s_t, a_t) = r_t + beta*max_a Q(s_t+1, a_t+1)
   2. => Q(s_t, a_t) = r_t + beta*max_a'[r_a,t+1 +beta* Q(s_t+2, a')]
   3. if we assume the action a is chosen optimally,=> Q(s_t, a_t) = r_t + beta* r_t+1 + beta^2*max_a'Q(s_t+2, a')

3. Noisy networks: make exploration more efficient by adding noise to the network weights

4. Prioritized replay buffer: uniform sampling of our experience is not the best way to train

5. Dueling DQN: improve convergence speed by making our network's architecture more closely represent the problem that we are solving

6. Categorical DQN: go beyond the single expected value of the action and work with full distributions

