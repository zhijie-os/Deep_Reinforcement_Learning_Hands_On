# Policy Gradient

1. No explicit exploration is needed: At the beginning, the network is initialized with random weights, and it returns a uniform probability distribution. This corresponds to random agent behavior.
2. No replay buffer is needed: Policy gradient methods belong to the on-policy methods class, which means that we can't train on data obtained from the old policy. Good: the such methods converge faster. Bad: require much more interactions with the environment than off-policy methods such DQN.
3. No target network is needed: 
   