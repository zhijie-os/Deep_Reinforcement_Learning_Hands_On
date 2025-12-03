# target net class is for addressing the boostrapping problem
# in DQN, we use Bellman equation to update Q-network
# Q(s, a) ← r + γ * max_a' Q(s', a')

# if we use the same network for both Q(s, a) and Q(s', a')
# we are moving the target, the network is constantly changing, the target is shifting

# solution: target network = Online Network + Target Network
# online network: actively trained network
# target network: stable network used for target value computation

# The target network's weights are periodically synced from the online network.


from lib import *


if __name__ == "__main__":
    net = DQNNet()
    print(net)
    tgt_net = ptan.agent.TargetNet(net)
    print("Main net:", net.ff.weight)
    print("Target net:", tgt_net.target_model.ff.weight)

    net.ff.weight.data += 1.0
    print("After update")
    print("Main net:", net.ff.weight)
    print("Target net:", tgt_net.target_model.ff.weight)

    
    tgt_net.sync()
    print("After sync")
    print("Main net:", net.ff.weight)
    print("Target net:", tgt_net.target_model.ff.weight)