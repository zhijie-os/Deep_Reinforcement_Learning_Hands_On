import torch
import torch.nn as nn


class DQN(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(DQN, self).__init__()

        # output size of a convolutional layer = (size - kernel_size) / stride + 1 
        self.conv = nn.Sequential(
            # input_shape[0] # four stacked frames, 4 frames show movement
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4), # detect edges, simple patterns 
            nn.ReLU(),  # used ReLU for nonlinearity
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            # detect shapes, object part
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            # detect complex objects, game elements
            nn.ReLU(), # Simple, prevents vanishing gradients, computationally efficient
            nn.Flatten(),
        )
        size = self.conv(torch.zeros(1, *input_shape)).size()[-1]
        # FC Layers as "Decision Makers"
        self.fc = nn.Sequential(
            nn.Linear(size, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        ) #

    def forward(self, x: torch.ByteTensor):
        # scale on GPU
        xx = x / 255.0
        return self.fc(self.conv(xx))
    # The output of the model is Q-values for every action available in the environment