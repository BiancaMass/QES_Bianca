import numpy as np
import torch
import itertools
import torch.nn as nn
import torch.nn.functional as F

# TODO: consider giving it weights and parameters as input if you are going to use it pre-trained


class ClassicalCritic(nn.Module):
    def __init__(self, image_shape):
        super().__init__()
        self.image_shape = image_shape

        self.fc1 = nn.Linear(int(np.prod(self.image_shape)), 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, x):
        if type(x) == list:
            flat_list = list(itertools.chain(*itertools.chain(*x)))
            x = torch.tensor(flat_list).view(32, 28, 28).to(dtype=torch.float32)
        x = x.view(x.shape[0], -1)
        x = F.leaky_relu(self.fc1(x), 0.2)
        x = F.leaky_relu(self.fc2(x), 0.2)
        return self.fc3(x)

