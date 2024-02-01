import numpy as np
import torch
import itertools
import torch.nn as nn
import torch.nn.functional as F


class ClassicalCritic(nn.Module):
    def __init__(self, image_shape):
        """ Initialize the critic network (3-layers classical critic neural network).

        :param image_shape: tuple. The shape of the input images in the format (height, width).

        Attributes:
        image_shape (tuple): The shape of the input images.
        fc1 (nn.Linear): The first fully connected layer with 512 output units.
        fc2 (nn.Linear): The second fully connected layer with 256 output units.
        fc3 (nn.Linear): The third fully connected layer with 1 output unit (scalar).

        """
        super().__init__()
        self.image_shape = image_shape

        self.fc1 = nn.Linear(int(np.prod(self.image_shape)), 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, x):
        """ Forward pass of the classical critic network.

        :param x: tensor or list of tensors. Input images. It can be either a single image tensor
                  or a list of image tensors of length batch_size.
        :return: torch.Tensor. A tensor containing the critic scores for the input image,
                 of length batch size. So, one score per image that was given as input.
        """
        if type(x) == list:
            batch_size = len(x)
            flat_list = list(itertools.chain(*itertools.chain(*x)))
            x = torch.tensor(flat_list).view(batch_size, self.image_shape[0],
                                             self.image_shape[1]).to(dtype=torch.float32)
        else:
            x = x.to(dtype=torch.float32)

        x = x.view(x.shape[0], -1)

        x = F.leaky_relu(self.fc1(x), 0.2)
        x = F.leaky_relu(self.fc2(x), 0.2)
        return self.fc3(x)

