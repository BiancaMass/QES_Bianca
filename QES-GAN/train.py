"""
This script will have to:
1. Determine the initial parameters, for example by calling them from a config file
2. Initiate the Critic and Generator as objects first
3. Start a training loop of as many iterations as you want to evolve, for the evolutionary
algorithm. Inside it:
    - Change the generator into its children
    - Possibly train the children for a few iterations
    - Evaluate the children based on some quality measure that you still have to decide (either a
      pre-trained critic or a image quality measure)
    - Create a new generation of children based on the results of the previous ones
        - If children are good, make them the new best individuals and filiate from there
        - If children are worse than parent, go back to parent

4. Return the best ansatz and properly train it.

Optional:
If ansatz after training is actually not so good, restart the process again. Determine a maximum
number of times the process runs.
"""

import os
import argparse
import math
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from torchvision.utils import save_image

from utils.dataset import load_mnist, select_from_dataset
# from utils.wgan import compute_gradient_penalty
from networks.critic import ClassicalCritic
from networks.generator import Q_Generator
import configs.training_config as training_config

image_size = training_config.IMAGE_SIZE
classes = training_config.CLASSES

dataset = select_from_dataset(load_mnist(image_size=image_size), 1000, classes)

def train(dataset, n_data_qubits, n_garbage_qubits, batch_size, output_dir):
    # Note: for now only implementing the GAN, not the generator

    # TODO: implement a 'if gpu available then use it' statement
    device = torch.device("cpu")  #  Note: you can change to GPU if available

    n_epochs = training_config.N_EPOCHS

    n_qubits = n_data_qubits + n_garbage_qubits
    latent_vec = torch.rand(batch_size, n_qubits, device=device)

    os.makedirs(output_dir, exist_ok=False)

    critic = ClassicalCritic(image_shape=(training_config.IMAGE_SIZE, training_config.IMAGE_SIZE))
    generator = Q_Generator(n_qubits=n_qubits, latent_vector=latent_vec)

    pass





