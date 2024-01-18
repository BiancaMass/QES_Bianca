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

import configs.training_config as training_config
from utils.dataset import load_mnist, select_from_dataset
from networks.critic import ClassicalCritic
from networks.generator import Q_Generator
from utils.gradient_penalty import compute_gradient_penalty


def train(dataset, n_data_qubits, n_garbage_qubits, output_dir):
    """

    :param dataset:
    :param n_data_qubits: integer.
    :param n_garbage_qubits: integer.
    :param output_dir: str.

    :return:
    """
    # Note: for now only implementing the GAN, not the evolutionary algorithm

    os.makedirs(output_dir, exist_ok=False)

    # TODO: implement a 'if gpu available then use it' statement
    device = torch.device("cpu")  #  Note: you can change to GPU if available

    n_epochs = training_config.N_EPOCHS
    batch_size = training_config.BATCH_SIZE
    n_qubits = n_data_qubits + n_garbage_qubits

    # DataLoader from Pytorch to efficiently load and iterate over batches from the given dataset.
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=1)
    latent_vec = torch.rand(batch_size, n_qubits, device=device)

    # Note: critic should be pre-trained
    critic = ClassicalCritic(image_shape=(training_config.IMAGE_SIZE, training_config.IMAGE_SIZE))
    generator = Q_Generator(n_qubits=n_qubits, latent_vector=latent_vec)

    critic = critic.to(device)
    generator = generator.to(device)

    # Initialize an Adam optimizer for the generator.
    #  TODO: now this gives an error because the generator does not have parameters
    #   either remove the generator training for now or (more complicated) add the parameter
    #   function for its training. I'd recommend removing the training for now.
    optimizer_G = Adam(generator.parameters(),
                       lr=training_config.LR_G,
                       betas=(training_config.B1,
                              training_config.B2))

    # Load model checkpoint for the discriminator (i.e. a pretrained discriminator)
    # TODO: make sure this file exists
    critic.load_state_dict(torch.load(output_dir + f"/critic.pt"))

    for i, (real_images, _) in enumerate(dataloader):
        # Move real images to the specified device.
        real_images = real_images.to(device)
        # latent vector from uniform distribution
        latent_vec = torch.rand(batch_size, n_qubits, device=device)

        # Give generator latent vector z to generate images
        fake_images = generator(latent_vec)  # TODO: generator wont work now

        # Compute the critic's predictions for real and fake images.
        real_validity = critic(real_images)  # Real images.
        fake_validity = critic(fake_images)  # Fake images.

        # Calculate the gradient penalty and adversarial loss.
        gradient_penalty = compute_gradient_penalty(critic, real_images, fake_images, device)
        d_loss = -torch.mean(real_validity) + torch.mean(
            fake_validity) + training_config.LAMBDA_GP * gradient_penalty

        # Calculate Wasserstein distance - this is the score of how good the generation was
        wasserstein_distance = torch.mean(real_validity) - torch.mean(fake_validity)

        # TODO: this would be the time to train the generator before the evolution routine

        pass






if __name__ == "__main__":
    image_size = training_config.IMAGE_SIZE
    classes = training_config.CLASSES
    dataset = select_from_dataset(load_mnist(image_size=image_size), 1000, classes)
    train(dataset, n_data_qubits=training_config.IMAGE_SIZE, n_garbage_qubits=0,
          output_dir=training_config.OUTPUT_DIR)