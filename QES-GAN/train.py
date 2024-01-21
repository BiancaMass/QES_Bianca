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


def train_GAN(dataset, n_patches, n_data_qubits, n_garbage_qubits,
              output_dir):
    # TODO: this is not actually training so consider changing function name
    """

    :param dataset:
    :param n_data_qubits: integer.
    :param n_garbage_qubits: integer.
    :param output_dir: str.

    :return:
    """
    # Note: for now only implementing the GAN, not the evolutionary algorithm

    os.makedirs(output_dir, exist_ok=False)

    # Note: implement a 'if gpu available then use it' statement
    device = torch.device("cpu")  #  Note: you can change to GPU if available

    batch_size = training_config.BATCH_SIZE
    n_total_qubits = n_data_qubits + n_garbage_qubits
    n_patches = n_patches

    # DataLoader from Pytorch to efficiently load and iterate over batches from the given dataset.
    print("Loading Dataset")
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=1)

    # Note: critic should be pre-trained (or use another function to score the ansatz)
    critic = ClassicalCritic(image_shape=(training_config.IMAGE_SIDE, training_config.IMAGE_SIDE))
    generator = Q_Generator(batch_size=batch_size,
                            n_patches=n_patches,
                            n_data_qubits=n_data_qubits,
                            n_ancillas=n_garbage_qubits,
                            image_shape=(training_config.IMAGE_SIDE,training_config.IMAGE_SIDE),
                            pixels_per_patch=training_config.PIXELS_PER_PATCH)

    critic = critic.to(device)
    generator = generator.to(device)

    # Load model checkpoint for the discriminator (i.e. a pretrained discriminator)
    # Note: you can consider using another cost function and not a critic in the evolutionary
    #  process
    critic.load_state_dict(torch.load('./output/' + f"/critic-80.pt"))  # Note: hardcoded for dev.

    for i, (real_images, _) in enumerate(dataloader):
        # each real_images is a batch of images, a tensor of size
        # (batch_size, colorChannels, width, height)

        # Move real images to the specified device.
        real_images = real_images.to(device)
        # latent vector from uniform distribution
        # latent_vec = torch.rand(n_qubits, device=device)
        latent_vec = torch.rand(batch_size, n_total_qubits, device=device)

        # Generate fake_images by giving the a generator latent vector z
        fake_images = generator(latent_vec)

        # Compute the critic's predictions for real and fake images.
        real_validity = critic(real_images)  # Real images.
        fake_validity = critic(fake_images)  # Fake images.

        # Calculate the gradient penalty and adversarial loss.
        # TODO: error here
        gradient_penalty = compute_gradient_penalty(critic, real_images, fake_images, device)
        d_loss = -torch.mean(real_validity) + torch.mean(
            fake_validity) + training_config.LAMBDA_GP * gradient_penalty

        # Calculate Wasserstein distance - this is the score of how good the generation was
        wasserstein_distance = torch.mean(real_validity) - torch.mean(fake_validity)

        # Note: this would be the time to train the generator before the evolution routine

        pass






if __name__ == "__main__":
    image_size = training_config.IMAGE_SIDE
    classes = training_config.CLASSES
    dataset = select_from_dataset(load_mnist(image_size=image_size), 1000, classes)
    n_patches = training_config.N_PATCHES
    train_GAN(dataset,
              n_data_qubits=training_config.N_DATA_QUBITS,
              n_garbage_qubits=training_config.N_ANCILLAS,
              n_patches=n_patches,
              output_dir=training_config.OUTPUT_DIR)

