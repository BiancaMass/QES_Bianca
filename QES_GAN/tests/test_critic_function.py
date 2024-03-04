"""
Only for the CLASSICAL GENERATOR AND CRITIC from WGAN-GP.
"""

import os
import torch
from statistics import mean
from torch.utils.data import DataLoader
from QuantumEvolutionaryAlgorithms.QES_GAN.utils.dataset import load_mnist, select_from_dataset
from QuantumEvolutionaryAlgorithms.QES_GAN.networks.CGCC import ClassicalGAN
from QuantumEvolutionaryAlgorithms.QES_GAN.utils.plotting import plot_image_tensor


def main(critic_path:str, generator_path:str):
    """
    Loads a pre-trained critic and generator networks, and output critic scores for real images,
    generated images, and random noise.
    Only works for the classical generator and critic WGAN-GP at the moment.

    :param critic_path: str. Path to the .pt file containing the pre-trained critic net.
    :param generator_path: str.  Path to the .pt file containing the pre-trained generator net.

    :returns: None.
    """

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load pre-trained critic
    image_size = 28
    image_shape = (image_size, image_size)
    latent_size = 100
    classes = [0, 1]
    batch_size = 64

    # Load the pre-trained critic
    critic_net = ClassicalGAN.ClassicalCritic(image_shape=image_shape)
    critic_net.load_state_dict(torch.load(critic_path, map_location=device))

    # Load the pre-trained generator
    generator_net = ClassicalGAN.ClassicalGenerator(latent_size=latent_size, image_shape=image_shape)
    generator_net.load_state_dict(torch.load(generator_path, map_location=torch.device('cpu')))

    # Load nets to device
    critic_net = critic_net.to(device)
    generator_net = generator_net.to(device)

    # Loading the dataset
    os.chdir(os.path.dirname(os.path.abspath(__file__)))  # making sure the dir path is right
    dataset = select_from_dataset(load_mnist(image_size=image_size), 1000, classes)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=1)

    avg_validities_real = []
    avg_validities_generated = []
    avg_validities_noise = []

    for i, (real_images, _) in enumerate(dataloader):
        real_images = real_images.to(device)  # batch of 64

        # Real images
        real_validity = critic_net(real_images)
        avg_validities_real.append(float(torch.mean(real_validity)))

        # Generated images
        latent_input = torch.randn(len(real_images), latent_size, device=device)
        generated_images = generator_net(latent_input)
        generated_validity = critic_net(generated_images)
        avg_validities_generated.append(float(torch.mean(generated_validity)))

        # Random noise
        noise = torch.randn(len(real_images), 28, 28, device=device)
        noise_validity = critic_net(noise)
        avg_validities_noise.append(float(torch.mean(noise_validity)))

        # If you want to plot tge images:
        # plot_image_tensor(torch.squeeze(real_images), 4, 8)
        # plot_image_tensor(generated_images.detach().numpy(), 4, 8)
        # plot_image_tensor(noise.detach().numpy(), 4, 8)

    validity_real = mean(avg_validities_real)
    validity_generated = mean(avg_validities_generated)
    validity_noise = mean(avg_validities_noise)

    print(f'Average validity for real images: {validity_real}')
    print(f'Average validity for generated images: {validity_generated}')
    print(f'Average validity for random noise: {validity_noise}')


if __name__ == '__main__':
    input_folder_path = '/Users/bmassacci/main_folder/maastricht/academics/quantum_thesis' \
                        '/scripts/QES-Bianca/QuantumEvolutionaryAlgorithms/QES_GAN/input/'
    path_to_critic = input_folder_path + "critic_300_classic.pt"
    path_to_generator = input_folder_path + "generator_300_classic.pt"
    main(critic_path=path_to_critic, generator_path=path_to_generator)

