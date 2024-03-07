import scipy
from scipy import stats


import os
import torch
from statistics import mean
from torch.utils.data import DataLoader
from QuantumEvolutionaryAlgorithms.QES_GAN.utils.dataset import load_mnist, select_from_dataset
from QuantumEvolutionaryAlgorithms.QES_GAN.networks.CGCC import ClassicalGAN
from QuantumEvolutionaryAlgorithms.QES_GAN.utils.plotting import plot_image_tensor


def emd_test(generator_path:str):
    """
    This function tests the performance of scipy's earth mover distance (EMD) function for my use
    case. It calculates and prints the EMD for:
    - real images from the MNIST dataset and noise images
    - real images from the MNIST dataset and generated images by a quantum generator (pre-loaded from a .qasm file)
    - generated images by the quantum circuit and noise images
    - real images from the MNIST dataset and completely black images
    - real images from the MNIST dataset and completely white images

    :param generator_path: str.  Path to the .qasm file containing the pre-trained generator net.

    :returns: None.
    """

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load pre-trained critic
    image_size = 28
    image_shape = (image_size, image_size)
    latent_size = 100
    classes = [0, 1]
    batch_size = 200

    # Load the pre-trained generator
    generator_net = ClassicalGAN.ClassicalGenerator(latent_size=latent_size, image_shape=image_shape)
    generator_net.load_state_dict(torch.load(generator_path, map_location=torch.device('cpu')))

    generator_net = generator_net.to(device)

    # Loading the dataset
    os.chdir(os.path.dirname(os.path.abspath(__file__)))  # making sure the dir path is right
    dataset = select_from_dataset(load_mnist(image_size=image_size), 1000, classes)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=1)

    # Full black and full white images
    black_image = torch.zeros(1, 28, 28, device=device)
    white_image = torch.ones(1, 28, 28, device=device) * 255

    distances_real_noise = []
    distances_real_gen = []
    distances_gen_noise = []
    distances_real_black = []
    distances_real_white = []

    for i, (real_images, _) in enumerate(dataloader):
        real_images = real_images.to(device)  # batch of 64

        # Generated images
        latent_input = torch.randn(len(real_images), latent_size, device=device)
        generated_images = generator_net(latent_input)

        # Random noise
        noise = torch.randn(len(real_images), 28, 28, device=device)

        # Full black and full white
        black_images_flat = black_image.view(black_image.size(0), -1).repeat(real_images.size(0), 1)
        white_images_flat = white_image.view(white_image.size(0), -1).repeat(real_images.size(0), 1)

        # Flatten the tensors into 1D distributions
        generated_images_flat = generated_images.view(generated_images.size(0), -1)
        real_images_flat = real_images.view(real_images.size(0), -1)
        noise_images_flat = noise.view(noise.size(0), -1)

        # Convert the tensors to numpy arrays
        generated_images_flat_np = generated_images_flat.cpu().detach().numpy()
        real_images_flat_np = real_images_flat.cpu().detach().numpy()
        noise_images_flat_np = noise_images_flat.cpu().detach().numpy()
        black_images_flat_np = black_images_flat.cpu().detach().numpy()
        white_images_flat_np = white_images_flat.cpu().detach().numpy()

        distance_real_noise = scipy.stats.wasserstein_distance(real_images_flat_np.flatten(),
                                                               noise_images_flat_np.flatten())
        distances_real_noise.append(distance_real_noise)

        distance_real_gen = scipy.stats.wasserstein_distance(real_images_flat_np.flatten(),
                                                             generated_images_flat_np.flatten())
        distances_real_gen.append(distance_real_gen)

        distance_gen_noise = scipy.stats.wasserstein_distance(generated_images_flat_np.flatten(),
                                                    noise_images_flat_np.flatten())
        distances_gen_noise.append(distance_gen_noise)

        distance_real_black = scipy.stats.wasserstein_distance(real_images_flat_np.flatten(),
                                                               black_images_flat_np.flatten())
        distances_real_black.append(distance_real_black)

        distance_real_white = scipy.stats.wasserstein_distance(real_images_flat_np.flatten(),
                                                               white_images_flat_np.flatten())
        distances_real_white.append(distance_real_white)

        # If you want to plot tge images:
        # plot_image_tensor(generated_images.detach().numpy(), 4, 8)
        # plot_image_tensor(noise.detach().numpy(), 4, 8)

    distance_real_noise_avg = mean(distances_real_noise)
    distance_real_gen_avg = mean(distances_real_gen)
    distance_gen_noise_avg = mean(distances_gen_noise)
    distance_real_black_avg = mean(distances_real_black)
    distance_real_white_avg = mean(distances_real_white)

    print(f'Average EM-Distance b/w Real and Noise distribution: {distance_real_noise_avg}')
    print(f'Average EM-Distance b/w Real and Gen distribution: {distance_real_gen_avg}')
    print(f'Average EM-Distance b/w Gen and Noise distribution: {distance_gen_noise_avg}')
    print(f'Average EM-Distance b/w Real and Full Black distribution: {distance_real_black_avg}')
    print(f'Average EM-Distance b/w Real and Full White distribution: {distance_real_white_avg}')


if __name__ == '__main__':
    input_folder_path = '/Users/bmassacci/main_folder/maastricht/academics/quantum_thesis' \
                        '/scripts/QES-Bianca/QuantumEvolutionaryAlgorithms/QES_GAN/input/'
    path_to_generator = input_folder_path + "generator_300_classic.pt"
    emd_test(generator_path=path_to_generator)
