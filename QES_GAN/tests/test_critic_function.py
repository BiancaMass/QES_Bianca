import os
import torch
from statistics import mean
from torch.utils.data import DataLoader
from QuantumEvolutionaryAlgorithms.QES_GAN.utils.dataset import load_mnist, select_from_dataset
from QuantumEvolutionaryAlgorithms.QES_GAN.networks.critic import ClassicalCritic
from QuantumEvolutionaryAlgorithms.QES_GAN.utils.plotting import plot_image_tensor


def main():
    image_size = 28
    classes = [0, 1]
    batch_size = 32
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    critic_net = ClassicalCritic(image_shape=(28, 28))
    critic_net = critic_net.to(device)
    critic_net.load_state_dict(torch.load(
        '/Users/bmassacci/main_folder/maastricht/academics/quantum_thesis/scripts/QES-Bianca'
        '/QuantumEvolutionaryAlgorithms/QES_GAN/output' + f"/critic-510.pt"))  # Note: hardcoded

    # loading the dataset
    os.chdir(os.path.dirname(os.path.abspath(__file__)))  # making sure the dir path is right
    dataset = select_from_dataset(load_mnist(image_size=image_size), 1000, classes)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=1)

    avg_validities = []

    for i, (real_images, _) in enumerate(dataloader):
        real_images = real_images.to(device)
        real_validity = critic_net(real_images)  # Real images.
        avg_validity = float(torch.mean(real_validity))
        avg_validities.append(avg_validity)
        # If you want to plot:
        # plot_image_tensor(torch.squeeze(real_images), 4, 8)

    validity = mean(avg_validities)
    print(f'Average validities across all images of all batches is: {validity}')


if __name__ == '__main__':
    main()
