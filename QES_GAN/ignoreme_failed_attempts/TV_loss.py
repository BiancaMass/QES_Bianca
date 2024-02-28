"""
Total Variation Loss
From https://www.sciencedirect.com/science/article/pii/016727899290242F?via%3Dihub
Nonlinear total variation based noise removal algorithms, Rudin et al., 1992.
Found in Wikipedia, https://en.wikipedia.org/wiki/Total_variation_denoising, 2D images.
This NOT a differentiable metric.
"""

import torch
import os
import numpy as np

from dataset import select_from_dataset, load_mnist
from QuantumEvolutionaryAlgorithms.QES_GAN.configs import training_config
from torch.utils.data import DataLoader


def total_variation_loss(img):
    """
    Compute the Total Variation Loss according to the specified formula:
    V(y) = sum of sqrt((y[i+1, j] - y[i, j])^2 + (y[i, j+1] - y[i, j])^2)

    :param img: Tensor of shape (N, C, H, W) where
                N is the batch size,
                C is the number of channels,
                H is the height of the image, and
                W is the width of the image.
    :return: Total Variation Loss.
    """
    # Calculate the squared difference of neighboring pixel-values
    # The squared variation in the horizontal direction (ignoring the last column)
    pixel_dif1_squared = (img[:, :, 1:, :-1] - img[:, :, :-1, :-1]) ** 2
    # The squared variation in the vertical direction (ignoring the last row)
    pixel_dif2_squared = (img[:, :, :-1, 1:] - img[:, :, :-1, :-1]) ** 2

    # Calculate the sqrt of the sum of squared differences and sum them up
    loss = torch.sum(torch.sqrt(pixel_dif1_squared + pixel_dif2_squared))

    return loss


def main():
    # Test with simple generated images
    random_image = torch.randn(1, 1, 28, 28)
    zeros_image = torch.zeros(1, 1, 28, 28)
    ones_image = torch.ones(1, 1, 28, 28)

    # Compute the total variation loss
    tv_loss_1 = total_variation_loss(random_image)
    tv_loss_2 = total_variation_loss(zeros_image)
    tv_loss_3 = total_variation_loss(ones_image)

    print(f"Total Variation Loss random noise image: {tv_loss_1.item()}")
    print(f"Total Variation Loss all zeros image: {tv_loss_2.item()}")
    print(f"Total Variation Loss all ones image: {tv_loss_3.item()}")

    # Test with REAL MNIST images
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # loading the dataset
    os.chdir(os.path.dirname(os.path.abspath(__file__)))  # setting path to dir of this file
    dataset = select_from_dataset(dataset=load_mnist(image_size=training_config.IMAGE_SIDE),
                                  per_class_size=320,
                                  labels=training_config.CLASSES)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=1)

    real_tv_losses = []
    for i, (real_images, _) in enumerate(dataloader):
        real_images = real_images.to(device)
        tv_loss_mnist = total_variation_loss(real_images)
        print(f'Total Variation Loss for Real Images Batch {i}: {tv_loss_mnist} \n')
        real_tv_losses.append(tv_loss_mnist)

    mean_real_tv_loss = np.mean(real_tv_losses)
    print(f'Mean Total Variation Loss across all batches: {mean_real_tv_loss}')


if __name__ == '__main__':
    main()