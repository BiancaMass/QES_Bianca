"""
Code based on
https://machinelearningmastery.com/how-to-implement-the-frechet-inception-distance-fid-from-scratch/

ISSUE: it gives ridiculously high values (even for the exact same data! I think there is an issue
in converting numbers from imaginary to real, but have not explored it fully.
"""


import os
import numpy as np
import torch
import tensorflow as tf
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

# Assuming fid_self_implemented.py contains these functions
from fid_self_implemented import preprocess_images, calculate_features, calculate_fid

def add_noise_to_images(images, noise_factor=0.5):
    # Adding random noise
    noisy_images = images + noise_factor * torch.randn_like(images)
    noisy_images = np.clip(noisy_images, 0., 1.)  # Ensure pixel values are in [0, 1]
    return noisy_images


def compute_features(dataloader, model, add_noise=False):
    features_list = []
    with torch.no_grad():  # No need to compute gradients
        for batch in dataloader:
            images, _ = batch
            if add_noise:
                images = add_noise_to_images(images)  # Add noise for generated images
            images_rgb = images.repeat(1, 3, 1, 1)
            images_preprocessed = preprocess_images(images_rgb.numpy())  # Convert to numpy and preprocess
            features = calculate_features(images_preprocessed, model)
            features_list.append(features)
    return np.vstack(features_list)


def main():
    # Configuration
    batch_size = 64  # Adjust as needed
    image_size = 28  # MNIST images are 28x28

    # Transform to convert images to the required format
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # MNIST is single channel, so mean and std are for one channel
    ])

    # Load MNIST dataset
    mnist_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)

    subset_indices = list(range(1000))  # Example: first 1000 samples
    mnist_subset = Subset(mnist_dataset, subset_indices)

    # DataLoader for the real images
    dataloader_real = DataLoader(mnist_subset, batch_size=batch_size, shuffle=True, num_workers=1)

    model = tf.keras.applications.InceptionV3(include_top=False, pooling='avg', input_shape=(299, 299, 3))

    real_features = compute_features(dataloader_real, model, add_noise=False)
    gen_features = compute_features(dataloader_real, model, add_noise=True)

    fid_score = calculate_fid(real_features, gen_features)
    print("FID score with noise:", fid_score)

if __name__ == '__main__':
    main()
