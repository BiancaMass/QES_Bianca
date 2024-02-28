from collections import OrderedDict

"This works but it is way too expensive, it wont do it. And it needs at least 2048 images per " \
"batch to correctly comopute the covariance matrices."

import torch
import numpy as np
import tensorflow as tf
from pytorch_fid.inception import InceptionV3
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from ignite.engine import *
from ignite.metrics import *


def add_noise_to_images(images, noise_factor=0.5):
    # Adding random noise
    noisy_images = images + noise_factor * torch.randn_like(images)
    noisy_images = np.clip(noisy_images, 0., 1.)  # Ensure pixel values are in [0, 1]
    return noisy_images


# Preprocessing images
def preprocess_images(images, target_size=(299, 299)):
    # Convert PyTorch tensor to numpy if it's not already
    if isinstance(images, torch.Tensor):
        images = images.numpy()

    # Ensure images are in the format (batch_size, height, width, channels)
    images = np.transpose(images, (0, 2, 3, 1))

    # Resize images to 299x299
    images_resized = tf.image.resize(images, [299, 299])

    # Preprocess the images using the preprocessing function for InceptionV3
    images_preprocessed = tf.keras.applications.inception_v3.preprocess_input(images_resized)

    return images_preprocessed


def main():
    def eval_step(engine, batch):
        return batch

    default_evaluator = Engine(eval_step)

    # wrapper class as feature_extractor
    class WrapperInceptionV3(nn.Module):

        def __init__(self, fid_incv3):
            super().__init__()
            self.fid_incv3 = fid_incv3

        @torch.no_grad()
        def forward(self, x):
            y = self.fid_incv3(x)
            y = y[0]
            y = y[:, :, 0, 0]
            return y

    # use cpu rather than cuda to get comparable results
    device = "cpu"

    # pytorch_fid model
    dims = 2048
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
    model = InceptionV3([block_idx]).to(device)

    # wrapper model to pytorch_fid model
    wrapper_model = WrapperInceptionV3(model)
    wrapper_model.eval();

    # Configuration
    batch_size = 2048
    image_size = 28
    n_channels = 3
    needed_size = 299

    # comparable metric
    pytorch_fid_metric = FID(num_features=dims, feature_extractor=wrapper_model)
    pytorch_fid_metric.attach(default_evaluator, "fid")

    ######### MNIST #########
    # Define the transform for MNIST dataset
    transform = transforms.Compose([
        transforms.ToTensor(),  # Converts to [0, 1] range and adds channel dimension
    ])
    # Load MNIST dataset
    mnist_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    subset_indices = list(range(2048))  # Example: first 1000 samples
    mnist_subset = Subset(mnist_dataset, subset_indices)
    subset_loader = DataLoader(mnist_subset, batch_size=batch_size, shuffle=False)

    states = []
    for images, _ in subset_loader:
        # y_true = preprocess_images(images)
        # y_pred = y_true
        # y_pred = add_noise_to_images(y_true)
        images_preprocessed_tf = preprocess_images(images)  # images is a PyTorch tensor
        # y_pred = add_noise_to_images(y_true)
        images_preprocessed_np = images_preprocessed_tf.numpy()
        y_true = torch.from_numpy(images_preprocessed_np).float()

        y_true_rgb = y_true.repeat_interleave(3, dim=-1)
        y_true_rgb = y_true_rgb.permute(0, 3, 1, 2)

        y_pred_rgb = y_true_rgb

        state = default_evaluator.run([[y_pred_rgb, y_true_rgb]])
        states.append(float(state.metrics["fid"]))

    print(states, "\n")
    print(np.mean(states))


if __name__ == '__main__':
    main()
