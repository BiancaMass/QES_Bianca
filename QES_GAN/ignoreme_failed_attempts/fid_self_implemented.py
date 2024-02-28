"""
Code based on
https://machinelearningmastery.com/how-to-implement-the-frechet-inception-distance-fid-from-scratch/

ISSUE: it gives ridiculously high values (even for the exact same data! I think there is an issue
in converting numbers from imaginary to real, but have not explored it fully.
"""


import tensorflow as tf
import torch
import numpy
from numpy import cov
from numpy import trace
from numpy import iscomplexobj
from numpy.random import random
from scipy.linalg import sqrtm


# Load the inception model
def load_inception_model(input_shape=(299, 299, 3)):
    model = tf.keras.applications.InceptionV3(include_top=False, pooling='avg',
                                              input_shape=input_shape)
    return model


# Preprocessing images
def preprocess_images(images, target_size=(299, 299)):
    # Convert PyTorch tensor to numpy if it's not already
    if isinstance(images, torch.Tensor):
        images = images.numpy()

    # Ensure images are in the format (batch_size, height, width, channels)
    images = numpy.transpose(images, (0, 2, 3, 1))

    # Resize images to 299x299
    images_resized = tf.image.resize(images, [299, 299])

    # Preprocess the images using the preprocessing function for InceptionV3
    images_preprocessed = tf.keras.applications.inception_v3.preprocess_input(images_resized)

    return images_preprocessed


# Calculate feature vectors
def calculate_features(preprocessed_images, model):
    features = model(preprocessed_images)
    return features


# Compute FID Score
def calculate_fid(real_features, gen_features):
    # Calculate the mean and covariance of both sets
    mu1, sigma1 = real_features.mean(axis=0), cov(real_features, rowvar=False)
    mu2, sigma2 = gen_features.mean(axis=0), cov(gen_features, rowvar=False)

    # Calculate the squared difference in means
    ssdiff = numpy.sum((mu1 - mu2) ** 2.0)

    # Calculate the sqrt of the product of covariances
    covmean = sqrtm(sigma1.dot(sigma2))

    # Control for imaginary numbers
    if iscomplexobj(covmean):
        covmean = covmean.real

    # Calculate FID
    fid = ssdiff + trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid
