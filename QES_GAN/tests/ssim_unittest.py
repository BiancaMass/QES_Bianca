import unittest
import numpy as np
import torch
from skimage.metrics import structural_similarity
from QuantumEvolutionaryAlgorithms.QES_GAN.utils.image_based_fitness_functions import calculate_ssim


class TestCalculateSSIM(unittest.TestCase):
    def setUp(self):
        # Single image setup
        self.single_image_np = np.random.rand(48, 48).astype(np.float32)
        self.single_image_tensor = torch.from_numpy(self.single_image_np)

        # Batch setup
        self.batch_image_np = np.random.rand(8, 48, 48).astype(np.float32)  # Batch of 8 images
        self.batch_image_tensor = torch.from_numpy(self.batch_image_np)

        # Setup for correctness tests
        self.image = np.random.rand(256, 256).astype(np.float32)  # Single random image
        self.negative_image = 1 - self.image  # Negative of the image for contrast test
        self.identical_batch = np.stack([self.image for _ in range(4)])  # Batch of identical images
        self.mixed_batch = np.stack([self.image if i % 2 == 0 else self.negative_image for i in
                                     range(4)])  # Batch with alternating images and their negatives

    def test_ssim_single_image_numpy(self):
        """Test SSIM calculation for single images with numpy arrays."""
        ssim_value = calculate_ssim(self.single_image_np, self.single_image_np, batch=False)
        self.assertIsInstance(ssim_value, float)

    def test_ssim_single_image_tensor(self):
        """Test SSIM calculation for single images with PyTorch tensors."""
        ssim_value = calculate_ssim(self.single_image_tensor, self.single_image_tensor, batch=False)
        self.assertIsInstance(ssim_value, float)

    def test_ssim_batch_numpy(self):
        """Test SSIM calculation for a batch of images with numpy arrays."""
        ssim_value = calculate_ssim(self.batch_image_np, self.batch_image_np, batch=True)
        self.assertIsInstance(ssim_value, float)

    def test_ssim_batch_tensor(self):
        """Test SSIM calculation for a batch of images with PyTorch tensors."""
        ssim_value = calculate_ssim(self.batch_image_tensor, self.batch_image_tensor, batch=True)
        self.assertIsInstance(ssim_value, float)

    def test_ssim_empty_images(self):
        """Test SSIM calculation raises an error for empty images."""
        empty_np = np.array([])
        with self.assertRaises(ValueError):
            calculate_ssim(empty_np, empty_np, batch=False)

    def test_ssim_mismatched_dimensions(self):
        """Test SSIM calculation raises an error for images with mismatched dimensions."""
        with self.assertRaises(ValueError):
            calculate_ssim(self.single_image_np, np.random.rand(128, 128).astype(np.float32), batch=False)

    def test_ssim_correctness_single_image(self):
        """Test SSIM is 1 for identical single images."""
        ssim_value = calculate_ssim(self.image, self.image, batch=False)
        self.assertEqual(ssim_value, 1)

    def test_ssim_correctness_single_image_contrast(self):
        """Test SSIM is less than 1 for contrasting single images."""
        ssim_value = calculate_ssim(self.image, self.negative_image, batch=False)
        self.assertLess(ssim_value, 1)

    def test_ssim_correctness_batch_identical(self):
        """Test SSIM is 1 for a batch of identical images."""
        ssim_value = calculate_ssim(self.identical_batch, self.identical_batch, batch=True)
        self.assertEqual(ssim_value, 1)

    def test_ssim_correctness_batch_mixed(self):
        """Test SSIM is less than 1 for a batch of alternating images and their negatives."""
        ssim_value = calculate_ssim(self.identical_batch, self.mixed_batch, batch=True)
        # Assuming at least one pair of contrasting images yields SSIM < 1
        self.assertLess(ssim_value, 1)


if __name__ == '__main__':
    unittest.main()
