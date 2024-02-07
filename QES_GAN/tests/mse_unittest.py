import unittest
import numpy as np
import torch
from QuantumEvolutionaryAlgorithms.QES_GAN.utils.image_based_fitness_functions import calculate_mse


class TestCalculateMSE(unittest.TestCase):
    def test_black_images(self):
        """Test MSE calculation for two black images."""
        black_image1 = np.zeros((48, 48, 3), dtype=np.float32)
        black_image2 = np.zeros((48, 48, 3), dtype=np.float32)
        mse = calculate_mse(black_image1, black_image2)
        self.assertEqual(mse, 0.0)

    def test_white_images(self):
        """Test MSE calculation for two white images."""
        white_image1 = np.ones((48, 48, 3), dtype=np.float32) * 255
        white_image2 = np.ones((48, 48, 3), dtype=np.float32) * 255
        mse = calculate_mse(white_image1, white_image2)
        self.assertEqual(mse, 0.0)

    def test_grayscale_noise_images(self):
        """Test MSE calculation for two images with grayscale noise."""
        noise_image1 = np.random.rand(48, 48, 3).astype(np.float32)
        noise_image2 = noise_image1 + np.random.normal(0, 0.1, (48, 48, 3)).astype(np.float32)
        mse = calculate_mse(noise_image1, noise_image2)
        self.assertGreater(mse, 0.0)

    def test_identical_images(self):
        """Test MSE calculation for two identical images."""
        identical_image1 = np.random.rand(48, 48, 3).astype(np.float32)
        identical_image2 = identical_image1.copy()
        mse = calculate_mse(identical_image1, identical_image2)
        self.assertEqual(mse, 0.0)

    def test_images_as_tensors(self):
        """Test MSE calculation for two images represented as PyTorch tensors."""
        tensor_image1 = torch.rand(48, 48, 3, dtype=torch.float32)
        tensor_image2 = torch.rand(48, 48, 3, dtype=torch.float32)
        mse = calculate_mse(tensor_image1, tensor_image2)
        self.assertGreaterEqual(mse, 0.0)  # MSE should be >= 0

    def test_images_as_tensors_equal(self):
        """Test MSE calculation for two identical images represented as PyTorch tensors."""
        tensor_image1 = torch.rand(48, 48, 3, dtype=torch.float32)
        tensor_image2 = tensor_image1
        mse = calculate_mse(tensor_image1, tensor_image2)
        self.assertEqual(mse, 0.0)

    def test_single_color_channel_images(self):
        """Test MSE calculation for two images with a single color channel."""
        single_channel_image1 = np.random.rand(48, 48).astype(np.float32)
        single_channel_image2 = np.random.rand(48, 48).astype(np.float32)
        mse = calculate_mse(single_channel_image1, single_channel_image2)
        self.assertGreaterEqual(mse, 0.0)  # MSE should be >= 0

    def test_opposite_single_channel_images(self):
        """Test MSE calculation for a completely white image and a completely black image"""
        black_image = np.zeros((48, 48), dtype=np.float32)  # Single channel black image
        white_image = np.ones((48, 48), dtype=np.float32) * 255  # Single channel white image
        mse = calculate_mse(black_image, white_image)
        # The max possible squared difference is 255^2.
        expected_mse = (255 ** 2)
        self.assertEqual(mse, expected_mse)

    def test_different_shapes(self):
        """Test that a ValueError is raised for images of different shapes."""
        image1 = np.random.rand(48, 48, 3).astype(np.float32)  # 3-channel image
        image2 = np.random.rand(48, 48).astype(np.float32)  # Single channel image, different shape
        with self.assertRaises(ValueError):
            calculate_mse(image1, image2)

    def test_different_dimensions(self):
        """Test that a ValueError is raised for images of different shapes."""
        image1 = np.random.rand(48, 48, 3).astype(np.float32)  # 3-channel image
        image2 = np.random.rand(24, 48, 3).astype(np.float32)
        with self.assertRaises(ValueError):
            calculate_mse(image1, image2)

    def test_small_images(self):
        """Test MSE calculation for very small images."""
        image1 = np.ones((1, 1), dtype=np.float32)
        image2 = np.zeros((1, 1), dtype=np.float32)
        mse = calculate_mse(image1, image2)
        self.assertEqual(mse, 1.0)

    def test_large_images(self):
        """Test MSE calculation for large images."""
        image1 = np.zeros((1024, 1024), dtype=np.float32)
        image2 = np.ones((1024, 1024), dtype=np.float32) * 255
        mse = calculate_mse(image1, image2)
        expected_mse = (255 ** 2)
        self.assertEqual(mse, expected_mse)

    def test_high_contrast_images(self):
        """Test MSE calculation for images with extreme contrasts."""
        image1 = np.zeros((48, 48, 3), dtype=np.float32)
        image1[:, 24:, :] = 255  # Half black, half white
        image2 = np.ones((48, 48, 3), dtype=np.float32) * 255
        mse = calculate_mse(image1, image2)
        expected_mse = (255 ** 2) / 2  # Only half of the image contributes to the error
        self.assertEqual(mse, expected_mse)

    def test_different_data_types(self):
        """Test MSE calculation for images with different data types."""
        image1 = np.random.randint(0, 256, (48, 48, 3), dtype=np.uint8)
        image2 = image1.astype(np.float32)  # Convert to float32
        mse = calculate_mse(image1, image2)
        self.assertEqual(mse, 0.0)

    def test_empty_images(self):
        """Test that a ValueError is raised for empty images."""
        image1 = np.array([], dtype=np.float32).reshape(0, 0)
        image2 = np.array([], dtype=np.float32).reshape(0, 0)
        with self.assertRaises(ValueError):
            calculate_mse(image1, image2)

    def test_batch_of_images(self):
        """Test MSE calculation for a batch of images."""
        batch_size = 4
        image_shape = (48, 48, 3)
        batch_image1 = np.random.rand(batch_size, *image_shape).astype(np.float32)
        batch_image2 = np.random.rand(batch_size, *image_shape).astype(np.float32)
        mse = calculate_mse(batch_image1, batch_image2)
        self.assertGreaterEqual(mse, 0.0)  # MSE should be >= 0

    def test_batch_mse_correctness(self):
        """Test MSE calculation correctness for a batch of identical images."""
        batch_size = 5
        image_shape = (48, 48, 3)
        identical_image = np.random.rand(*image_shape).astype(np.float32)
        batch_image1 = np.tile(identical_image, (batch_size, 1, 1, 1))
        batch_image2 = np.tile(identical_image, (batch_size, 1, 1, 1))
        mse = calculate_mse(batch_image1, batch_image2)
        self.assertEqual(mse, 0.0)  # MSE should be 0 for identical images

    def test_batch_tensor_input(self):
        """Test MSE calculation for a batch of images represented as PyTorch tensors."""
        batch_size = 3
        image_shape = (48, 48, 3)
        tensor_image1 = torch.rand((batch_size, *image_shape), dtype=torch.float32)
        tensor_image2 = torch.rand((batch_size, *image_shape), dtype=torch.float32)
        mse = calculate_mse(tensor_image1, tensor_image2)
        self.assertGreaterEqual(mse, 0.0)  # MSE should be >= 0


if __name__ == '__main__':
    unittest.main()
