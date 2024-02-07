import matplotlib.pyplot as plt
import torch


def plot_image_tensor(image_tensor, rows, cols, figsize=None):
    """
    Plots multiple images from a 3D tensor.

    Args:
        image_tensor (torch.Tensor): A 3D tensor containing images (shape: num_images x height x width).
        rows (int): Number of rows in the plot grid.
        cols (int): Number of columns in the plot grid.
        figsize (tuple, optional): Figure size. If None, size is automatically determined.
    """
    # Determine the number of images
    num_images = min(image_tensor.shape[0], rows * cols)

    if figsize is None:
        figsize = (cols * 2, rows * 2)  # Default figsize

    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = axes.flatten()  # Flatten in case of single row or column

    for i in range(num_images):
        ax = axes[i]
        ax.imshow(image_tensor[i], cmap='gray', interpolation='none')
        ax.axis('off')

    # Turn off axes for any unused subplots
    for j in range(num_images, rows * cols):
        axes[j].axis('off')

    plt.tight_layout()
    plt.show()

# Example usage
# image_tensor is your tensor of images with shape [num_images, height, width]
# plot_image_tensor(image_tensor, rows=4, cols=5)


def plot_tensor(tensor):
    """
    Plot a PyTorch tensor of size [28, 28] as an image.

    Args:
    - tensor (torch.Tensor): A PyTorch tensor of size [28, 28].

    Returns:
    - None (displays the image plot)
    """
    if not isinstance(tensor, torch.Tensor):
        raise ValueError("Input must be a PyTorch tensor.")

    if tensor.shape != torch.Size([28, 28]):
        raise ValueError("Input tensor must have size [28, 28].")

    # Convert the tensor to a NumPy array for plotting
    image = tensor.numpy()

    # Create a figure and plot the image
    plt.figure(figsize=(5, 5))  # Adjust the figure size as needed
    plt.imshow(image, cmap='gray')
    plt.axis('off')  # Hide the axis labels
    plt.show()

# Example usage:
# Assuming you have a tensor named 'my_tensor' of size [28, 28]
# plot_tensor(my_tensor)