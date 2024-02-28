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
    Plot a PyTorch tensor as an image.

    :param tensor: tensor. The image.

    Returns:
    - None (displays the image plot)
    """
    if not isinstance(tensor, torch.Tensor):
        raise ValueError("Input must be a PyTorch tensor.")

    # Convert the tensor to a NumPy array for plotting
    image = tensor.numpy()

    # Create a figure and plot the image
    plt.figure(figsize=(5, 5))  # Adjust the figure size as needed
    plt.imshow(image, cmap='gray')
    plt.axis('off')  # Hide the axis labels
    plt.show()


def save_tensor(tensor, filename):
    """
    Plot a PyTorch tensor as an image.
    
    :param tensor: tensor. The image
    :param filename: string. The filename where to save the image.
    """
    if not isinstance(tensor, torch.Tensor):
        raise ValueError("Input must be a PyTorch tensor.")

    # Convert the tensor to a NumPy array for plotting
    image = tensor.numpy()

    # Create a figure and plot the image
    plt.figure()
    plt.imshow(tensor, cmap='gray')  # Assuming the tensor is grayscale
    plt.axis('off')  # Optional: Remove axes for cleaner visualization
    # Save the figure
    plt.savefig(filename, bbox_inches='tight', pad_inches=0)
    plt.close()  # Close the figure to free memory
