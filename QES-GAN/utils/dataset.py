import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Subset

"""Code credits: PQWGAN"""


def load_mnist(file_location='./datasets', image_size=None):
    """
    Loads the MNIST dataset and applies optional transformations.
    If the dataset is not in the specified folder, it will automatically download it from the
    internet and store it in that folder. Else, it will load from that folder.

    :param file_location: str. The path to the directory where the dataset will be stored.
    :param image_size: tuple or None. The desired size of the images in the dataset. If provided,
           the images will be resized to this size using bilinear interpolation. Defaults to None.

    :returns torch.utils.data.Dataset. The MNIST dataset object.

    """
    if not image_size is None:
        transform = transforms.Compose(
            [transforms.Resize(image_size), transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    else:
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    mnist_train = torchvision.datasets.MNIST(root=file_location, train=True, download=True, transform=transform)
    return mnist_train


def select_from_dataset(dataset, per_class_size, labels):
    """
    Selects a subset of data from a dataset based on specified labels and the desired size per class.

    :param dataset: torch.utils.data.Dataset. The dataset to select data from.
    :param per_class_size: int. The desired number of samples per class in the subset.
    :param labels: list. A list of labels to include in the subset.

    :returns torch.utils.data.Subset. The subset of the dataset containing the selected samples.

    """
    # Create a list to store the indices of samples for each class in the dataset
    indices_by_label = [[] for _ in range(10)]

    # Iterate through the dataset and populate the list with the indices of samples based on their labels.
    for i in range(len(dataset)):
        current_class = dataset[i][1]
        indices_by_label[current_class].append(i)
    # Stores the indices of the desired labels in `indices_of_desired_labels`.
    indices_of_desired_labels = [indices_by_label[i] for i in labels]

    # Create a `Subset` object from the original dataset using the selected indices,
    # limited to `per_class_size` number of samples per class
    # Returns the resulting subset of the dataset
    return Subset(dataset, [item for sublist in indices_of_desired_labels for item in sublist[:per_class_size]])


if __name__ == '__main__':
    dataset = select_from_dataset(load_mnist(file_location='../dataset', image_size=28), 1000,
                                  [0,1])