import scipy
from scipy import stats
import os
import torch
from statistics import mean
from torch.utils.data import DataLoader

from QuantumEvolutionaryAlgorithms.QES_GAN.utils.dataset import load_mnist, select_from_dataset
from QuantumEvolutionaryAlgorithms.QES_GAN.networks.generator_methods import from_patches_to_image


def emd_scoring_function(real_images_preloaded, batch_size, qc,
                         n_tot_qubits, n_ancillas, n_patches,
                         pixels_per_patch, patch_width, patch_height, sim, device):
    """
    Calculates a score for a given quantum circuit based on a pre-trained critic network.
    The function generates a batch of images using the specified quantum circuit and latent vectors.
    It then passes these generated images through the critic network to obtain a score,
    which represents the 'fitness' of the quantum circuit in generating images that resemble real images.
    The function returns the average score for the whole the batch, providing an overall
    evaluation of the quantum circuit.

    :param batch_size: int. The number of images to generate and evaluate in one batch.
    :param qc: qiskit.circuit.quantumcircuit.QuantumCircuit. The quantum circuit to be executed.
    :param n_tot_qubits: int. The total number of qubits used in the quantum circuit.
    :param n_ancillas: int. The number of ancillary qubits in the quantum circuit.
    :param n_patches: int. The number of patches into which each image is divided.
    :param pixels_per_patch: int. The number of pixels in each patch of an image.
    :param sim: StatevectorSimulator. The simulator used to run the quantum circuit.
    :param patch_width: width (in pixels) of each patch.
    :param patch_height: height (in pixels) of each patch.
    :param device: the device (cpu or gpu) on which computations are to be performed.

    :return: float. The average score for the generated batch of images, as evaluated by the critic.
    """

    # os.chdir(os.path.dirname(os.path.abspath(__file__)))  # making sure the dir path is right
    # dataset = select_from_dataset(load_mnist(image_size=img_size), 1000, classes)
    # dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=1)
    # real_images_list = real_images_preloaded
    # for i, (real_imgs, _) in enumerate(dataloader):
    #     real_images_list = real_imgs  # Directly assign the batch of images
    #     break  # Exit the loop after the first batch

    generated_images_list = []
    for batch_index in range(batch_size):
        generated_image = from_patches_to_image(quantum_circuit=qc,
                                                n_tot_qubits=n_tot_qubits,
                                                n_ancillas=n_ancillas,
                                                n_patches=n_patches,
                                                pixels_per_patch=pixels_per_patch,
                                                patch_width=patch_width,
                                                patch_height=patch_height,
                                                sim=sim)
        generated_images_list.append(generated_image)

    real_images_tensor = real_images_preloaded
    generated_images_tensor = torch.stack(generated_images_list)

    real_images_flat = real_images_tensor.view(real_images_tensor.size(0), -1)
    generated_images_flat = generated_images_tensor.view(generated_images_tensor.size(0), -1)

    generated_images_flat_np = generated_images_flat.cpu().detach().numpy()
    real_images_flat_np = real_images_flat.cpu().detach().numpy()

    distance_real_gen = scipy.stats.wasserstein_distance(real_images_flat_np.flatten(),
                                                         generated_images_flat_np.flatten())


    return distance_real_gen
