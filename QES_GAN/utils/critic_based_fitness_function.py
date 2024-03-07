import numpy as np
import torch

from QuantumEvolutionaryAlgorithms.QES_GAN.networks.generator_methods import from_patches_to_image


def scoring_function(batch_size, critic, qc,
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
    :param critic: torch.nn.Module. The pre-trained critic network used for evaluating the images.
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
    generated_images = []
    for batch_index in range(batch_size):
        generated_image = from_patches_to_image(quantum_circuit=qc,
                                                n_tot_qubits=n_tot_qubits,
                                                n_ancillas=n_ancillas,
                                                n_patches=n_patches,
                                                pixels_per_patch=pixels_per_patch,
                                                patch_width=patch_width,
                                                patch_height=patch_height,
                                                sim=sim)
        generated_images.append(generated_image)


    generated_images_tensor = torch.stack(generated_images)
    generated_images_tensor = generated_images_tensor.to(device)

    fake_validity = critic(generated_images_tensor)

    # Calculate the average score for this batch as the organism's score
    average_score = float(torch.mean(fake_validity.detach()))

    return average_score
