# Critic Network: You should have a critic network that takes an image as input and outputs a
# scalar value. This network is trained to approximate the Wasserstein distance. For real
# images, it should output higher values; for generated images, lower values.

# THE CRITIC CAN BE TRAINED ONLY ONCE AND KEPT GOOD! FOR EVALUATION OF NEW ANSATZES - HOWEVER
# THERE ARE SOME RISKS WITH THIS APPROACH:
# 1. risking overfitting the generator to whatever critic you are training to, making the result of
# the evolution less generalizable
# 2. Using a fixed critic for evaluation, means having a consistent benchmark, allowing to compare
# different ansatzes architectures under the same standards. However, this approach assumes the
# critic is well-trained and can generalize across a wide variety of generated images. The critic
# needs to be trained on a diverse set of images and should be robust enough to handle a wide
# range of generator outputs
# 3. As a solution, you could consider starting with a pretty solid critic and and re-training it
# every so often, especially if you notice signs of overfitting, such as stagnation in the
# evolutionary processes, discrepancies between scores and visual quality. It is also good
# practice to validate the  generators with a separate set of real images that the critic has not
# been trained on.
#
# Scoring Function: The scoring function will take an 'organism' (in this case, an image
# generated by the GAN's generator) and pass it through the critic network to get the Wasserstein
# distance.
import numpy as np
from statistics import mean

from networks.generator_methods import from_patches_to_image


def scoring_function(batch_size, critic, qc,
                     n_tot_qubits, n_ancillas, n_patches,
                     pixels_per_patch, sim):
    """
    Calculates a score for a given quantum circuit based on a pre-trained critic network.
    The function generates a batch of images using the specified quantum circuit and latent vectors.
    It then passes these generated images through the critic network to obtain a score,
    which represents the 'fitness' of the quantum circuit in generating images that resemble real images.
    The function returns the average score for the whole the batch, providing an overall
    evaluation of the quantum circuit.

    # TODO: determine data type of some of these params
    :param batch_size: int.
    :param critic: nn.Module.
    :param qc: ???
    :param n_tot_qubits: int.
    :param n_ancillas: int.
    :param n_patches: int.
    :param pixels_per_patch: int.
    :param sim: ???. (string?)

    :return: float. The average score for the generated batch of images, as evaluated by the critic.
    """
    latent_vector = np.random.rand(batch_size, n_tot_qubits)
    generated_images = []
    for batch_index in range(batch_size):
        generated_image = from_patches_to_image(latent_vector=latent_vector[batch_index],
                                                quantum_circuit=qc,
                                                n_tot_qubits=n_tot_qubits,
                                                n_ancillas=n_ancillas,
                                                n_patches=n_patches,
                                                pixels_per_patch=pixels_per_patch,
                                                sim=sim)
        generated_images.append(generated_image)

    # Evaluate the generated images using the pre-trained critic network
    fake_validity = critic(generated_images)

    # Calculate the average score for this batch as the organism's score
    average_score = mean(fake_validity)  # TODO: check that this gives a good score for real images

    return average_score