import numpy as np
import torch
import torch.nn as nn

from qiskit import QuantumCircuit, QuantumRegister, execute, Aer, IBMQ
from qiskit.circuit import Parameter


# TODO: write function documentation

def get_probabilities(quantum_circuit, n_tot_qubits, sim):
    p = np.zeros(2 ** n_tot_qubits)

    # if self.gpu:  # Note: if GPU
    #     sim.set_options(device='GPU')
    #     print('gpu used')

    job = execute(quantum_circuit, sim)  # Execute the circuit `qc` on the simulator `sim`
    result = job.result()  # Retrieves the result of the execution
    statevector = result.get_probabilities(quantum_circuit)  # Extract the state vector

    # Calculate probabilities:
    # Iterate over each element of s.v. For each element, calculate the probability of the
    # corresponding quantum state (the square of the abs. value of its amplitude)
    for i in range(len(np.asarray(statevector))):
        p[i] = np.absolute(statevector[i]) ** 2  # store probs in array `p`

    return p


def from_probs_to_pixels(latent_vector, quantum_circuit, n_tot_qubits, n_ancillas):
    qc = quantum_circuit(latent_vector)
    probs = get_probabilities(qc)
    # Exclude the ancilla qubits values
    probs_given_ancilla_0 = probs[:2 ** (n_tot_qubits - n_ancillas)]
    # making sure the sum is exactly 1.0
    post_measurement_probs = probs_given_ancilla_0 / sum(probs_given_ancilla_0)

    # normalise image between [-1, 1]
    post_processed_patch = ((post_measurement_probs / max(post_measurement_probs)) - 0.5) * 2
    return post_processed_patch


def from_patches_to_images(batch_size, image_shape, n_patches, latent_vector, pixels_per_patch):
    # Forward method to output the post_processed_patch
    images_batch = torch.empty((batch_size, image_shape[0], image_shape[1]))
    # Loop across number of batches (total number of output images)
    for batch_image_index in range(batch_size):
        patches = torch.empty((0, n_patches))  # store patches of current image
        # Loop across sub-generators (one generation per patch)
        # TODO: the issue with this code is that all sub-generators are exactly identical
        #  because the difference between patches in the PQWGAN come from the weights,
        #  which differ for each sub-generator, while in my case they are all the same generator
        for patch in range(n_patches):
            current_patch = from_probs_to_pixels(latent_vector[batch_image_index])
            current_patch = current_patch[:pixels_per_patch]
            # TODO: THIS ASSUMES A PATCH IS A ROW, as it does not take shape into account!!!
            current_patch = torch.reshape(torch.from_numpy(current_patch),
                                          (1, pixels_per_patch))
            patches = torch.cat((current_patch, patches))
        images_batch[batch_image_index] = patches

    return images_batch
