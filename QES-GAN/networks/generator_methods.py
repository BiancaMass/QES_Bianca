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
    sim = Aer.get_backend(sim + '_simulator')  # TODO: not the right place to put it, only for
    # debugging
    job = execute(quantum_circuit, sim)  # Execute the circuit `qc` on the simulator `sim`
    result = job.result()  # Retrieves the result of the execution
    statevector = result.get_statevector(quantum_circuit)
    # Calculate probabilities:
    # Iterate over each element of s.v. For each element, calculate the probability of the
    # corresponding quantum state (the square of the abs. value of its amplitude)
    for i in range(len(np.asarray(statevector))):
        p[i] = np.absolute(statevector[i]) ** 2  # store probs in array `p`

    return p


def from_probs_to_pixels(latent_vector, quantum_circuit, n_tot_qubits, n_ancillas, sim):
    probs = get_probabilities(quantum_circuit=quantum_circuit, n_tot_qubits=n_tot_qubits, sim=sim)
    # Exclude the ancilla qubits values
    probs_given_ancilla_0 = probs[:2 ** (n_tot_qubits - n_ancillas)]
    # making sure the sum is exactly 1.0
    post_measurement_probs = probs_given_ancilla_0 / sum(probs_given_ancilla_0)

    # normalise image between [-1, 1]
    post_processed_patch = ((post_measurement_probs / max(post_measurement_probs)) - 0.5) * 2
    return post_processed_patch


def from_patches_to_image(latent_vector, quantum_circuit, n_tot_qubits, n_ancillas,
                          n_patches, pixels_per_patch, sim):
    """

    :param latent_vector:
    :param quantum_circuit:
    :param n_tot_qubits:
    :param n_ancillas:
    :param n_patches:
    :param pixels_per_patch:
    :param sim:
    :return: A single image
    """
    final_image = torch.empty((0, n_patches))  # store patches of current image
    # TODO: the issue with this code is that all sub-generators are exactly identical
    #  because the difference between patches in the PQWGAN come from the weights,
    #  which differ for each sub-generator, while in my case they are all the same generator
    for patch in range(n_patches):
        current_patch = from_probs_to_pixels(latent_vector=latent_vector,  # latent_vector[n_patches]
                                             quantum_circuit=quantum_circuit,
                                             n_tot_qubits=n_tot_qubits,
                                             n_ancillas=n_ancillas,
                                             sim=sim)
        current_patch = current_patch[:pixels_per_patch]
        # Note: This assumes patch is a row, as it does not take shape into account.
        current_patch = torch.reshape(torch.from_numpy(current_patch),
                                      (1, pixels_per_patch))
        final_image = torch.cat((current_patch, final_image))

    return final_image
