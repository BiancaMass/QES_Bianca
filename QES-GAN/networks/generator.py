import numpy as np
import torch
import torch.nn as nn

from qiskit import QuantumCircuit, QuantumRegister, execute, Aer, IBMQ
from qiskit.circuit import Parameter


class Q_Generator(nn.Module):
    def __init__(self, n_qubits):
        super().__init__()
        self.n_qubits = n_qubits

    def q_circuit(self, latent_vector):
        # Create a quantum circuit with n_qubits
        qc = QuantumCircuit(QuantumRegister(self.n_qubits, 'qubit'))

        # Ensure latent_vector elements are in the correct format
        if isinstance(latent_vector, torch.Tensor):
            latent_vector = latent_vector.detach().cpu().numpy().tolist()

        # Applying RY rotations based on the latent vector
        for i in range(self.n_qubits):
            qc.ry(latent_vector[i], i)

        for qbit in range(self.n_qubits):
            qc.h(qbit)

        return qc

    def get_statevector(self, qc):
        p = np.zeros(2 ** self.n_qubits)
        sim = Aer.get_backend('statevector' + '_simulator')  # Note: hard coded

        # if self.gpu:  # Note: if GPU
        #     sim.set_options(device='GPU')
        #     print('gpu used')

        job = execute(qc, sim)  # Execute the circuit `qc` on the simulator `sim`
        result = job.result()   # Retrieves the result of the execution
        statevector = result.get_statevector(qc)  # Extract the state vector

        # Calculate probabilities
        for i in range(len(np.asarray(statevector))):  # iterate over each element of s.v.
            # For each element in the state vector, calculate the probability of the
            # corresponding quantum state (the square of the abs. value of its amplitude)
            p[i] = np.absolute(statevector[i]) ** 2  # store probs in array `p`

        return p

    def from_probs_to_image(self, latent_vector):
        qc = self.q_circuit(latent_vector)
        probs = self.get_statevector(qc)
        # probs_given_ancilla_0 = probs[:2 ** (self.n_qubits - self.n_ancillas)]
        # post_measurement_probs = probs_given_ancilla_0 / torch.sum(probs_given_ancilla_0)

        post_measurement_probs = torch.tensor(probs) / torch.sum(probs)

        # normalise image between [-1, 1]
        post_processed_patch = ((post_measurement_probs / torch.max(
            post_measurement_probs)) - 0.5) * 2
        return post_processed_patch

    def forward(self, latent_vector):
        # Forward method to output the post_processed_patch
        return self.from_probs_to_image(latent_vector)
