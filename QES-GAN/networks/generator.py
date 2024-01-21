import numpy as np
import torch
import torch.nn as nn

from qiskit import QuantumCircuit, QuantumRegister, execute, Aer, IBMQ
from qiskit.circuit import Parameter


class Q_Generator(nn.Module):
    # TODO: this will have to become parameterized in order to change ansatz structure each time
    def __init__(self, batch_size, n_patches, n_data_qubits, n_ancillas, image_shape,
                 pixels_per_patch):
        super().__init__()
        self.batch_size = batch_size
        self.n_data_qubits = n_data_qubits
        self.n_patches = n_patches
        self.n_tot_qubits = n_data_qubits + n_ancillas
        self.n_ancillas = n_ancillas
        self.image_shape = image_shape
        self.pixels_per_patch = pixels_per_patch

    def q_circuit(self, latent_vector):
        # Create a quantum circuit with n_tot_qubits
        qc = QuantumCircuit(QuantumRegister(self.n_tot_qubits, 'qubit'))

        # Ensure latent_vector elements are in the correct format
        if isinstance(latent_vector, torch.Tensor):
            latent_vector = latent_vector.detach().cpu().numpy().tolist()

        # Applying RY rotations based on the latent vector
        for i in range(self.n_tot_qubits):
            qc.ry(latent_vector[i], i)

        for qbit in range(self.n_tot_qubits):
            qc.h(qbit)

        return qc

    def get_statevector(self, qc):
        p = np.zeros(2 ** self.n_tot_qubits)
        sim = Aer.get_backend('statevector' + '_simulator')  # Note: hard coded

        # if self.gpu:  # Note: if GPU
        #     sim.set_options(device='GPU')
        #     print('gpu used')

        job = execute(qc, sim)  # Execute the circuit `qc` on the simulator `sim`
        result = job.result()   # Retrieves the result of the execution
        statevector = result.get_statevector(qc)  # Extract the state vector

        # Calculate probabilities:
        # Iterate over each element of s.v. For each element, calculate the probability of the
        # corresponding quantum state (the square of the abs. value of its amplitude)
        for i in range(len(np.asarray(statevector))):
            p[i] = np.absolute(statevector[i]) ** 2  # store probs in array `p`

        return p

    def from_probs_to_pixels(self, latent_vector):
        qc = self.q_circuit(latent_vector)
        probs = self.get_statevector(qc)
        # Exclude the ancilla qubits values
        probs_given_ancilla_0 = probs[:2 ** (self.n_tot_qubits - self.n_ancillas)]
        # making sure the sum is exactly 1.0
        post_measurement_probs = probs_given_ancilla_0 / sum(probs_given_ancilla_0)

        # normalise image between [-1, 1]
        post_processed_patch = ((post_measurement_probs / max(post_measurement_probs)) - 0.5) * 2
        return post_processed_patch

    def forward(self, latent_vector):
        # Forward method to output the post_processed_patch
        images_batch = torch.empty((self.batch_size, self.image_shape[0], self.image_shape[1]))
        for batch_image_num in range(self.batch_size):
            patches = torch.empty((0, self.n_patches))  # store patches of current image
            for patch in range(self.n_patches):
                # TODO: now every patch is the same, you have to change generator structure (like the
                #  weights of the parameters??) - this is something to think about and understand well
                #  namely, where does the difference in patch come from.

                current_patch = self.from_probs_to_pixels(latent_vector[batch_image_num])
                current_patch = current_patch[:self.pixels_per_patch]
                # TODO: THIS ASSUMES A PATCH IS A ROW, as it does not take shape into account!!!
                current_patch = torch.reshape(torch.from_numpy(current_patch), (1, self.pixels_per_patch))
                patches = torch.cat((current_patch, patches))
            images_batch[batch_image_num] = patches

        return images_batch
