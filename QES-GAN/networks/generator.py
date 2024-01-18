from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit import Parameter
import numpy as np
import torch
import torch.nn as nn


class Q_Generator(nn.Module):
    def __init__(self, n_qubits, latent_vector):
        super().__init__()
        self.n_qubits = n_qubits
        self.latent_vector = latent_vector

    def q_circuit(self, latent_vector):
        # Create a quantum circuit with n_qubits
        qc = QuantumCircuit(QuantumRegister(self.n_qubits, 'qubit'))

        # Applying RY rotations based on the latent vector
        for i in range(self.n_qubits):
            qc.ry(latent_vector[i], i)

        for qbit in range(self.n_qubits):
            qc.h(qbit)

        return qc
