import numpy as np
from qiskit import QuantumCircuit, QuantumRegister


n_tot_qubits = 5
n_data_qubits = 4
n_ancilla = n_tot_qubits - n_data_qubits
n_comp_basis = 2 ** n_tot_qubits

latent_vector = np.random.rand(n_tot_qubits)



qc_0 = QuantumCircuit(QuantumRegister(n_tot_qubits, 'qubit'))
# Applying RY rotations based on the latent vector
for i in range(n_tot_qubits):
    qc_0.ry(latent_vector[i], i)

for qbit in range(n_tot_qubits):
    qc_0.h(qbit)