import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from qiskit import QuantumCircuit
from qiskit.circuit.library import RYGate, RZGate


class PQWGAN_QGCC():
    def __init__(self, input_dimensions, output_dimensions, n_qubits, n_ancillas, n_layers,
                 dest_qubit_indexes):
        """
       Initializes the Quantum-Classical GAN Class

       Args:
       input_dimensions (tuple): Tuple of two integers (width, height).
       output_dimensions (tuple): Tuple of two integers (width, height).
       n_qubits (int): Number of qubits for the quantum circuit.
       n_ancillas (int): Number of ancillary qubits for the quantum circuit.
       n_layers (int): Number of layers in the quantum circuit.
       dest_qubit_indexes (list): List of destination qubit indexes.
       """
        # Initialize the PQWGAN_QGCC class: create the discriminator and generator entities.
        input_width, input_height = input_dimensions
        self.input_width = input_width
        self.input_height = input_height

        output_width, output_height = output_dimensions
        self.output_width = output_width
        self.output_height = output_height

        self.input_pixels = input_width * input_height
        self.output_pixels = output_width * output_height

        self.critic = self.ClassicalCritic(self.output_pixels)
        self.generator = self.QuantumGenerator(output_dimensions=(self.output_width, self.output_height),
                                               n_qubits=n_qubits,
                                               n_ancillas=n_ancillas,
                                               n_layers=n_layers,
                                               dest_qubit_indexes=dest_qubit_indexes)

    class ClassicalCritic(nn.Module):
        # takes the image either real or generated and decides which one it is
        def __init__(self, output_pixels):
            super().__init__()
            self.output_pixels = output_pixels

            self.fc1 = nn.Linear(int(self.output_pixels), 128)  # Note: was 512
            self.fc2 = nn.Linear(128, 64)  # Note: was 256
            self.fc3 = nn.Linear(64, 1)

        def forward(self, x):
            x = x.view(x.shape[0], -1)  # Flatten input image
            # Apply leaky ReLU activation to the 1st and 2nd fully connected layer outputs
            x = F.leaky_relu(self.fc1(x), 0.2)
            x = F.leaky_relu(self.fc2(x), 0.2)
            return self.fc3(x)  # Return the output of the 3rd fully connected layer

    class QuantumGenerator(nn.Module):
        def __init__(self, output_dimensions,
                     n_qubits, n_ancillas, n_layers, dest_qubit_indexes):
            super().__init__()
            # self.input_image_flat = input_image_flat
            out_width, out_height = output_dimensions
            self.output_width = out_width
            self.output_height = out_height
            self.output_pixels = out_width * out_height
            self.n_qubits = n_qubits
            self.n_ancillas = n_ancillas
            self.n_layers = n_layers
            self.dest_qubit_indexes = dest_qubit_indexes
            # TODO: PENNYLANE
            # self.q_device = qml.device("default.qubit", wires=n_qubits)
            self.q_circ = QuantumCircuit(n_qubits)

            ### Initialize weights ###
            # Each Rot gate needs 3 parameters, hence we have 3 random values per qubit per layer
            # sampled from a uniform  distr. over [0, 1).
            # CHECK: are these updated during training?
            weights = np.random.rand(self.n_layers, self.n_qubits, 3)
            # convert into trainable param with torch framework
            # self.params = Variable(torch.tensor(weights), requires_grad=True)
            self.params = nn.Parameter(torch.tensor(weights, dtype=torch.float32),
                                       requires_grad=True)
            # It contains a quantum function (the variational circuit) as well as the computational
            # device the QVC is executed on.
            # TODO: PENNYLANE
            # self.qnode = qml.QNode(func=self.circuit,     # defined below
            #                        device=self.q_device,  # the pennylane device initialized above
            #                        interface="torch")    # The interface for classical backprop.

        def forward(self, x):  # CHECK: does it get the input image as input?
            """ Perform a forward pass through the QuantumGenerator.

            Args:
                x (torch.Tensor): Input tensor containing a batch of images.
            Returns:
                torch.Tensor: Output tensor containing a batch of processed images, reshaped to
                (num_images, width, length).
            """
            # Initialize an empty list to store the outputs
            output_images_list = []

            for input_image in x:
                generator_output = self.partial_trace_and_postprocess(input_image, self.params).float()
                # Ensure generator_output is 2D (batch size x output size)
                generator_output = generator_output.unsqueeze(0)
                # Append the generator_output to the list
                output_images_list.append(generator_output)

            # Concatenate all outputs along dimension 1
            output_images = torch.cat(output_images_list, 1)

            # Reshape output (num_images, width, length)
            final_out = output_images.view(x.shape[0], self.output_width,
                                           self.output_height)

            # TODO: normalize between 0 and 1   ??

            return final_out

        def generate_circuit(self, weights, input_image_flat):
            circuit = QuantumCircuit(self.n_qubits)
            # Dynamic circuit construction based on weights and input
            # Example: Applying RY and Rot gates based on weights
            for qubit_index, pixel_value in enumerate(input_image_flat):
                circuit.ry(pixel_value, qubit_index)
            for i in range(self.n_layers):
                for j in range(self.n_qubits):
                    theta, phi, lam = weights[i, j]
                    circuit.rz(phi, j)
                    circuit.ry(theta, j)
                    circuit.rz(lam, j)
            return circuit


        def circuit(self, input_image_flat, weights, dest_qubit_indexes):
            # TODO: PENNYLANE
            # Encode each input pixel value in a qubit
            for pixel, qubit_index in enumerate(dest_qubit_indexes):
                self.q_circ.append(RYGate(input_image_flat[pixel], qubit_index))

            # Apply Hadamard to all qubits
            for qubit in range(self.n_qubits):
                self.q_circ.h(qubit)

            for i in range(self.n_layers):
                for j in range(self.n_qubits):
                    # weights indexed: [layer, qubit, param for the specific gate (θ, ϕ, λ) ]
                    theta, phi, lam = weights[i, j]
                    self.q_circ.append(RZGate(phi), [j])
                    self.q_circ.append(RYGate(theta), [j])
                    self.q_circ.append(RZGate(lam), [j])
                    qml.Rot(weights[i, j][0],
                            weights[i, j][1],
                            weights[i, j][2],
                            wires=j)

            return qml.probs(wires=list(range(self.n_qubits)))

        def partial_trace_and_postprocess(self, input_image, weights):
            # Compute the probabilities by running the quantum circuit
            if input_image.dim() > 1:
                input_image = input_image.view(-1)

            circuit_probs = self.qnode(input_image, weights, self.dest_qubit_indexes)

            post_measurement_probs = circuit_probs / torch.sum(circuit_probs)
            post_processed_patch = (post_measurement_probs / torch.max(post_measurement_probs))

            truncated_output_tensor = post_processed_patch[:self.output_pixels]

            # Sum the squared differences between the output pixels and the target pixels
            # cost += torch.sum((truncated_output_tensor - target_image_tensor) ** 2)

            # TODO: think about presence of ancillas - insert ancillas
            # probs_given_ancilla_0 = probs[:2 ** (self.n_qubits - self.n_ancillas)]
            # post_measurement_probs = probs_given_ancilla_0 / torch.sum(probs_given_ancilla_0)
            #
            # # normalize between 0, 1
            # post_processed_patch = (post_measurement_probs / torch.max(post_measurement_probs))

            return truncated_output_tensor
