"""
Vincenzo's code, edited by Bianca.
"""

import numpy as np
import random
import math
from qiskit import QuantumCircuit, QuantumRegister, execute, Aer, IBMQ
from qiskit.circuit.library import RYGate, RXGate, RZGate, RXXGate, RYYGate, RZZGate
from qiskit_aer import AerSimulator
from qiskit.providers.fake_provider import FakeMumbaiV2
from decimal import Decimal, getcontext

from networks.generator_methods import get_probabilities, from_probs_to_pixels, from_patches_to_image
from generator_fitness_function import scoring_function

# np.random.seed(123)  # for replicability


# TODO: improve documentation
# Found issue: the latent vector is never given to the new circuits. This needs to be changes.
# So, right now I define a circuit with an initial latent vector and change the gates (inlcuding
# the encoding gates) but instead I should keep the encoding gates, change the latent vector,
# and change the other gates with the evolutionary algorithm.

class Qes:
    """
    Hybrid quantum-classical optimization technique
    """

    def __init__(self, n_data_qubits, n_ancilla, image_shape,
                 batch_size, critic_net,
                 n_children, n_max_evaluations,
                 shots, simulator, noise, gpu,
                 dtheta, action_weights, multi_action_pb,
                 max_gen_no_improvement,
                 **kwargs):

        """ Initialization of the population and its settings
        :param n_data_qubits: integer. Number of data qubits for the circuit.
        :param n_ancilla: integer. Number of ancilla qubits for the circuit.
        :param image_shape: tuple. weight and height of the image.
        :param batch_size: int. Batch size to evaluate a qc (how many times to call it for the
        fitness)
        :param n_children: integer. Number of children for each generation.
        :param n_max_evaluations: integer. Maximum number of times a new generated ansatz is
        evaluated.
        :param shots: integer. Number of executions on the circuit to get the prob. distribution.
        :param simulator: string. Qiskit simulator. Either 'statevector' or 'qasm'.
        :param noise: Boolean. True if a noisy simulation is required, False otherwise.
        :param gpu: Boolean. If True, it simulates quantum circuits on GPUs if available.
        # TODO: add code that checks if GPU is available and if not prints a warning if True was
            selected.
        :param dtheta: float. Maximum displacement for the angle parameter in the mutation action.
        :param action_weights: list. Probability to choose between the 4 possible actions. Their
        sum must be 100.
        :param multi_action_pb: float. Probability to get multiple actions in the same generation.
        :param max_gen_no_improvement: integer. Number of generations with no improvements after which some changes will be applied
        :keyword max_depth: integer. It fixes an upper bound on the quantum circuits depth.
        """

        self.n_data_qubits = n_data_qubits
        self.n_ancilla = n_ancilla
        self.n_tot_qubits = n_data_qubits + n_ancilla

        self.image_width, self.image_height = image_shape[0], image_shape[1]
        self.n_pixels = image_shape[0] * image_shape[1]
        self.n_patches = image_shape[0]  # TODO hard coded for now assuming one patch per row
        self.pixels_per_patch = int(self.n_pixels/self.n_patches)
        self.batch_size = batch_size
        self.critic_net = critic_net

        self.n_children = n_children
        self.n_max_evaluations = n_max_evaluations
        self.shots = shots
        self.simulator = simulator
        self.gpu = gpu
        self.noise = noise
        self.dtheta = dtheta
        self.action_weights = action_weights
        self.multi_action_pb = multi_action_pb
        self.max_gen_no_improvement = max_gen_no_improvement + 1

        # Number of generations for the evolution algorithm
        self.n_generations = math.ceil(n_max_evaluations / n_children)

        #######################
        # CREATE THE 0-TH INDIVIDUAL (QUANTUM CIRCUIT)
        #######################

        # Number of the computational basis states in the `n_qubits` qubits Hilbert space
        self.N = 2 ** self.n_tot_qubits

        ### START VANILLA CIRCUIT ###
        latent_vector_0 = np.random.rand(self.n_tot_qubits)

        qc_0 = QuantumCircuit(QuantumRegister(self.n_tot_qubits, 'qubit'))
        # Applying RY rotations based on the latent vector
        for i in range(self.n_tot_qubits):
            qc_0.ry(latent_vector_0[i], i)

        for qbit in range(self.n_tot_qubits):
            qc_0.h(qbit)
        ### END VANILLA CIRCUIT ###

        # CURRENT generation parameters
        self.ind = qc_0  # best individual at the beginning is the vanilla circuit
        # Population (circuits) generated in the current generation from the best qc of the
        # previous one
        self.population = [self.ind]
        # Candidate solution (real-valued vectors) in the current generation
        self.candidate_sol = None
        # Fitness values (real values) of the candidate solutions in the current generation
        self.fitnesses = None
        # Actions taken in the current generation
        self.act_choice = None

        # Best individuals over ALL generations
        self.best_individuals = [qc_0]  # At initialization it is the 0-th circuit
        # List of circuits depths over the generations
        self.depth = [self.ind.depth()]
        # Best solution (real-valued vector) in the current generation
        self.best_solution = []
        # Fitness values (real value) over the generations
        self.best_fitness = []
        # Best actions taken over the generations
        self.best_actions = None

        # Useful Attributes:
        # Number of generations without improvements found
        self.no_improvements = 0
        # Number of evaluations of the fitness function
        self.fitness_evaluations = 0
        # Current number of generation in the classical evolutionary algorithm
        self.current_gen = 0
        # Control over multi actions in the current generations
        self.counting_multi_action = None
        # Restrict to quantum circuits with a chosen max_depth
        self.max_depth = None

        # Add a max_depth argument if provided with additional arguments (kwargs)
        for max_depth in kwargs.values():
            self.max_depth = max_depth
        # All the algorithm data we need to store
        self.output = None

        print('Initial quantum circuit: \n', self.ind)

    def action(self):
        """ It generates n_children of the individual and apply one of the 4 POSSIBLE ACTIONS(add,
        delete, swap, mutate) on each of them. Then the new quantum circuits are stored in the
        attribute 'population' """

        population = []
        print(f"Parent ansatz \n \n {self.ind}")
        # Copy and modify the circuit as many times as the chosen branching factor (`n_children`)
        for i in range(self.n_children):
            qc = self.ind.copy()  # Current child (copy of the current parent)
            if self.max_depth is not None:  # if user gave a max_depth as input argument
                # the depth is the length of the critical path (longest sequence of gates)
                if qc.depth() >= self.max_depth - 1:  # if current depth is one step away from max
                    counter = 1  # set counter to 1, i.e., only apply one action to the circuit
                    self.counting_multi_action = 0  # to avoid additional actions to be applied
                else:
                    # apply *at least* one action
                    counter = self.multiaction().counting_multi_action + 1
            else:
                # apply *at least* one action
                counter = self.multiaction().counting_multi_action + 1

            self.act_choice = random.choices(['A', 'D', 'S', 'M'], weights=self.action_weights,
                                             k=counter)  # outputs a list of k-number of actions
            angle = random.random() * 2 * math.pi
            gate_list = [qc.rx, qc.ry, qc.rz, qc.rxx, qc.ryy, qc.rzz]
            gate_dict = {'rx': RXGate, 'ry': RYGate, 'rz': RZGate,
                         'rxx': RXXGate, 'ryy': RYYGate, 'rzz': RZZGate}
            position = 0
            print('action choice:', self.act_choice, 'for the copy number: ', i)

            for j in range(counter):  # go over the selected actions for this one child
                # ADD a random gate on a random qubit at the end of the parent quantum circuit
                if self.act_choice[j] == 'A':
                    print("ADDING action was selected \n")
                    # Chooses 2 locations for the destination qubit(s). Only one will be used for
                    # rx, ry, rz, two will be used for rxx, ryy, rzz
                    position = random.sample([i for i in range(len(qc.qubits))], k=2)
                    # Choose the type of gate (pick an index for the gates list)
                    choice = random.randint(0, len(gate_list) - 1)
                    if 0 <= choice < 3:
                        print(f"Adding a {gate_list[choice]} gate at position: {position[0]}")
                        gate_list[choice](angle, position[0])
                    else:
                        print(f"Adding a {gate_list[choice]} gate at positions: {position}")
                        gate_list[choice](angle, position[0], position[1])

                    print('Circuit after ADD ACTION:')
                    print(qc)

                # DELETE a random gate in a random position of the parent quantum circuit
                elif self.act_choice[j] == 'D':
                    print("DELETE action was selected")
                    # Pick a position for the gate to remove.
                    # Exclude the the first n_tot_qubits gates (encoding gates)
                    position = random.randint(self.n_tot_qubits, len(qc.data) - 1)
                    qc.data.remove(qc.data[position])
                    print("Circuit after DELETE action")
                    print(qc)

                # SWAP: Remove a gate in a random position and replace it with a new gate randomly
                # chosen
                elif self.act_choice[j] == 'S':
                    print("SWAP action was selected \n")
                    if len(qc.data) - 1 - self.n_tot_qubits > 0:
                        # Pick a position for the gate to swap
                        # An int b/w n_tot_qubits and (n_tot_gates-2)
                        # Exclude the the first n_tot_qubits gates (encoding gates)
                        position = random.randint(self.n_tot_qubits, len(qc.data) - 2)
                    # CHECK: where is position defined if not in the if statement?
                    gate_to_remove = qc.data[position][0]
                    gate_to_add = random.choice(list(gate_dict.values()))(angle)
                    # Avoid removing and adding the same gate
                    while gate_to_add.name == gate_to_remove.name:
                        gate_to_add = random.choice(list(gate_dict.values()))(angle)
                    if gate_to_add.name == 'rzz' or gate_to_add.name == 'rxx' or gate_to_add.name == 'ryy':
                        n_qubits = 2
                    else:
                        n_qubits = 1
                    # number of qubits affected by the gate to be swapped
                    length = len(qc.data[position][1])
                    # if we are swapping gates with the same amount of qubits
                    if length == n_qubits:
                        element_to_remove = list(qc.data[position])
                        element_to_remove[0] = gate_to_add  # swap the gates
                        element_to_add = tuple(element_to_remove)
                        qc.data[position] = element_to_add
                    elif length > n_qubits:
                        element_to_remove = list(qc.data[position])
                        element_to_remove[0] = gate_to_add
                        element_to_remove[1] = [random.choice(qc.data[position][1])]
                        element_to_add = tuple(element_to_remove)
                        qc.data[position] = element_to_add
                    elif length < n_qubits:
                        element_to_remove = list(qc.data[position])
                        element_to_remove[0] = gate_to_add
                        qubits_available = []
                        for q in qc.qubits:
                            if [q] != qc.data[position][1]:
                                qubits_available.append(q)
                        qubits_ = [qc.data[position][1], random.choice(qubits_available)]
                        random.shuffle(qubits_)
                        element_to_remove[1] = qubits_
                        element_to_add = tuple(element_to_remove)
                        qc.data[position] = element_to_add

                # MUTATE: Choose a gate and change its angle by a value between [θ-d_θ, θ+d_θ]
                elif self.act_choice[j] == 'M':
                    print("MUTATE action was selected \n")
                    to_not_select = 'h'  # because h is not a rotation gate, so cannot be MUTATED
                    check = True
                    gate_to_mutate = None

                    # if the only gates left are encoding, there is nothing to mutate this round
                    if len(qc.data) == self.n_tot_qubits:
                        check = False
                        print("Skipping mutate action because there are no gates available to "
                              "mutate")

                    # TODO: not gonna work if there are only hadamard gates
                    while check:
                        # pick a number between n_tot_qubits and n_tot_gates to select the gate
                        # to mutate while avoiding the first n_tot_qubits gates (encoding)
                        position = random.choice([i for i in range(self.n_tot_qubits,
                                                                   len(qc.data))])
                        gate_to_mutate = qc.data[position]

                        if gate_to_mutate[0].name != to_not_select:
                            check = False
                    angle_new = qc.data[position][0].params[0] + random.uniform(0, self.dtheta)
                    element_to_mute = list(gate_to_mutate)
                    element_to_mute[0] = gate_dict[gate_to_mutate[0].name](angle_new)
                    element_to_add = tuple(element_to_mute)
                    qc.data[position] = element_to_add
                    print(f'Muted gate {element_to_mute} into {element_to_mute} at '
                          f'position {position}')

                # In case of multiactions we are appending more circuits to the population,
                # if you don't want that put the next code line outside of the for loop on counter
            population.append(qc)
        self.population = population
        print(f'Current population after the action(): {self.population}')
        return self

    def encode(self):
        """
        It transforms a quantum circuit (ind) in a string of real values of length 2^N, where N=len(ind).
        """

        latent_vector = np.random.rand(self.n_tot_qubits)  # TODO: this is not used!!

        if self.simulator == 'statevector':
            self.noise = False
        if self.noise:
            backend = FakeMumbaiV2()
            sim = AerSimulator.from_backend(backend)
        else:
            sim = Aer.get_backend(self.simulator + '_simulator')
        # TODO: add condition that if GPU was set but is not available, it goes back
        #  to CPU and prints a warning that it did that
        if self.gpu:
            sim.set_options(device='GPU')  # Check this works with DSDI infrastructure
            print('gpu used')

        self.candidate_sol = []
        # Let qasm be more free because of the shot noise
        if self.simulator == 'qasm':
            if self.no_improvements > 10:
                self.population.insert(0, self.best_individuals[-1])

        for j in range(len(self.population)):
            qc = self.population[j].copy()
            resulting_image = np.zeros(self.n_pixels)

            # Set up the type of simulator we want to use
            if self.simulator == 'qasm':  # TODO: reintegrate qasm
                # gotta insert from_patches_to_image() but with the right simulator
                raise NotImplementedError("qasm implementation is currently not supported.")
                # qc.measure_all()
                # # print(self.shots)
                # job = execute(qc, sim, shots=self.shots, seed_simulator=random.randint(0, 100))
                # result = job.result()
                # counts = result.get_counts()
                # for i in counts.keys():
                #     # Conversion from binary to decimal, considering qiskit writes from the right to the left
                #     getcontext().prec = 20
                #     index = int(i[::-1], 2)
                #     p[index] = Decimal(str(counts[i])) / Decimal(str(self.shots))

            elif self.simulator == 'statevector':
                # TODO: ISSUE -> EACH ROW IS THE SAME!!
                resulting_image = from_patches_to_image(quantum_circuit=qc,
                                                        n_tot_qubits=self.n_tot_qubits,
                                                        n_ancillas=self.n_ancilla,
                                                        n_patches=self.n_patches,
                                                        pixels_per_patch=self.pixels_per_patch,
                                                        sim=sim)

            if self.current_gen == 0:
                self.best_solution.append(resulting_image)

            self.candidate_sol.append(resulting_image)
        return self


    @property
    def fitness(self):
        """Evaluates the fitness of quantum circuits using a pre-trained classical NN critic.

        The method calculates the fitness of quantum circuit candidates.
        Handles both noiseless and noisy simulations based on the `simulator` attribute.
        Updates the `fitnesses` list with calculated fitness values for each.

        :return: instance. Self, with updated fitnesses values.
        """
        # Create an empty list to store calculated fitness values
        self.fitnesses = []
        # print(len(self.candidate_sol))
        # print(self.n_children)

        if self.simulator == 'statevector':
            self.noise = False
        if self.noise:
            backend = FakeMumbaiV2()
            sim = AerSimulator.from_backend(backend)
        else:
            sim = Aer.get_backend(self.simulator + '_simulator')

        # if there are more candidates than chosen number of children
        # TODO: why?
        if len(self.candidate_sol) > self.n_children:
            print(self.candidate_sol[0])
            self.best_fitness[-1] = scoring_function(batch_size=self.batch_size,
                                                     critic=self.critic_net,
                                                     # qc=self.ind,
                                                     qc=self.population[0],  # TODO: why
                                                     n_tot_qubits=self.n_tot_qubits,
                                                     n_ancillas=self.n_ancilla,
                                                     n_patches=self.n_patches,
                                                     pixels_per_patch=self.pixels_per_patch,
                                                     sim=sim)

            self.fitness_evaluations += 1
            del self.candidate_sol[0]
            del self.population[0]

        for i in range(len(self.candidate_sol)):  # should this be population instead?
            self.fitnesses.append(scoring_function(batch_size=self.batch_size,
                                                   critic=self.critic_net,
                                                   # qc=self.ind,
                                                   qc=self.population[i],  # TODO: correct?
                                                   n_tot_qubits=self.n_tot_qubits,
                                                   n_ancillas=self.n_ancilla,
                                                   n_patches=self.n_patches,
                                                   pixels_per_patch=self.pixels_per_patch,
                                                   sim=sim))
            print(self.fitnesses)
            self.fitness_evaluations += 1

        if self.current_gen == 0:
            self.best_fitness.append(self.fitnesses[0])

        print('Fitness evaluations: ', self.fitness_evaluations)
        return self

    def multiaction(self):
        """
        It permits the individuals to get more actions in the same generations. Probability of
        assigning multiple action depends on the parameter `multi_action_pb`.

        :return: instance. Self, with updated multi-action count (`counting_multi_action`).
        """

        print('multiaction() called')
        self.counting_multi_action = 0
        rand = random.uniform(0, 1)
        # Question: why set it this way? Repeated many times until random happens to be > m.a.probs.
        # Note: like this, it does not increase counting_multi_action with a probability of
        #  multi_action_pb because it does a series of independent runs. I wrote a code to see
        #  how much it increased (see script-analysis.py).
        while rand < self.multi_action_pb:
            self.counting_multi_action += 1
            rand = random.uniform(0, 1)
        print('multiaction counter: ', self.counting_multi_action)
        return self

    def evolution(self):
        """
        Performs a (1, n_children) evolutionary strategy on quantum circuits to maximize fitness.

        Iterates through generations, applying actions to parent circuits and evaluating offspring fitness.
        Selects the best offspring as the new parent. Adjusts `dtheta` to mitigate local minima and
        adheres to termination criteria based on fitness evaluations or circuit depth.

        Updates attributes like `best_individuals`, `depth`, `best_fitness`, `best_solution`, and others
        during the process.

        :returns: Self, with updated evolutionary process attributes.
        :rtype: instance
        """
        self.best_actions = []  # Check: why is this stored
        action_weights = self.action_weights
        theta_default = self.dtheta
        for g in range(self.n_generations):
            print('\ngeneration:', g)
            if g == 0:
                self.encode().fitness

            else:
                # perform action on parent_ansatz, and then calculate fitness
                self.action().encode().fitness

                # TODO: should this be argmin for my function??
                index = np.argmax(self.fitnesses)  # index of the highest fitness value is
                if self.fitnesses[index] > self.best_fitness[g - 1]:
                    print('improvement found')
                    self.best_individuals.append(self.population[index])
                    self.ind = self.population[index]
                    self.depth.append(self.ind.depth())
                    self.best_fitness.append(self.fitnesses[index])
                    self.best_solution.append(self.candidate_sol[index])
                    for i in range(self.counting_multi_action + 1):
                        self.best_actions.append(self.act_choice[i])

                    self.no_improvements = 0

                else:
                    print('no improvements found')
                    self.no_improvements += 1
                    self.best_individuals.append(self.ind)
                    self.depth.append(self.ind.depth())
                    self.best_fitness.append(self.best_fitness[g - 1])
                    self.best_solution.append(self.best_solution[g - 1])
                print('best qc:\n_qubits', self.ind)
                print('circuit depth:', self.depth[g])
                print('best solution so far:\n_qubits', self.best_solution[g])

                # Add some controls to reduce probabilities to get stuck in local minima: change hyperparameter value
                if self.no_improvements == self.max_gen_no_improvement:
                    print('Dtheta increased to avoid local minima')
                    self.dtheta += 0.1
                    # In principle, we might also increase the multi-action probability: self.multi_action_pb
                elif self.no_improvements == 0:
                    self.dtheta = theta_default
                print('dtheta:', self.dtheta)
                # Termination criteria
                if self.fitness_evaluations == self.n_max_evaluations:
                    break

                if self.max_depth is not None:
                    if self.depth[g] >= self.max_depth:
                        self.action_weights = [0, 20, 0, 80]
                else:
                    self.action_weights = action_weights
                    print('action weights: ', self.action_weights)
            self.current_gen += 1
            print('Number of generations with no improvements: ', self.no_improvements)
            print('best fitness so far: ', self.best_fitness[g])
        print('QES solution: ', self.best_solution[-1])
        return self

    def data(self):
        """ It stores in output all the data required of the algorithm during the evolution"""
        algo = self.evolution()
        self.output = [algo.best_solution, algo.best_individuals[0], algo.best_individuals[-1],
                       algo.depth,
                       algo.best_actions, algo.best_fitness,
                       algo.best_fitness[-1]]
        return self
