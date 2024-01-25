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

np.random.seed(123)


class Qes:
    """
    Hybrid quantum-classical optimization technique, implemented as maximization, for real
    functions of real variables.
    """

    def __init__(self, n_qubits, g, n_copy, n_max_evaluations, shots, simulator, noise, gpu,
                 obj_function,
                 min_value_gene, max_value_gene, dtheta, action_weights, multi_action_pb,
                 max_gen_no_improvement,
                 **kwargs):
        """ Initialization of the population and its settings
        # :param dim: integer. Problem (function) size
        :param n_qubits: integer. Number of qubits for the circuit
        :param g: integer. Number of garbage qubits
        :param n_copy: integer. Number of individuals generated at each iteration of the evolution strategy
        :param n_max_evaluations: integer. Termination criteria over the number of fitness evaluations
        :param shots: integer. Number of executions on a quantum circuit to get the probability distribution
        :param simulator: string. Qiskit simulator either statevector or qasm (statevector is
                          without noise)
        :param noise: Boolean. True if a noisy simulation is required, False otherwise
        :param gpu: True or False. If True, it simulates quantum circuits on GPUs, otherwise it does not
        :param obj_function: string. Name of the objective function to minimize   # TODO: change
        :param min_value_gene: float. Lower bound on the domain of the objective function   # TODO: change
        :param max_value_gene: float. Upper bound on the domain of the objective function   # TODO: change
        :param dtheta: float. Maximum displacement for the angle parameter in the mutation action
        :param action_weights: list. Probab (x100) to choose between the 4 possible actions (their sum must be 100)
        :param multi_action_pb: float. Probability to get multiple actions in the same generation
        :param max_gen_no_improvement: integer. Number of generations with no improvements after which some changes will be applied
        :keyword max_depth: integer. It fixes an upper bound on the quantum circuits depth"""

        # self.dim = dim  #Note: referred to the function size, used to calculate num qubits
        # Note: now I am directly inputting number of qubits, but I could calculate it based on
        #  the image size, then I would just input the image dimensions for example.
        self.n_qubits = n_qubits
        self.g = g
        self.n_copy = n_copy  # number of children
        self.n_max_evaluations = n_max_evaluations
        self.shots = shots
        self.simulator = simulator
        self.gpu = gpu
        self.noise = noise
        self.obj_function = obj_function  # TODO: change
        self.min_value_gene = min_value_gene  # TODO: change
        self.max_value_gene = max_value_gene  # TODO: change
        self.dtheta = dtheta
        self.action_weights = action_weights
        self.multi_action_pb = multi_action_pb
        self.max_gen_no_improvement = max_gen_no_improvement + 1

        # Number of generations of the classical evolutionary strategy (integer)
        self.n_gen = math.ceil(n_max_evaluations / n_copy)

        #######################
        # CREATE THE 0-TH INDIVIDUAL (QUANTUM CIRCUIT)
        #######################

        # Calculate the number of qubits (integer) for the circuit
        # Note: currently an argument of the  function
        # self.n_qubits = math.ceil(math.log(self.dim, 2)) + self.g

        # Number of the computational basis states in the n_qubits-qubits Hilbert space (integer)
        self.N = 2 ** self.n_qubits

        ### CIRCUIT ###
        qr = QuantumRegister(self.n_qubits, 'qubit')
        # cr = ClassicalRegister(self.n_qubits, 'bit')
        ind = QuantumCircuit(qr)

        # Initialize individual by applying H gate on each qubit
        for qbit in range(self.n_qubits):
            ind.h(qbit)
        # Add a random RY rotation gate on a random qubit to break the initial symmetry
        # Note: consider not doing this in early stage of development
        ind.ry(random.random() * 2 * math.pi, random.randint(0, self.n_qubits - 1))

        # Best individual in the CURRENT generation
        self.ind = ind  # a circuit configuration - initialized as the basic circuit
        # Population (quantum circuits) generated in the current generation from the best qc of
        # the previous one
        self.population = [self.ind]
        # Candidate solution (real-valued vectors) in the current generation
        self.candidate_sol = None
        # Fitness values (real values) of the candidate solutions in the current generation
        self.fitnesses = None
        # Actions taken in the current generation
        self.act_choice = None

        # Best individuals (quantum circuits) over ALL the generations
        # At initialization it is the 0-th circuit
        self.best_individuals = [ind]
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

        print('Initial quantum circuit:', self.ind)

    def action(self):
        """ It generates n_copy of the individual and apply one of the 4 POSSIBLE ACTIONS(add,
        delete, swap, mutate) on each of them. Then the new quantum circuits are stored in the
        attribute 'population' """

        population = []
        # Copy and modify the circuit as many times as you chose your branching factor to be (
        # i.e., number of children)
        for i in range(self.n_copy):
            qc = self.ind.copy()  # Current child (copy)
            if self.max_depth is not None:  # if user gave a max_depth as input argument
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
                                             k=counter)
            angle = random.random() * 2 * math.pi
            gate_list = [qc.rx, qc.ry, qc.rz, qc.rxx, qc.ryy, qc.rzz]
            gate_dict = {'rx': RXGate, 'ry': RYGate, 'rz': RZGate,
                         'rxx': RXXGate, 'ryy': RYYGate, 'rzz': RZZGate}
            position = 0
            print('action choice:', self.act_choice, 'for the copy number: ', i)

            for j in range(counter):

                # Add a random gate on a random qubit at the end of the parent quantum circuit
                if self.act_choice[j] == 'A':
                    position = random.sample([i for i in range(len(qc.qubits))], k=2)
                    choice = random.randint(0, len(gate_list) - 1)
                    if 0 <= choice < 3:
                        gate_list[choice](angle, position[0])
                    else:
                        gate_list[choice](angle, position[0], position[1])

                # Delete a random gate in a random position of the parent quantum circuit
                elif self.act_choice[j] == 'D':
                    position = random.randint(0, len(qc.data) - 1)
                    qc.data.remove(qc.data[position])

                # Remove a gate in a random position and replace it with a new gate randomly chosen
                elif self.act_choice[j] == 'S':
                    if len(qc.data) - 1 > 0:
                        position = random.randint(0, len(qc.data) - 2)
                    gate_to_remove = qc.data[position][0]
                    gate_to_add = random.choice(list(gate_dict.values()))(angle)
                    while gate_to_add.name == gate_to_remove.name:
                        gate_to_add = random.choice(list(gate_dict.values()))(angle)
                    if gate_to_add.name == 'rzz' or gate_to_add.name == 'rxx' or gate_to_add.name == 'ryy':
                        n_qubits = 2
                    else:
                        n_qubits = 1
                    lenght = len(qc.data[position][1])
                    if lenght == n_qubits:
                        element_to_remove = list(qc.data[position])
                        element_to_remove[0] = gate_to_add
                        element_to_add = tuple(element_to_remove)
                        qc.data[position] = element_to_add
                    elif lenght > n_qubits:
                        element_to_remove = list(qc.data[position])
                        element_to_remove[0] = gate_to_add
                        element_to_remove[1] = [random.choice(qc.data[position][1])]
                        element_to_add = tuple(element_to_remove)
                        qc.data[position] = element_to_add
                    elif lenght < n_qubits:
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

                # Choose a gate and change the angle by adding a value between
                # [theta-dtheta, theta+dtheta]
                elif self.act_choice[j] == 'M':
                    to_not_select = 'h'  # Note: in current model only Hadamard =! rotation gate
                    check = True
                    gate_to_mute = None

                    while check:
                        position = random.choice([i for i in range(len(qc.data))])
                        gate_to_mute = qc.data[position]

                        if gate_to_mute[0].name != to_not_select:
                            check = False
                    angle_new = qc.data[position][0].params[0] + random.uniform(0, self.dtheta)
                    element_to_mute = list(gate_to_mute)
                    element_to_mute[0] = gate_dict[gate_to_mute[0].name](angle_new)
                    element_to_add = tuple(element_to_mute)
                    qc.data[position] = element_to_add

                # In case of multiactions we are appending more circuits to the population,
                # if you don't want that put the next code line outside of the for loop on counter
            population.append(qc)
        self.population = population
        return self

    def encode(self):  # TODO: adapt to your application
        """
        It transforms a quantum circuit (ind) in a string of real values of length 2^N, where N=len(ind).
        """
        # N = number of computational basis states (2**n)
        # dim e' il numero di dimensioni della funzione
        # prima era che n_qubits = log(dim)
        t = self.N - self.dim  # question: still not super sure why this is calculated
        if self.simulator == 'statevector':
            self.noise = False
        if self.noise:
            raise NotImplementedError("Noise implementation is currently not supported.")
            # backend = FakeMumbaiV2()
            # sim = AerSimulator.from_backend(backend)
        else:
            sim = Aer.get_backend(self.simulator + '_simulator')
        # Setup Gpu
        if self.gpu:
            sim.set_options(device='GPU')
            print('gpu used')

        self.candidate_sol = []
        # Let qasm be more free because of the shot noise
        if self.simulator == 'qasm':
            # Comment out and comment in the other 2 lines if you want to support noise simulation
            if self.noise:
                raise NotImplementedError("Noise implementation is currently not supported.")
            # if self.no_improvements > 10:
            #     self.population.insert(0, self.best_individuals[-1])

        for j in range(len(self.population)):
            qc = self.population[j].copy()
            p, individual = np.zeros(self.N), np.zeros(self.N)

            # Set up the type of simulator we want to use
            if self.simulator == 'qasm':
                raise NotImplementedError("Noise implementation is currently not supported.")
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

            # Noise-free simulation
            elif self.simulator == 'statevector':
                # Execute the circuit `qc` on the simulator `sim`
                job = execute(qc, sim)  # Returns a job object (the task of running the circuit on the simulator)
                result = job.result()  # Retrieves the result of the execution
                # Extract the state vector of the quantum circuit after its execution
                statevector = result.get_statevector(qc)
                # Calculate probabilities
                for i in range(len(np.asarray(statevector))):  # iterate over each element of s.v.
                    # For each element in the state vector, calculate the probability of the
                    # corresponding quantum state (the square of the abs. value of its amplitude)
                    p[i] = np.absolute(statevector[i]) ** 2  # store probs in array `p`

            # Apply the 'linear' map between [0,1] and [min_value_gene, max_value_gene]
            for i in range(self.N):
                # question: why?
                if p[i] > 1 / self.dim:
                    p[i] = (1 / self.dim)
                # individual[i] = ((p[i]) * (self.max_value_gene - self.min_value_gene) * (
                #         self.N - t)) + self.min_value_gene

            if self.current_gen == 0:
                self.best_solution.append(individual[:self.dim])

            self.candidate_sol.append(individual[:self.dim])
        return self


    @property
    def fitness(self):
        """
        It creates the fitness evaluation function for candidate solutions and store it in the attribute .fn
        """
        # Create an empty list to store calculated fitness values
        self.fitnesses = []
        # print(len(self.candidate_sol))
        # print(self.n_copy)

        # if there are more candidates than chosen number of children
        # Question: why is this if statement needed? Couldn't it be just the one below?
        if len(self.candidate_sol) > self.n_copy:
            self.best_fitness[-1] = self.obj_function(self.candidate_sol[0])  # TODO change obj.fun.
            self.fitness_evaluations += 1
            del self.candidate_sol[0]
            del self.population[0]

        for i in range(len(self.candidate_sol)):
            self.fitnesses.append(self.obj_function(self.candidate_sol[i]))  # TODO: change obj.fun.
            self.fitness_evaluations += 1

        if self.current_gen == 0:
            self.best_fitness.append(self.fitnesses[0])

        print('Fitness evaluations: ', self.fitness_evaluations)
        return self

    def multiaction(self):
        """ It permits the individuals to get more actions in the same generations depending
        on given parameter multi-action probability (multi_action_pb)"""
        self.counting_multi_action = 0
        rand = random.uniform(0, 1)
        # Question: why set it this way? Repeated many times until random happens to be > m.a.probs.
        # Note: like this, it does not inrease counting_multi_action with a probability of
        #  multi_action_pb because it does a series of independent runs. I wrote a code to see
        #  how much it increased (see script-analysis.py).
        while rand < self.multi_action_pb:
            self.counting_multi_action += 1
            rand = random.uniform(0, 1)
        # print('multiaction counter: ', self.counting_multi_action)
        return self

    # TODO: unpack this function
    def evolution(self):
        """
        Evolutionary Strategy (1,n_copy) over quantum circuits. Maximization
        """
        self.best_actions = []
        action_weights = self.action_weights
        theta_default = self.dtheta
        for g in range(self.n_gen):
            print('\ngeneration:', g)
            if g == 0:
                self.encode().fitness

            else:
                self.action().encode().fitness

                index = np.argmax(self.fitnesses)
                # print('Fitness:',self.best_fitness)
                # print('Individuals:', self.best_individuals)
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
                    # print('no improvements')
                    self.no_improvements += 1
                    self.best_individuals.append(self.ind)
                    self.depth.append(self.ind.depth())
                    self.best_fitness.append(self.best_fitness[g - 1])
                    self.best_solution.append(self.best_solution[g - 1])
                # print('best qc:\n_qubits', self.ind)
                print('circuit depth:', self.depth[g])
                # print('best solution so far:\n_qubits', self.best_solution[g])

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