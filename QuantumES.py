import numpy as np
import pandas as pd
import random
import math
import os.path
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, execute, Aer
from qiskit.circuit.library import RYGate, RXGate, RZGate, RXXGate, RYYGate, RZZGate


class Qes:
    """
    Hybrid quantum-classical optimization technique for real functions of real variables
    """
    def __init__(self, dim, g, n_copy, n_gen, shots, simulator, gpu, obj_function, min_value_gene, max_value_gene,
                 dtheta, action_weights, multi_action_pb):
        """ Initialization of the poipulation and its properties
        :param dim: integer. Problem size
        :param g: integer. Number of garbage qubits
        :param n_copy: integer. Number of individuals generated at each iteration of the evolution strategy
        :param n_gen: integer. Number of generations
        :param shots: integer. Number of executions on a quantum circuit to get the probability distribution
        :param simulator: string.
        :param gpu: True or False. If True, it simulates quantum circuits on GPUs, otherwise it does not
        :param obj_function: string. Name of the objective function to minimize
        :param min_value_gene: float. Lower bound on the domain of the objective function
        :param max_value_gene: float. Upper bound on the domain of the objective function
        :param dtheta: float. Maximum displacement for the angle parameter in the mutation action
        :param action_weights: list of four integers summing to 100. """

        self.dim = dim
        self.g = g
        self.n_copy = n_copy
        self.n_gen = n_gen
        self.shots = shots
        self.simulator = simulator
        self.gpu = gpu
        self.obj_function = obj_function
        self.min_value_gene = min_value_gene
        self.max_value_gene = max_value_gene
        self.dtheta = dtheta
        self.action_weights = action_weights
        self.multi_action_pb = multi_action_pb
        # Number of Qubits required
        self.n = math.ceil(math.log(self.dim, 2)) + self.g
        self.N = 2 ** self.n
        # Initialize individual by applying H gate on each qubit and a random RY rotation gate on a random qubit
        # to break the initial symmetry.
        qr = QuantumRegister(self.n, 'qubit')
        ind = QuantumCircuit(qr)
        for _ in range(self.n):
            ind.h(_)
        ind.ry(random.random() * 2 * math.pi, random.randint(0, self.n - 1))
        self.ind = ind
        self.depth = [self.ind.depth()]
        # Best individual (quantum circuit), best_sol (real-valued vector) anf fitness value of the current generation
        self.best_individual = [ind]
        self.best_solution = []
        self.best_fitness = []
        self.best_actions = None

        # population (quantum circuits), candidate_sol (real-valued vectors), fitness  of the current generation
        self.population = None
        self.candidate_sol = None
        self.fitnesses = None
        self.act_choice = None

        # Useful Attributes
        self.no_improvements = 0
        self.fitness_evaluations = 0
        self.current_gen = 0

    def action(self):
        """ It generates n_copy of the individual and apply one of the 4 POSSIBLE ACTIONS(add, delete, swap, mutate) on
        each of them. Then the new quanrum circuits are stored in the attribute'population' """
        self.act_choice = random.choices(['A', 'D', 'S', 'M'], weights=self.action_weights, k=self.n_copy)
        population = []
        for i in range(self.n_copy):
            qc = self.ind.copy()

            angle = random.random() * 2 * math.pi
            gate_list = [qc.rx, qc.ry, qc.rz, qc.rxx, qc.ryy, qc.rzz]
            gate_dict = {'rx': RXGate, 'ry': RYGate, 'rz': RZGate,
                         'rxx': RXXGate, 'ryy': RYYGate, 'rzz': RZZGate}
            position = 0

            if self.act_choice[i] == 'A':
                "It adds a random gate on a random qubit at the end of the parent quantum circuit"
                position = random.sample([i for i in range(len(qc.qubits))], k=2)
                choice = random.randint(0, len(gate_list) - 1)
                if 0 <= choice < 3:
                    gate_list[choice](angle, position[0])
                else:
                    gate_list[choice](angle, position[0], position[1])
                population.append(qc)

            elif self.act_choice[i] == 'D':
                "It deletes a random gate in a random position of the parent quantum circuit"
                position = random.randint(0, qc.depth() - 1)
                qc.data.remove(qc.data[position])
                population.append(qc)

            elif self.act_choice[i] == 'S':
                "It removes a gate in a random position and replace it with a new gate randomly chosen"
                if len(qc.data) - 1 > 0:
                    position = random.randint(0, len(qc.data) - 1)
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
                population.append(qc)

            elif self.act_choice[i] == 'M':
                "It chooses a gate and changes the angle by adding a value between [theta-dtheta,theta+dtheta]"
                to_not_select = 'h'
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
                population.append(qc)
        self.population = population
        return self

    def encode(self):
        """
        It transforms a quantum circuit (ind) in a string of real values of length 2^N, where N=len(ind).
        """
        t = self.N - self.dim
        sim = Aer.get_backend(self.simulator + '_simulator')
        self.candidate_sol = []
        if self.current_gen == 0:
            self.population = [self.ind]
        for j in range(len(self.population)):
            qc = self.population[j]
            p, individual = np.zeros(self.N), np.zeros(self.N)
            # Setup Gpu
            if self.gpu:
                sim.set_options(device='GPU')
            # Set up the type of simulator we want to use
            if self.simulator == 'qasm':
                qc.measure_all()
                job = execute(qc, sim, shots=self.shots, seed_simulator=random.randint(1, 150))
                result = job.result()
                counts = result.get_counts()
                for i in counts.keys():
                    index = int(i[::-1], 2)
                    print(i)
                    p[index] = counts[i] / self.shots
            elif self.simulator == 'statevector':
                job = execute(qc, sim)
                result = job.result()
                statevector = result.get_statevector(qc)
                for i in range(len(np.asarray(statevector))):
                    p[i] = np.absolute(statevector[i]) ** 2
            # Apply the 'linear' map between [0,1] and [min_value_gene, max_value_gene]
            for i in range(self.N):
                if p[i] > 1 / self.dim:
                    p[i] = (1 / self.dim)
                individual[i] = ((p[i])*(self.max_value_gene-self.min_value_gene)*(self.N-t))+self.min_value_gene
            if self.current_gen == 0:
                self.best_solution.append(individual[:self.dim])
            else:
                self.candidate_sol.append(individual[:self.dim])
        return self

    @property
    def fitness(self):
        """
        It creates the fitness evaluation function for candidate solutions and store it in the attribute .fn
        """
        self.fitnesses = []
        if self.current_gen == 0:
            self.candidate_sol = self.best_solution
            print(self.candidate_sol)
        for j in range(len(self.candidate_sol)):
            fn = 0
            if self.obj_function == 'sphere':
                fn = sum(-gene * gene for gene in self.candidate_sol[j])
            elif self.obj_function == 'schwefel':
                self.min_value_gene = -65.536
                self.max_value_gene = 65.536
                for i in range(len(self.candidate_sol[j])):
                    fn += -sum([gene ** 2 for gene in self.candidate_sol[j][:i]])
            elif self.obj_function == 'ackley':
                self.min_value_gene = -32.768
                self.max_value_gene = 32.768
                A = sum(gene * gene for gene in self.candidate_sol[j])
                B = sum(math.cos(2 * math.pi * gene) for gene in self.candidate_sol[j])
                fn = 20 * math.exp(-0.2*math.sqrt(A/self.dim))+math.exp(B/self.dim)-20-math.e
            elif self.obj_function == 'rastrigin':
                for i in range(self.dim):
                    fn += -(10+self.candidate_sol[j][i]**2-10*math.cos(2*self.candidate_sol[j][i]*math.pi))
            elif self.obj_function == 'rosenbrock':
                for i in range(self.dim - 1):
                    fn += -((100 * (self.candidate_sol[j][i + 1] - self.candidate_sol[j][i] ** 2) ** 2) +
                            (self.candidate_sol[j][i] - 1) ** 2)
            if self.current_gen == 0:
                self.best_fitness.append(fn)
            else:
                self.fitnesses.append(fn)

            self.fitness_evaluations += 1
        return self

    def multiaction(self):
        """ It permits the individuals to get more actions in the same generations depending on multi_action_pb"""
        # Add multiple actions with probability multi_action_pb
        rand = random.uniform(0, 1)
        while rand < self.multi_action_pb:
            self.action()
            rand = random.uniform(0, 1)
        return self

    def evolution(self):
        """
        Evolutionary Strategy (1,n_copy) over quantum circuits
        """
        self.best_actions = []
        theta = self.dtheta
        for g in range(self.n_gen):
            print('generation:', g)
            if g == 0:
                self.encode().fitness
            else:
                self.action().multiaction().encode().fitness
                index = np.argmax(self.fitnesses)

                if self.fitnesses[index] > self.best_fitness[g-1]:
                    print('improvement found')
                    self.best_individual.append(self.population[index])
                    self.ind = self.population[index]
                    self.depth.append(self.ind.depth())
                    print('circuit depth:\n', self.depth[g-1])
                    print('best qc:\n', self.population[index])
                    self.best_fitness.append(self.fitnesses[index])
                    print('best fitness:\n', self.fitnesses[index])
                    self.best_solution.append(self.candidate_sol[index])
                    print('best solution:\n', self.candidate_sol[index])
                    self.best_actions.append(self.act_choice[index])
                    print('best action:\n', self.act_choice[index])
                    # Counter for number of generations from the last improvement found
                    self.no_improvements = 0
                else:
                    self.no_improvements += 1
                    self.depth.append(self.ind.depth())
                    self.best_fitness.append(self.best_fitness[g-1])
                    self.best_solution.append(self.best_solution[g-1])
                    self.best_individual.append(self.ind
                                                )
                # Add some controls to reduce getting stuck in local minima: change hyperparameter value
                if self.no_improvements > self.n_gen/10:
                    print('no improvements')
                    self.dtheta += 0.1
                else:
                    self.dtheta = theta
                # Termination criteria
                if self.fitness_evaluations == 100000:
                    break

            self.current_gen += 1
        print(len(self.best_fitness), len(self.best_solution), len(self.best_individual),len(self.best_actions))
        print('Fitness over generations:', self.best_fitness)
        return self


def write(directory, n_iter, dim, g, n_copy, n_gen, shots, simulator, gpu, obj_function, min_value_gene, max_value_gene,
          dtheta, action_weights, multi_action_pb):

    data = []
    for i in range(n_iter):
        algo = Qes(dim=dim, g=g, n_copy=n_copy, n_gen=n_gen, shots=shots, simulator=simulator, gpu=gpu,
                   obj_function='sphere', min_value_gene=min_value_gene, max_value_gene=max_value_gene,
                   dtheta=dtheta, action_weights=action_weights, multi_action_pb=multi_action_pb).evolution()
        data.append([algo.best_solution, algo.best_individual, algo.best_actions, algo.best_fitness, algo.depth])
        df = pd.DataFrame(data, columns=['best_ind', 'best_qc', 'best_actions', 'fitnesses', 'depth'])
        df.to_pickle(os.path.join(directory, obj_function+'_dim_'+str(dim)+'_g_'+str(g)+'_'+simulator+'.pkl'))
    return print('file saved')


write(directory='experiments', n_iter=2, dim=10, g=1, n_copy=4, n_gen=10, shots=100000, simulator='statevector',
      gpu=False, obj_function='sphere', min_value_gene=-5, max_value_gene=5, dtheta=0.1,
      action_weights=[50, 10, 10, 30], multi_action_pb=0.1)
#Qes(dim=10, g=1, n_copy=4, n_gen=100, shots=1000, simulator='statevector', gpu=False, obj_function='sphere', min_value_gene=-5, max_value_gene=5,
 #dtheta=0.1, action_weights=[25,25,25,25], multi_action_pb=0.2).evolution()