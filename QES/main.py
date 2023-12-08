import QuantumES as qes
import fitness_functions as test
import pandas as pd
import os.path


def write(dim, n_iter=10,
          directory='experiments/',
          g=1,
          n_copy=10,
          n_max_evaluations=10000,
          shots=1024,
          simulator='qasm',
          noise=False,
          gpu=False,
          obj_function=test.sphere,
          min_value_gene=-5,
          max_value_gene=5,
          dtheta=0.1,
          action_weights=[50, 10, 10, 30],
          multi_action_pb=0.1,
          max_gen_no_improvements=20, **kwargs):
    """
    It writes a pickle file with 6 columns and n_iter rows.
    Each row represent an independent run, while columns are, in order: the best solutions, the best quantum circuits,
    the relative depth, the actions made on the circuits, the best fitness values and the final fitness value found.
    All those variable are list over the generations.

    :param directory: string. Path where the file has to be saved
    :param n_iter: integer. Number of independent runs of the algorithm
    :param dim: integer. Problem size
    :param g: integer. Number of garbage qubits
    :param n_copy: integer. Number of individuals generated at each iteration of the evolution strategy
    :param n_max_evaluations: integer. Termination criteria over the number of fitness evaluations
    :param shots: integer. Number of executions on a quantum circuit to get the probability distribution
    :param simulator: string. statevector or qasm
    :param noise: Boolean. True if a noisy simulation is required, False otherwise
    :param gpu: True or False. If True, it simulates quantum circuits on GPUs, otherwise it does not
    :param obj_function: string. Name of the objective function to minimize
    :param min_value_gene: float. Lower bound on the domain of the objective function
    :param max_value_gene: float. Upper bound on the domain of the objective function
    :param dtheta: float. Maximum displacement for the angle parameter in the mutation action
    :param action_weights: list of four integers summing to 100
    :param multi_action_pb: float. Probability to get multiple actions in the same generation
    :param max_gen_no_improvements: integer. Maximum number of generations with no improvements,
                                    then some changes will be applied

    :keyword max_depth: integer. It fixes an upper bound on the quantum circuits depth"""

    data = []
    for depth in kwargs.values():
        max_depth = depth
    for i in range(n_iter):
        print('independent run number:', i)
        data.append(qes.Qes(dim=dim, g=g, n_copy=n_copy, n_max_evaluations=n_max_evaluations, shots=shots,
                            simulator=simulator, noise=noise, gpu=gpu, obj_function=obj_function,
                            min_value_gene=min_value_gene,
                            max_value_gene=max_value_gene, dtheta=dtheta, action_weights=action_weights,
                            multi_action_pb=multi_action_pb,
                            max_gen_no_improvement=max_gen_no_improvements, max_depth=max_depth).data().output)
        df = pd.DataFrame(data, columns=['best_sol', 'initial_qc', 'final_qc', 'depth', 'best_actions', 'fitnesses',
                                         'best_fitness'])
        obj_function_name = test.get_variable_name(obj_function)
        file_name = os.path.join(directory, obj_function_name + '/' + obj_function_name + '_dim_' + str(dim) + '_g_' +
                                 str(g) + '_' + simulator + '.pkl')
        df.to_pickle(os.path.join(file_name))
    return print('file .pkl saved')


DIM = [80]
for d in DIM:
    write(dim=d, max_depth=50)
