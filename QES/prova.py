import QuantumES as qes
import fitness_functions as test


dim = 27
n_iter = 1
directory = 'experiments/'
g = 2
n_copy = 12
n_max_evaluations = 20000
shots = 100000
simulator = 'statevector'
noise = False
gpu = False
obj_function = test.track_reconstruction
min_value_gene = 0.1
max_value_gene = 0.9
dtheta = 0.1
action_weights = [50, 10, 10, 30]
multi_action_pb = 0.1
max_gen_no_improvements = 10
max_depth = 20

qes.Qes(dim=dim, g=g, n_copy=n_copy, n_max_evaluations=n_max_evaluations, shots=shots,
        simulator=simulator, noise=noise, gpu=gpu, obj_function=obj_function,
        min_value_gene=min_value_gene,
        max_value_gene=max_value_gene, dtheta=dtheta, action_weights=action_weights,
        multi_action_pb=multi_action_pb,
        max_gen_no_improvement=max_gen_no_improvements, max_depth=max_depth).data().output

print('Real solution: ', test.track_reconstruction(test.sol_c))
print('Real solution: ', test.sol_c)

