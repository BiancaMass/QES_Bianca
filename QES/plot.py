import numpy as np
import pandas as pd
import statistics
import matplotlib.pyplot as plt


def get_quantum_data(function, dim, g, simulator):
    """
    It gets all the quantum data you need for plots
    :param g: garbage qubits
    :param function: string type, function name
    :param dim: int type, dimension
    :return: 5 list type of length 2: best fitness at the last generation on n_iter, final fitnesses on all n_iter,
    mean on n_iter, fitness VS generations of the best iteration
    """
    file_name = 'experiments/'+function+'/' + function + '_dim_' + str(dim)+'_g_'+str(g)+'_'+simulator+'.pkl'
    df = pd.read_pickle(file_name)
    print('Dataframe shape: ', df.shape)
    best_fitness = df['best_fitness'].values.tolist()
    # Converting maximization in minimization
    best_fitness = [-i for i in best_fitness]
    print('All final fitness values:', best_fitness)
    best_overall = min(best_fitness)
    print('Best fitness value over the runs: ', best_overall)
    #mean = statistics.mean(best_fitness[0])
    """print('Mean value over the final fitnesses of all the independent runs: ', mean)
    median = statistics.median(best_fitness[0])
    print('Median value over the final fitnesses of all the independent runs: ', median)
    #stdev = statistics.stdev(best_fitness[0])
    #print('Standard Deviation over the final fitnesses of all the independent runs: ', stdev)
    
    # The following variables are list over the generations
    fitness = df['fitnesses'].iloc[index_best]
    initial_qc = df['initial_qc'].iloc[index_best]
    print('Initial circuit of the best run:\n', initial_qc)
    
    """
    index_best = best_fitness.index(best_overall)
    final_qc = df['final_qc'].iloc[index_best]
    print('Final circuit of the best run:\n', final_qc)
    best_depth = df['depth'].iloc[index_best]
    print('Circuit depth of the best run:', best_depth[-1])
    best_sol = df['best_sol'].iloc[index_best]
    print('Best solutions over generations (best run):', best_sol)

    return index_best, best_overall, best_fitness, final_qc, best_depth


best_run = get_quantum_data(function='track_reconstruction', dim=8, g=1, simulator='statevector')

