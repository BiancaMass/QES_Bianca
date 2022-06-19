import random
import numpy
import math
import pandas as pd
import matplotlib.pyplot as plt
import ast
import os.path
import json
from statistics import mean
from deap import base
from deap import creator
from deap import tools
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, execute, BasicAer
from qiskit.providers.aer import AerSimulator
from qiskit.test.mock import FakeMontreal
from qiskit import IBMQ, transpile

#IBMQ.enable_account('3a1e44c417a24cfd7ad48ef3a7f9580be5b94fb0964c798bef468426bc6f6961761a3084e081b884a443d40c445497863'
                    #'fdd687953f085b8a95e67d1ca5cce70')


def createToolbox(ind_size):
    # Creating Classes
    creator.create('FitnessMax', base.Fitness, weights=(1.0,))
    creator.create('Individual', list, fitness=creator.FitnessMax)
    toolbox = base.Toolbox()
    toolbox.register('attr_float', random.uniform, a=-5, b=5)
    toolbox.register('individual', tools.initRepeat, creator.Individual, toolbox.attr_float,
                     n=ind_size)
    toolbox.register('population', tools.initRepeat, list, toolbox.individual)
    toolbox.register("population_guess", initPopulation, list, creator.Individual, "Best_40/pop_init.json")
    ''' The crossover operator will be add in the update function, which have the possibility to choose between 
    classic and quantum ones'''
    # Mutation
    toolbox.register('mutate', tools.mutGaussian, mu=0.0, sigma=0.2, indpb=0.2)
    # Selection
    toolbox.register('select_RWS', tools.selRoulette)
    return toolbox


def initPopulation(pcls, ind_init, filename):
    with open(filename, "r") as pop_file:
        contents = json.load(pop_file)
    return pcls(ind_init(c) for c in contents)


def quartile1(x):
    return numpy.percentile(x, q=25)


def quartile3(x):
    return numpy.percentile(x, q=75)


def createStats():
    # Statistical Features
    stats = tools.Statistics(key=lambda ind: ind.fitness.values)
    stats.register("avg", numpy.mean)
    stats.register("std", numpy.std)
    stats.register("min", numpy.min)
    stats.register("max", numpy.max)
    stats.register("q1", quartile1)
    stats.register("q3", quartile3)
    return stats


def sphere(individual):
    if isinstance(individual, list):
        x = individual[:]
    else:
        x = individual.values[:]
    f = 0
    for i in range(len(x)):
        f += -x[i] ** 2
    return f


def schwefel(individual):
    if isinstance(individual, list):
        x = individual[:]
    else:
        x = individual.values[:]
    f = 0
    for i in range(len(x)):
        x2 = [elem ** 2 for elem in x[:i]]
        f += -sum(x2)
    return f


def rastrigin(individual):
    if isinstance(individual, list):
        x = individual[:]
    else:
        x = individual.values[:]
    f = 0
    for i in range(len(x)):
        f += -(10 + x[i] ** 2 - 10 * math.cos(2 * x[i] * math.pi))
    return f


def rosenbrock(individual):
    if isinstance(individual, list):
        x = individual[:]
    else:
        x = individual.values[:]
    f = 0
    for i in range(len(x) - 1):
        f += -(100 * (x[i + 1] - x[i] ** 2) ** 2 + (x[i] - 1) ** 2)
    return f


def ackley(individual):
    if isinstance(individual, list):
        x = individual[:]
    else:
        x = individual.values[:]
    n = len(x)
    A, B = 0, 0
    for i in range(n):
        A += x[i] ** 2
        B += math.cos(2 * math.pi * x[i])
    f = 20 * math.exp(-0.2 * math.sqrt(A / n)) + math.exp(B / n) - 20 - math.e
    return f


def checkBounds(min, max):
    def decorator(func):
        def wrapper(*args, **kargs):
            offspring = func(*args, **kargs)
            for child in offspring:
                for i in range(len(child)):
                    if child[i] > max:
                        child[i] = max
                    elif child[i] < min:
                        child[i] = min
            return offspring

        return wrapper

    return decorator


def split(word):
    return [char for char in word]


def Quantum_Crossover(parent1, parent2, backend_name, best, beta=None):
    '''if beta < 0 or beta > math.pi / 2:
        raise 'Value Error: beta must be in [0, pi/2]'''
    n = int(len(parent1))
    d1, d2 = [], []
    for i in range(n):
        d1.append(abs(parent1[i] - best[i]))
        d2.append(abs(parent2[i] - best[i]))

    parents = QuantumRegister(n, 'parents')
    choice = ClassicalRegister(n, 'children')
    qc = QuantumCircuit(parents, choice)
    for i in range(n):
        qc.h(parents[i])
        if d1[i] == 0 and d2[i] == 0:
            beta = 0
        else:
            beta = (d1[i] - d2[i])/(d1[i]+d2[i])
        if d1[i] >= d2[i]:
            qc.ry(beta, parents[i])
        else:
            qc.ry(-beta, parents[i])
        qc.measure(parents[i], choice[i])

    if backend_name == 'fake':
        # Ideal Simulator without noise
        backend = BasicAer.get_backend('qasm_simulator')
        job = execute(qc, backend, shots=1, seed_simulator=random.randint(1, 150))
        result = job.result()
        counts = result.get_counts()

    elif backend_name == 'fake_noise':
        # Simulator with noise
        backend = FakeMontreal()
        sim_ideal = AerSimulator()
        result = sim_ideal.run(transpile(qc, sim_ideal)).result()
        counts = result.get_counts(0)

    else:

        provider = IBMQ.get_provider(hub='ibm-q', group='open', project='main')
        backend = provider.get_backend(backend_name)
        job = execute(qc, backend, shots=1, seed_simulator=random.randint(1, 150))
        result = job.result()
        counts = result.get_counts()

    bit = 0
    for i in counts.keys():
        bit = i
    choices = split(bit)
    child1, child2 = [], []
    for i in range(n):
        if choices[i] == '0':
            child1.append(parent1[i])
            child2.append(parent2[i])
        elif choices[i] == '1':
            child1.append(parent2[i])
            child2.append(parent1[i])
        else:
            raise 'Crossover Operator Error: bit\'s value does not possible'
    return child1, child2


def updatedGA(toolbox, pop_size, cxpb, mutpb, n_gen, cx_operator, test, stats, num_marked=5, hof=tools.HallOfFame(1),
              n_iter=None, pop_init=None, beta=None, backend=None, verbose=False):
    # Creating Crossover Operator
    if cx_operator == 'onep':
        toolbox.register('mate', tools.cxOnePoint)
    elif cx_operator == 'twop':
        toolbox.register('mate', tools.cxTwoPoint)
    elif cx_operator == 'uniform':
        toolbox.register('mate', tools.cxUniform, indpb=0.2)
    elif cx_operator == 'blend':
        toolbox.register('mate', tools.cxBlend, alpha=0.5)
    elif cx_operator == 'quantum_crossover':
        toolbox.register('mate', Quantum_Crossover)
    elif cx_operator == 'quantum_crossover_noise':
        toolbox.register('mate', Quantum_Crossover)
    else:
        raise 'Error: the selected crossover operator is not available'

    # Choosing test function
    if test == 'sphere':
        toolbox.register('evaluate', sphere)
        # toolbox.decorate("mate", checkBounds(-5.12, 5.12))
        toolbox.decorate("mutate", checkBounds(-5.12, 5.12))
    elif test == 'rastrigin':
        toolbox.register('evaluate', rastrigin)
        # toolbox.decorate("mate", checkBounds(-5.12, 5.12))
        toolbox.decorate("mutate", checkBounds(-5.12, 5.12))
    elif test == 'schwefel':
        toolbox.register('evaluate', schwefel)
        # toolbox.decorate("mate", checkBounds(-65.536, 65.536))
        toolbox.decorate("mutate", checkBounds(-65.536, 65.536))
    elif test == 'rosenbrock':
        toolbox.register('evaluate', rosenbrock)
        # toolbox.decorate("mate", checkBounds(-5.12, 5.12))
        toolbox.decorate("mutate", checkBounds(-5.12, 5.12))
    elif test == 'ackley':
        toolbox.register('evaluate', ackley)
        # toolbox.decorate("mate", checkBounds(-5.12, 5.12))
        toolbox.decorate("mutate", checkBounds(-32.768, 32.768))
    else:
        raise 'Error: the selected test function is not available'

    # Creating the population
    if pop_init == None:
        pop = toolbox.population(n=pop_size)
    elif pop_init == 'mine':
        toolbox.register("population_guess", initPopulation, list, creator.Individual,
                         '5_bits/nobetatuning/Init_pop/'+test+str(n_iter)+'.json')
        pop = toolbox.population_guess()

    else:
        pop = toolbox.clone(pop_init)
    pop_init = toolbox.clone(pop)

    # Defining the Logbook
    logbook = tools.Logbook()
    logbook.header = ["gen", "best pop"] + (stats.fields if stats else [])

    # Evaluate the entire population
    fitness = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitness):
        ind.fitness.values = [fit]

    hof.update(pop) if stats else {}

    record = stats.compile(pop) if stats else {}
    logbook.record(gen=0, **record, best_pop=pop_init)
    if verbose:
        print('Logbook:', logbook.stream)

    for g in range(n_gen):
        print('generazione:', g)
        # elitism
        bests = toolbox.clone(tools.selBest(pop, num_marked))
        elitist = bests[0]

        # Select the next generation individuals
        offspring = toolbox.select_RWS(pop, k=pop_size)

        # Clone the selected individuals
        offspring = list(map(toolbox.clone, offspring))

        # Apply crossover and mutation on the offspring
        # Even indices: [::2], Odd indices: [1::2]
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < cxpb:
                if cx_operator == 'quantum_crossover' or cx_operator == 'quantum_crossover_noise':
                    toolbox.mate(child1, child2, backend_name=backend, beta=beta, best=elitist)
                else:
                    toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            if random.random() < mutpb:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        # Evaluate the entire population
        fitness = list(map(toolbox.evaluate, offspring))
        for ind, fit in zip(offspring, fitness):
            ind.fitness.values = [fit]

        # The population is entirely replaced by the offspring
        # pop = offspring
        pop[:] = tools.selBest(offspring, pop_size - 1)
        pop.append(elitist)

        hof.update(pop) if stats else {}

        record = stats.compile(pop) if stats else {}
        # print(record)
        logbook.record(gen=g + 1, **record, best_pop=elitist)
        if verbose:
            print('Logbook:', logbook.stream)
    return pop, logbook, pop_init


# Fa la media su n_iter e stampa le stats sulle colonne. Il numero di righe è n_gen*len(cxpb)*len(mtpb)*len(cxop)
def medium_pd(name, n_gen, cxpb, mtpb, cxop, n_iter):
    df = pd.read_csv(name)
    df_medium = pd.DataFrame(columns=['cxop', 'cxpb', 'mtpb', 'gen', 'avg', 'std', 'min', 'max'])
    avg, std, min, max, q1, q3, op, mut, cspb, gen = [], [], [], [], [], [], [], [], [], []
    for o in range(len(cxop)):
        for c in range(len(cxpb)):
            for m in range(len(mtpb)):
                for g in range(n_gen + 1):
                    gen_avg, gen_std, gen_min, gen_max, gen_q1, gen_q3 = [], [], [], [], [], []
                    for j in range(n_iter):
                        index = (o * len(mtpb) * len(cxpb) + c * len(mtpb) + m) * n_iter + j
                        gen_avg.append(ast.literal_eval(df.iloc[index][g])['avg'])
                        gen_std.append(ast.literal_eval(df.iloc[index][g])['std'])
                        gen_min.append(ast.literal_eval(df.iloc[index][g])['min'])
                        gen_max.append(ast.literal_eval(df.iloc[index][g])['max'])
                        gen_q1.append(ast.literal_eval(df.iloc[index][g])['q1'])
                        gen_q3.append(ast.literal_eval(df.iloc[index][g])['q3'])
                    avg.append(mean(gen_avg))
                    std.append(mean(gen_std))
                    min.append(mean(gen_min))
                    max.append(mean(gen_max))
                    q1.append(mean(gen_q1))
                    q3.append(mean(gen_q3))
                    op.append(cxop[o])
                    mut.append(mtpb[m])
                    cspb.append(cxpb[c])
                    gen.append(g)
                df_medium = pd.merge(df_medium, pd.DataFrame({'cxop': op, 'cxpb': cspb, 'mtpb': mut, 'gen': gen,
                                                              'avg': avg, 'std': std, 'min': min, 'max': max, 'q1': q1,
                                                              'q3': q3}), how='outer')
    return df_medium


# Scrive due file per ogni funzione di benchmark: la simulazione rozza e quella con i dati mediati su n_iter
def write_csv(directory, ind_size, pop_size, n_gen, n_iter, test_f, cxpb, mtpb, cxop, beta=None, backend=None):
    toolbox = createToolbox(ind_size=ind_size)
    stats = createStats()

    for test in test_f:
        pop_init = []
        for o in cxop:
            data, Q = [], []
            for c in cxpb:
                for m in mtpb:
                    if o == 'quantum_crossover' or o == 'quantum_crossover_noise':
                        if o == 'quantum_crossover_noise':
                            backend = 'fake_noise'
                        algo = []
                        for i in range(n_iter):
                            print('Test=', test, ', cxop=', o, ', cxpb=', c, ', mtpb=', m, ', n_iter=', i)
                            '''Q.append(updatedGA(toolbox=toolbox, pop_size=pop_size, n_gen=n_gen, cxpb=c, mutpb=m,
                                               cx_operator=o, backend=backend, stats=stats,
                                               num_marked=5, test=test, pop_init='mine', n_iter=i,
                                               hof=tools.HallOfFame(1))[1])'''
                            if o == cxop[0] and c == cxpb[0] and m == mtpb[0]:
                                algo.append(updatedGA(toolbox=toolbox, pop_size=pop_size, n_gen=n_gen, cxpb=c, mutpb=m,
                                            cx_operator=o, backend=backend, stats=stats, num_marked=5,
                                            test=test, hof=tools.HallOfFame(1)))
                                Q.append(algo[i][1])
                                pop_init.append(algo[i][2])
                            else:
                                Q.append(updatedGA(toolbox=toolbox, pop_size=pop_size, n_gen=n_gen, cxpb=c, mutpb=m,
                                                   cx_operator=o, backend=backend, stats=stats,
                                                   num_marked=5, test=test, pop_init=pop_init[i],
                                                   hof=tools.HallOfFame(1))[1])
                        dataQ = pd.DataFrame(Q)
                        dataQ.to_csv(os.path.join(directory, test + '_' + o + '.csv'), index=False)

                    else:
                        for i in range(n_iter):
                            print('Test=', test, ', cxop=', o, ', cxpb=', c, ', mtpb=', m, ', n_iter=', i)
                            '''data.append(updatedGA(toolbox=toolbox, pop_size=pop_size, n_gen=n_gen, cxpb=c, mutpb=m,
                                                  cx_operator=o, stats=stats, num_marked=5, test=test,
                                                  pop_init='mine', n_iter=i, hof=tools.HallOfFame(1))[1])'''
                            if o == cxop[0] and c == cxpb[0] and m == mtpb[0]:
                                data.append(updatedGA(toolbox=toolbox, pop_size=pop_size, n_gen=n_gen, cxpb=c, mutpb=m,
                                            cx_operator=o, stats=stats, num_marked=5, test=test,
                                            hof=tools.HallOfFame(1))[1])
                                pop_init.append(updatedGA(toolbox=toolbox, pop_size=pop_size, n_gen=n_gen, cxpb=c,
                                                          mutpb=m, cx_operator=o, stats=stats, num_marked=5, test=test,
                                                          hof=tools.HallOfFame(1))[2])
                            else:

                                data.append(updatedGA(toolbox=toolbox, pop_size=pop_size, n_gen=n_gen, cxpb=c, mutpb=m,
                                                      cx_operator=o, stats=stats, num_marked=5, test=test,
                                                      pop_init=pop_init[i], hof=tools.HallOfFame(1))[1])

                        df_log = pd.DataFrame(data)
                        df_log.to_csv(os.path.join(directory, test+'_'+o+'.csv'), index=False)

    return print('write function has finished')


# Cerchiamo gli iperparametri migliori per ogni funzione di benchmark e op di crossover.
def tuning(directory, test, n_gen, cxpb, mtpb, cxop, n_iter):

    for test_ in test:
        frames = []
        for o in cxop:
            frames.append(pd.read_csv(directory+'/'+test_+'_'+o+'.csv'))
        data_for_test_functions = pd.concat(frames, ignore_index=True)
        data_for_test_functions.to_csv(os.path.join(directory, test_ + '.csv'), index=False)
        df = medium_pd(directory+'/' + test_ + '.csv', n_gen, cxpb, mtpb, cxop, n_iter)
        df.to_csv(os.path.join(directory, 'medium_' + test_ + '.csv'))

    hp = pd.DataFrame(columns=['Test function', 'Crossover operator', 'cxpb', 'mtpb'])
    tf, crop, crosspb, mutpb, avg, std, max, min = [], [], [], [], [], [], [], []
    for t in test:
        dataframe = pd.read_csv(directory+'/medium_' + t + '.csv', index_col=0)
        n_rows = len(dataframe.index)

        pase = int(n_rows / len(cxop))
        for i in range(len(cxop)):
            idx = dataframe.iloc[i * pase:pase * (i + 1)]['max'].idxmax()
            tf.append(t)
            crop.append(dataframe.iloc[idx]['cxop'])
            crosspb.append(dataframe.iloc[idx]['cxpb'])
            mutpb.append(dataframe.iloc[idx]['mtpb'])
            avg.append(dataframe.iloc[idx]['avg'])
            std.append(dataframe.iloc[idx]['std'])
            min.append(dataframe.iloc[idx]['min'])
            max.append(dataframe.iloc[idx]['max'])
        data = pd.DataFrame({'Test function': tf, 'Crossover operator': crop, 'cxpb': crosspb, 'mtpb': mutpb})
        hp = pd.merge(hp, data, how='outer')
    hp.to_csv(directory+'/hp_tuned.csv')
    return hp


# Save initial population of the n iterations
def initial_population(directory, name, iteraz):
    df = pd.read_csv(directory+'/'+name+'.csv')
    for i in range(iteraz):
        pop_init = df.iloc[i][0]
        pop = ast.literal_eval(pop_init)['best_pop']
        with open(directory+'/Init_pop/'+name+str(i)+'.json', 'w') as f:
            json.dump(pop, f)
    return print('file created: ', iteraz)


# Plotting
def plot(directory, cxop, cxpb, mtpb):
    df = pd.read_csv(directory+'/hp_tuned.csv')

    for i in range(len(df.index)):
        test_, cxop_, cxpb_, mtpb_ = df.iloc[i]['Test function'], df.iloc[i]['Crossover operator'], \
                                     df.iloc[i]['cxpb'], df.iloc[i]['mtpb']

        cxop_index, cxpb_index, mtpb_index = cxop.index(cxop_), cxpb.index(cxpb_), mtpb.index(mtpb_)
        data = pd.read_csv(directory+'/medium_' + test_ + '.csv')
        n_gen = data['gen'].max()

        starting_row = (mtpb_index + cxpb_index * len(mtpb) + cxop_index * len(mtpb) * len(cxpb)) * (n_gen + 1)
        best = []
        for j in range(n_gen + 1):
            best.append(data.iloc[starting_row + j]['max'])

        plt.plot([j for j in range(n_gen + 1)], best, label=cxop[i%len(cxop)])
        plt.title(test_)
        plt.legend()
        plt.savefig(directory+'/' + test_ + '.png')
        if i % len(cxop) == len(cxop)-1:
            plt.clf()
    return print(' gen vs fitness plot completed')


def plot_best(directory, test, cxop, n_iter):
    for i in range(len(cxop)):
        data = pd.read_csv(directory+'/' + test[0] + '_' + cxop[i] + '.csv')
        n_gen = ast.literal_eval(data.iloc[0, -1])['gen']
        max_list = []
        for k in range(n_iter):
            max_list.append(ast.literal_eval(data.iloc[k, -1])['max'])
        max_value = max(max_list)
        best_row = max_list.index(max_value)
        best = []
        for j in range(n_gen + 1):
            best.append(ast.literal_eval(data.iloc[best_row, j])['max'])

        plt.plot([j for j in range(n_gen + 1)], best, label=cxop[i])
    plt.title(test[0])
    plt.legend()
    plt.savefig(directory+'/' + test[0] + '.png')
    plt.clf()
    return print('Plot gen vs fitness completed')


def boxplot_best(directory, test, cxop, n_iter):
    data = []
    for i in range(len(cxop)):
        df = pd.read_csv(directory+'/' + test[0] + '_' + cxop[i] + '.csv')
        max_list = []
        for k in range(n_iter):
            max_list.append(ast.literal_eval(df.iloc[k, -1])['max'])
        data.append(max_list)
    plt.boxplot(data, patch_artist=True, labels=cxop, showfliers=False, showmeans=True)
    plt.title(test[0])
    plt.savefig(directory+'/' + test[0] + '-boxplot.png')
    return print('boxplot completed')


def boxplot(directory, test, cxop, cxpb, mtpb, n_iter):
    tuned = pd.read_csv(directory+'/hp_tuned.csv')
    test_, cxop_, cxpb_, mtpb_ = [], [], [], []

    for i in range(len(tuned.index)):
        test_.append(tuned.iloc[i]['Test function'])
        cxop_.append(tuned.iloc[i]['Crossover operator'])
        cxpb_.append(tuned.iloc[i]['cxpb'])
        mtpb_.append(tuned.iloc[i]['mtpb'])
    cxop_index, cxpb_index, mtpb_index = list(map(cxop.index, cxop_)), list(map(cxpb.index, cxpb_)), \
                                         list(map(mtpb.index, mtpb_))
        #cxop_index, cxpb_index, mtpb_index = cxop.index(cxop_), cxpb.index(cxpb_), mtpb.index(mtpb_)
    for t in range(len(test)):
        data = []
        for i in range(len(cxop)):
            df = pd.read_csv(directory+'/' + test[t] + '_' + cxop[i] + '.csv')
            k_init = n_iter*(len(mtpb)*cxpb_index[t*i+i]+mtpb_index[t*i+i])
            #print(test[t], cxop[i], k_init)
            max_list = []
            for k in range(n_iter):
                max_list.append(ast.literal_eval(df.iloc[k_init+k, -1])['max'])
            data.append(max_list)
        plt.boxplot(data, patch_artist=True, labels=cxop, showfliers=False, showmeans=False)
        plt.title(test[t])
        plt.savefig(directory+'/' + test[t] + '-boxplot.png')
        plt.clf()
    return print('boxplot completed')
