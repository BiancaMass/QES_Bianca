import random
import numpy
import math
import pandas as pd
import matplotlib.pyplot as plt
import ast
import os.path
from statistics import mean
from deap import base
from deap import creator
from deap import tools
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, execute, BasicAer
from qiskit import IBMQ

IBMQ.enable_account('3a1e44c417a24cfd7ad48ef3a7f9580be5b94fb0964c798bef468426bc6f6961761a3084e081b884a443d40c445497863'
                    'fdd687953f085b8a95e67d1ca5cce70')


def createToolbox(ind_size):
    # Creating Classes
    creator.create('FitnessMax', base.Fitness, weights=(1.0,))
    creator.create('Individual', list, fitness=creator.FitnessMax)
    toolbox = base.Toolbox()
    toolbox.register('attr_float', random.uniform, a=-5, b=5)
    toolbox.register('individual', tools.initRepeat, creator.Individual, toolbox.attr_float,
                     n=ind_size)
    toolbox.register('population', tools.initRepeat, list, toolbox.individual)
    ''' The crossover operator will be add in the update function, which have the possibility to choose between 
    classic and quantum ones'''
    # Mutation
    toolbox.register('mutate', tools.mutGaussian, mu=0.0, sigma=0.2, indpb=0.2)
    # Selection
    toolbox.register('select_RWS', tools.selRoulette)
    return toolbox


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
    for i in range(len(x)-1):
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
        A += x[i]**2
        B += math.cos(2*math.pi*x[i])
    f = 20*math.exp(-0.2*math.sqrt(A/n)) + math.exp(B/n)-20-math.e
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


def Quantum_Crossover(parent1, parent2, backend_name, best, beta):
    if beta < 0 or beta > math.pi / 2:
        raise 'Value Error: beta must be in [0, pi/2]'
    n = int(len(parent1))
    d1, d2 = [], []
    for i in range(n):
        d1.append(parent1[i] - best[i])
        d2.append(parent2[i] - best[i])

    parents = QuantumRegister(n, 'parents')
    choice = ClassicalRegister(n, 'children')
    qc = QuantumCircuit(parents, choice)
    for i in range(n):
        qc.h(parents[i])
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
              beta=None, backend=None, verbose=False):
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
    pop = toolbox.population(n=pop_size)

    # Defining the Logbook
    logbook = tools.Logbook()
    logbook.header = ["gen", "best pop"] + (stats.fields if stats else [])

    # Evaluate the entire population
    fitness = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitness):
        ind.fitness.values = [fit]

    hof.update(pop) if stats else {}

    record = stats.compile(pop) if stats else {}
    logbook.record(gen=0, **record)
    if verbose:
        print('Logbook:', logbook.stream)

    for g in range(n_gen):
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
                if cx_operator == 'quantum_crossover':
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
        pop[:] = tools.selBest(offspring, pop_size-1)
        pop.append(elitist)

        hof.update(pop) if stats else {}

        record = stats.compile(pop) if stats else {}
        # print(record)
        logbook.record(gen=g+1, **record, best_pop=elitist)
        if verbose:
            print('Logbook:', logbook.stream)
    return pop, logbook


# Fa la media su n_iter e stampa le stats sulle colonne. Il numero di righe è n_gen*len(cxpb)*len(mtpb)*len(cxop)
def medium_pd(name, n_gen, cxpb, mtpb, cxop, n_iter):
    df = pd.read_csv(name)
    df_medium = pd.DataFrame(columns=['cxop', 'cxpb', 'mtpb', 'gen', 'avg', 'std', 'min', 'max'])
    avg, std, min, max, q1, q3, op, mut, cspb, gen = [], [], [], [], [], [], [], [],  [], []
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
def write_csv(ind_size, pop_size, n_gen, n_iter, test_f, cxpb, mtpb, cxop, beta=None, backend=None):
    toolbox = createToolbox(ind_size=ind_size)
    stats = createStats()
    titles, logbooks, a = [], [], []
    for test in test_f:
        titles.append(test + '.csv')
        a = []
        for o in cxop:
            for c in cxpb:
                for m in mtpb:
                    if o == 'quantum_crossover':
                        for b in beta:
                            for i in range(n_iter):
                                a.append(updatedGA(toolbox=toolbox, pop_size=pop_size, n_gen=n_gen, cxpb=c, mutpb=m,
                                                   cx_operator=o, beta=b, backend=backend, stats=stats, num_marked=5,
                                                   test=test, hof=tools.HallOfFame(1))[1])

                    else:
                        for i in range(n_iter):
                            a.append(updatedGA(toolbox=toolbox, pop_size=pop_size, n_gen=n_gen, cxpb=c, mutpb=m,
                                               cx_operator=o, stats=stats, num_marked=5, test=test,
                                               hof=tools.HallOfFame(1))[1])
        logbooks.append(a)

    df_log, df = pd.DataFrame(), pd.DataFrame()
    for i in range(len(logbooks)):
        df_log = pd.DataFrame(logbooks[i])
        df_log.to_csv(os.path.join('Tuning', titles[i]), index=False)
    for i in titles:
        df = medium_pd('Tuning/' + i, n_gen, cxpb, mtpb, cxop, n_iter)
        df.to_csv(os.path.join('Tuning', 'medium_' + i))

    return df_log, df


# Cerchiamo gli iperparametri migliori per ogni funzione di benchmark e op di crossover.
def tuning(test, cxop):
    hp = pd.DataFrame(columns=['Test function', 'Crossover operator', 'cxpb', 'mtpb'])
    tf, crop, crosspb, mutpb, avg, std, max, min = [], [], [], [], [], [], [], []
    for t in test:
        dataframe = pd.read_csv('Tuning/medium_' + t + '.csv', index_col=0)
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
    hp.to_csv('Tuning/hp_tuned.csv')
    return hp


# Plotting
def plot(cxop, cxpb, mtpb):
    df = pd.read_csv('Tuning/hp_tuned.csv')

    for i in range(len(df.index)):
        test_, cxop_, cxpb_, mtpb_ = df.iloc[i]['Test function'], df.iloc[i]['Crossover operator'], \
                                     df.iloc[i]['cxpb'], df.iloc[i]['mtpb']

        cxop_index, cxpb_index, mtpb_index = cxop.index(cxop_), cxpb.index(cxpb_), mtpb.index(mtpb_)
        data = pd.read_csv('Tuning/medium_' + test_ + '.csv')
        n_gen = data['gen'].max()
        starting_row = (mtpb_index + cxpb_index * len(mtpb) + cxop_index * len(mtpb) * len(cxpb)) * (n_gen+1)
        best = []
        for j in range(n_gen+1):
            best.append(data.iloc[starting_row + j]['max'])

        plt.plot([j for j in range(n_gen+1)], best)
        plt.title(test_+' - '+cxop_)
        plt.savefig('Tuning/' + test_ + '_' + cxop_ + '.png')
        plt.show()
    return print('Plotting is finished')
