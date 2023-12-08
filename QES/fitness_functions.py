import math
import numpy as np
import trackhhl.track_qes as track

def sphere(vector):
    fn = sum(-component * component for component in vector)
    return fn


def schwefel(vector):
    fn = 0
    for i in range(len(vector)):
        fn += -sum([gene ** 2 for gene in vector[:i]])
    return fn


def rastrigin(vector):
    fn = 0
    for i in range(len(vector)):
        fn += -(10+vector[i]**2-10*math.cos(2*vector[i]*math.pi))
    return fn


def rosenbrock(vector):
    fn = 0
    for i in range(len(vector) - 1):
        fn += -((100 * (vector[i + 1] - vector[i] ** 2) ** 2) +
                (vector[i] - 1) ** 2)
    return fn


def ackley(vector):
    A = sum(gene * gene for gene in vector)
    B = sum(math.cos(2 * math.pi * gene) for gene in vector)
    fn = 20 * math.exp(-0.2 * math.sqrt(A / len(vector))) + math.exp(B / len(vector)) - 20 - math.e
    return fn


def get_variable_name(variable):
    for name in globals():
        if id(globals()[name]) == id(variable):
            return name
    for name in locals():
        if id(locals()[name]) == id(variable):
            return name
    return None


problem, sol, sol_c = track.problem()


def track_reconstruction(vector):
    vector = np.array(vector)
    fn = problem.evaluate(solution=vector)
    return fn[0][0]
