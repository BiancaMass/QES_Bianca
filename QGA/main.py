import operators as Op
import sys
from deap import tools
N_GEN = 10
POP_SIZE = 10
IND_SIZE = 4
N_ITER = 3
#TEST = ['sphere', 'rastrigin', 'schwefel', 'rosenbrock']
TEST = ['sphere', 'rastrigin', ]
CXPB = [0.5, 0.6, 0.7, 0.8, 0.9, 1]
#MTPB = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
MTPB = [0.3, 0.4]
#CXOP = ['onep', 'twop', 'uniform', 'blend', 'quantum_crossover']
CXOP = ['onep', 'blend', 'quantum_crossover']
BETA = [0.5]

'''pop, logbook = Op.updatedGA(toolbox, pop_size=POP_SIZE, n_gen=N_GEN, cxpb=CXPB[0], mutpb=MTPB[0],
                            cx_operator=CXOP[0], test=TEST[0], stats=stats, num_marked=5, hof=tools.HallOfFame(1))'''

Op.write_csv(ind_size=IND_SIZE, pop_size=POP_SIZE, n_gen=N_GEN, n_iter=N_ITER, test_f=TEST, cxpb=CXPB, mtpb=MTPB,
             cxop=CXOP, backend='fake', beta=BETA)
Op.tuning2(test=TEST, cxop=CXOP)