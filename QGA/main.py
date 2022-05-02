import operators as Op
import sys
sys.setrecursionlimit(4000)

N_GEN = 10
POP_SIZE = 10
IND_SIZE = 4
N_ITER = 2
TEST = ['sphere', 'rastrigin', 'schwefel', 'rosenbrock', 'ackley']
CXPB = [0.5, 0.6, 0.7, 0.8, 0.9, 1]
MTPB = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
CXOP = ['onep', 'twop', 'uniform', 'blend', 'quantum_crossover']
BETA = [0.5]

Op.write_csv(ind_size=IND_SIZE, pop_size=POP_SIZE, n_gen=N_GEN, n_iter=N_ITER, test_f=TEST, cxpb=CXPB, mtpb=MTPB,
             cxop=CXOP, backend='fake', beta=BETA)


Op.tuning(test=TEST, cxop=CXOP)

Op.plot(cxop=CXOP, cxpb=CXPB, mtpb=MTPB)