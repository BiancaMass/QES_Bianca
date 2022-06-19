import operators as Op
import sys

sys.setrecursionlimit(4000)

N_GEN = 30
POP_SIZE = 20
IND_SIZE = 5
N_ITER = 10
#TEST = ['rastrigin', ]
#CXPB = [0.7]
#MTPB = [0.6]
#CXOP = ['quantum_crossover_noise']
TEST = ['sphere', 'rastrigin', 'schwefel', 'rosenbrock', 'ackley']
CXPB = [0.5, 0.6, 0.7, 0.8, 0.9]
MTPB = [0.2, 0.3, 0.4, 0.5, 0.6]
CXOP = ['onep', 'twop', 'uniform', 'blend', 'quantum_crossover', 'quantum_crossover_noise']
directory = '5_bits/nobetatuning'

#Op.initial_population(directory=directory, name='sphere', iteraz=N_ITER)


#Op.write_csv(directory=directory, ind_size=IND_SIZE, pop_size=POP_SIZE, n_gen=N_GEN, n_iter=N_ITER, test_f=TEST,
             #cxpb=CXPB, mtpb=MTPB, cxop=CXOP, backend='fake',)

Op.tuning(directory=directory, test=TEST, n_gen=N_GEN, n_iter=N_ITER, cxop=CXOP, cxpb=CXPB, mtpb=MTPB,)

Op.plot(directory=directory, cxop=CXOP, cxpb=CXPB, mtpb=MTPB)
#Op.plot_best(directory, test=TEST, cxop=CXOP, n_iter=N_ITER)
Op.boxplot(directory, test=TEST, cxop=CXOP, cxpb=CXPB, mtpb=MTPB, n_iter=N_ITER)
#Op.boxplot_best(directory, test=TEST, cxop=CXOP, n_iter=N_ITER)
#Op.best(directory=directory, test=TEST, cxop=CXOP, n_gen=N_GEN)
