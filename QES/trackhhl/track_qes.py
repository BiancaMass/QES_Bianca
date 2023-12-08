from trackhhl.hamiltonians.simple_hamiltonian import SimpleHamiltonian
from trackhhl.toy.simple_generator import SimpleDetectorGeometry, SimpleGenerator
import numpy as np


# EXPERIMENTAL SETUP
def problem():
    N_DETECTORS = 4
    N_PARTICLES = 3
    detector = SimpleDetectorGeometry([i for i in range(N_DETECTORS)], [10000 for i in range(N_DETECTORS)],
                                          [10000 for i in range(N_DETECTORS)], [i + 1 for i in range(N_DETECTORS)])
    generator = SimpleGenerator(detector, theta_max=np.pi / 3, rng=np.random.default_rng(0))
    event = generator.generate_event(N_PARTICLES)
    # Problem Definition
    EPSILON = 1e-5
    GAMMA = 2
    DELTA = 1
    ham = SimpleHamiltonian(EPSILON, GAMMA, DELTA)
    ham.construct_hamiltonian(event)
    sol_c = ham.solve_classicaly()
    # Solution
    sol = [1 if seg.hit_from.track_id == seg.hit_to.track_id else 0 for seg in ham.segments]
    return ham, sol, sol_c
