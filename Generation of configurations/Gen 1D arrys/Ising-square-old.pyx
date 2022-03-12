import cython
import numpy as np
cimport numpy as np
from libc.time cimport time
from libc.math cimport exp
from mc_lib.rndm cimport RndmWrapper
from _common import tabulate_neighbors

#from mc_lib.observable cimport RealObservable

@cython.boundscheck(False)
@cython.wraparound(False)
cdef np.ndarray initState(np.ndarray[np.int_t, ndim=1, negative_indices=False, mode='c'] lattice,
                          RndmWrapper rndm):
    """

    :param   lattice: unconfigured array of future lattice (empty array)
    :param   rndm: class with random methods from mc_lib
    :return: change the 'Lattice' variable and return nothing
    """
    for i in range(lattice.shape[0]):
        lattice[i] = 1 if rndm.uniform() > 0.5 else -1
    return

@cython.boundscheck(False)
@cython.wraparound(False)
cdef np.ndarray mcmove(np.ndarray[np.int_t, ndim=1, negative_indices=False, mode='c'] config,
                       double beta,
                       np.ndarray[np.int_t, ndim=2, negative_indices=False] ngb,
                       RndmWrapper rndm):
    """
    One flip attempt
    :param    config: Current configuration of lattice
    :param    beta:   Inversed temperature of current configuration
    :param    L:      Linear size 'L' of lattice
    :param    ngb:    Array of neigbours
    """
    cdef:
        int site = int(config.shape[0] * rndm.uniform())
        double dE = 0
        long num_ngb = ngb[site, 0]
    for n in range(1, num_ngb + 1):
        site1 = ngb[site, n]
        dE += config[site1] * config[site]
    cdef double ratio = exp(-2 * dE * beta)
    if rndm.uniform() > ratio:
        return
    else:
        config[site] = -config[site]
    return

@cython.boundscheck(False)  # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire functio
cdef double calcEnergy(np.ndarray[np.int_t, ndim=1, negative_indices=False, mode='c'] config,
                       np.ndarray[np.int_t, ndim=2, negative_indices=False] ngb):
    """
        Count current configuration energy
        :param    config: Current configuration of lattice
        :param    ngb:    Array of neigbours
        return:
    """
    cdef:
        int site = 0
        double energy = 0
    for site in range(config.shape[0]):
        num_ngb = ngb[site, 0]
        for n in range(1, num_ngb + 1):
            site1 = ngb[site, n]
            energy += -1 * config[site] * config[site1]
    return energy / 4.

@cython.boundscheck(False)
@cython.wraparound(False)
def IsingSimulate(L, T, sweeps, int T_corr, int seed = np.random.randint(0, 1000), int rseed = 1234):
    """
        L - linear_size
        T - One temperature point
        Sweeps - number of L^2 Metropolis Monte-Karlo steps
    """
    cdef RndmWrapper rndm = RndmWrapper((rseed, seed))
    #cdef RealObservable e = RealObservable()
    cdef:
        float beta = 1.0 / T
        int sweep = 0
        int steps_per_sweep = L * L
        int num_therm = int(10 * L * L)
        int i = 0

    cdef:
        np.ndarray configs = np.empty([sweeps, L * L], dtype=int)
        np.ndarray[np.int_t, ndim = 2, negative_indices=False] ngb = tabulate_neighbors(L, kind='sq')
        np.ndarray[np.int_t, ndim = 1, negative_indices = False, mode = 'c'] config = np.empty(L * L, dtype=int)
    initState(config, rndm)

    for sweep in range(num_therm):
        for i in range(steps_per_sweep):
            mcmove(config, beta, ngb, rndm)

    for sweep in range(sweeps):
        for i in range(steps_per_sweep * T_corr):
            mcmove(config, beta, ngb, rndm)
        configs[sweep] = config.copy()
    return configs