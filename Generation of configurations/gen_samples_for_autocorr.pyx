#language_level=3
import cython
import numpy as np
cimport numpy as np
from libc.time cimport time
from libc.math cimport exp
from mc_lib.rndm cimport RndmWrapper
from _common import tabulate_neighbors
from mc_lib.observable cimport RealObservable

@cython.boundscheck(False)
@cython.wraparound(False)
cdef np.ndarray initState(int linear_size, RndmWrapper rndm):
    cdef np.ndarray[np.int_t, ndim=2,negative_indices=False,
                                                  mode='c'] lattice = np.zeros([linear_size, linear_size], dtype=int)
    cdef int i = 0
    cdef int j = 0
    for i in range(linear_size):
        for j in range(linear_size):
            lattice[i,j] = 1 if rndm.uniform() > 0.5 else -1
    return lattice

@cython.boundscheck(False)
@cython.wraparound(False)
cdef np.ndarray mcmove(np.ndarray config,double beta,int L,
                                                         np.ndarray [np.int_t, ndim=2,negative_indices=False] ngb,
                                                                                                              RndmWrapper rndm):
    """
    One Monte-Carlo step

    params:
        config: Current configuration of lattice
        beta:   Inversed temperature of current configuration
        L:      Linear size 'L' of lattice
        ngb:    Array of neigbours
    return:
    """
    cdef:
        double foo = rndm.uniform()
        int i = int(foo*(L*L)) //L
        int j = int(foo*(L*L)) %L
        int site =  config[i, j]
        double dE = 0
    for n in range(1,5):
        dE += site * config[ngb[i*L + j, n] //L, ngb[i*L + j, n] % L]
    cdef double ratio = exp(-2 * dE * beta)
    if rndm.uniform() > ratio:
        config[i, j] = site
    else:
        site *= -1
        config[i, j] = site
    return



@cython.boundscheck(False)  # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire functio
cdef double calcEnergy(np.ndarray config, int L, ngb):
    cdef:
        int i = 0
        int j = 0
        int site = 0
        double energy = 0
    for i in range(L):
        for j in range(L):
            site = config[i, j]
            for n in range(1,5):
                one_ngb = config[ngb[i*L + j, n] //L, ngb[i*L + j, n] % L]
                energy += -1 * site * one_ngb
    return energy / 4.


@cython.boundscheck(False)
@cython.wraparound(False)
def IsingSimulate(L, T, sweeps, int seed = np.random.randint(0,1000), int rseed = 1234):
    """
        L - linear_size
        T - One temperature point
        Sweeps - number of L^2 Metropolis Monte-Karlo steps

    """
    cdef RndmWrapper rndm = RndmWrapper((rseed, seed))
    cdef RealObservable e = RealObservable()
    cdef:
        float beta = 1.0 / T
        int sweep = 0
        int steps_per_sweep = L * L
        int num_therm = int(10 * L * L)
        int i = 0

    cdef:
        list configs = []
        np.ndarray [np.int_t, ndim=2,negative_indices=False] ngb = tabulate_neighbors(L, kind='sq')
        np.ndarray[np.int_t, ndim = 2, negative_indices = False, mode = 'c'] config
    config = initState(L, rndm)

    for sweep in range(num_therm):
        for i in range(steps_per_sweep):
            mcmove(config, beta, L, ngb, rndm)
    enes = np.zeros(sweeps)

    for sweep in range(sweeps):
        for i in range(steps_per_sweep):
            mcmove(config, beta, L, ngb, rndm)

        Et = calcEnergy(config, L, ngb)
        e.add_measurement(Et)
        enes[sweep] = Et

    error = e.errorbar
    mean_energ = e.mean
    converg = e.is_converged
    return enes, mean_energ, error,converg

