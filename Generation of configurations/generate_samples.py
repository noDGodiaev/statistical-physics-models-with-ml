import sys
import numpy as np
import gen_samples_for_autocorr as ng

if __name__ == '__main__':
    L = int(sys.argv[1])
    path = str(sys.argv[2])
    steps = int(sys.argv[3])
    gen_num = int(sys.argv[4])
    T_c = tc = 2 / (np.log(1 + 2 ** 0.5))
    for i in range(1):
        energies, mean_energ, error, converg  = ng.IsingSimulate(L=L, T=T_c, sweeps=steps)
        print(gen_num + i + 1, flush=True)
        np.save(path + 'energies' + str(L) +'_' + str(gen_num + i + 1) + '.npy', energies)
