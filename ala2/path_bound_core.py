import numpy as np
from numba import njit


@njit(cache=True)
def sample_paths(indptr, nbr, cum, qv, dve_p, dve_m, start_nodes, start_cum, inB, npaths, seed):
    np.random.seed(seed)
    mp = np.empty(npaths); mm = np.empty(npaths)
    for k in range(npaths):
        i = start_nodes[np.searchsorted(start_cum, np.random.rand())]
        a = 0.0; b = 0.0
        while not inB[i]:
            lo, hi = indptr[i], indptr[i + 1]
            e = lo + np.searchsorted(cum[lo:hi], np.random.rand())
            if e >= hi: e = hi - 1
            j = nbr[e]; dq = qv[j] - qv[i]
            a += dve_p[e] * dq; b += dve_m[e] * dq; i = j
        mp[k] = a; mm[k] = b
    return mp, mm


