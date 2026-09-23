"""
The equilibrium distribution of the hop coordinate, from an unbiased equilibrium trajectory.

Along the hop coordinate the leverage density is the inverse Boltzmann factor, nu proportional to exp(+beta F),
so F must come from an unbiased ensemble: the histogram of stored reactive frames is the density of reactive
paths, exp(-beta F) q(1-q), and the factor q(1-q) fills the barrier in.  For every equilibrium frame the vacant
site is located and s is computed for each lithium ion on its eight nearest-neighbour sites.

    python equilibrium_coordinate.py            ->  ../data/li3ocl_s_equilibrium_1000K.npy
"""
import numpy as np
from analyze_hops import ideal

OUT = "../data/li3ocl_s_equilibrium_1000K.npy"
FUSE = 1.4                                       # a fused lithium pair; the equilibrium ensemble never reaches it

d = np.load("../colab/step6_results/eq_1000K.npz")
X, Z, L = d["X"].astype(np.float64), d["numbers"], d["cell"]
a = L[0] / 3
S = np.array([(np.array(f) + [ix, iy, iz]) * a for ix in range(3) for iy in range(3) for iz in range(3)
              for f in ((.5, .5, 0), (.5, 0, .5), (0, .5, .5))])
pos0, _ = ideal(); li, fw = np.where(Z == 3)[0], np.where(Z != 3)[0]; ref = pos0[fw].mean(0)
Y = X[:, li] - (X[:, fw].mean(1) - ref)[:, None]                       # framework-relative, as in the hop analysis
dd = Y[:, :, None, :] - S[None, None]; dd -= L * np.round(dd / L)
occ = np.linalg.norm(dd, axis=-1).argmin(-1)
nn = np.linalg.norm((S[:, None] - S[None]) - L * np.round((S[:, None] - S[None]) / L), axis=-1)
neigh = [np.where((nn[i] > 0.1) & (nn[i] < a / np.sqrt(2) + 0.05))[0] for i in range(len(S))]

sv = []
for k in range(len(X)):
    vac = np.setdiff1d(np.arange(len(S)), occ[k])
    if len(vac) != 1: continue                                          # not the single-vacancy state
    v = vac[0]
    for i in neigh[v]:
        ion = np.where(occ[k] == i)[0]
        if len(ion) != 1: continue
        e = S[v] - S[i]; e -= L * np.round(e / L)
        u = Y[k, ion[0]] - S[i]; u -= L * np.round(u / L)
        sv.append(u @ e / (e ** 2).sum())
sv = np.array(sv)
np.save(OUT, sv)
h, ed = np.histogram(sv, bins=np.linspace(-0.25, 0.55, 33))
print(f"{len(sv)} (ion, vacancy) samples from {len(X)} equilibrium frames; saved {OUT}")
print("  s bin, counts, -ln p relative to the peak:")
for lo, hi, c in zip(ed[:-1], ed[1:], h):
    if c: print(f"   {lo:+.3f}..{hi:+.3f} {c:6d}  {np.log(h.max()/c):6.2f}")
