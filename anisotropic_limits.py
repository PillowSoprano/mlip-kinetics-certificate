"""
Which bound is the answer when the model error has a large component in degrees of freedom ORTHOGONAL to the reaction coordinate?

Toy: V(x, y) = B (x^2-1)^2 + k y^2 / 2  (x slow reaction coordinate, y orthogonal),  diffusion D_x = 1, D_y = r.
Error: dV(x, y) = bump on the barrier  +  'orthogonal noise' n(x, y) with zero mean under mu(y) at every x.
For separable V the reactive current runs along x, so flux tubes are the rows y = const and isocommittor slices are the columns x = const:
    parallel (tube) lower bound :  sum_y mu(y) / int nu(dx) e^{+dV(x,y)}          <- exact when y is FROZEN   (r -> 0)
    series  (slice) upper bound :  1 / int nu(dx) / < e^{-dV(x,.)} >_{mu(y)}      <- exact when y is FAST     (r -> infinity): first a free-energy
                                                                                     perturbation over y, then the 1D formula in x
The exact capacity ratio is computed from the resistor network for r = 1e-3 ... 1e3.
"""
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve

B, KY, NX, NY = 6.0, 8.0, 161, 61
x = np.linspace(-1.5, 1.5, NX); y = np.linspace(-1.4, 1.4, NY); hx, hy = x[1] - x[0], y[1] - y[0]
X, Y = np.meshgrid(x, y, indexing="ij"); idx = np.arange(NX * NY).reshape(NX, NY)
ex_i, ex_j = idx[:-1, :].ravel(), idx[1:, :].ravel(); ey_i, ey_j = idx[:, :-1].ravel(), idx[:, 1:].ravel()
inA = (X < -0.8).ravel(); inB = (X > 0.8).ravel()


def capacity(V, r):
    w = np.exp(-(V - V.min()) / 2).ravel()
    c = np.r_[w[ex_i] * w[ex_j] / hx**2, r * w[ey_i] * w[ey_j] / hy**2]; I = np.r_[ex_i, ey_i]; J = np.r_[ex_j, ey_j]
    W = sp.coo_matrix((np.r_[c, c], (np.r_[I, J], np.r_[J, I])), shape=(NX * NY,) * 2).tocsr()
    L = sp.diags(np.asarray(W.sum(1)).ravel()) - W; free = ~(inA | inB)
    q = np.zeros(NX * NY); q[inB] = 1; q[free] = spsolve(L[free][:, free].tocsc(), -(L[free][:, inB] @ np.ones(inB.sum())))
    return (c * (q[I] - q[J]) ** 2).sum() * np.exp(-V.min())


if __name__ == "__main__":
    V = B * (X**2 - 1) ** 2 + 0.5 * KY * Y**2
    muy = np.exp(-0.5 * KY * y**2); muy /= muy.sum()
    nux = np.exp(B * (x**2 - 1) ** 2) * (np.abs(x) <= 0.8); nux /= nux.sum()
    rng = np.random.default_rng(0); rows = []
    ratios = [1e-3, 1e-2, 1e-1, 1, 10, 100, 1000]
    print("orth. noise sd (kT) | parallel lower | exact ln(C_hat/C) for D_y/D_x = " + " ".join(f"{r:g}" for r in ratios) + " | series upper | frozen lower / upper")
    for sd in (0.0, 0.5, 1.0, 2.0):
        n = sum(rng.normal() * np.cos(rng.uniform(2, 7) * X + rng.uniform(0, 6)) * np.sin(rng.uniform(3, 9) * Y + rng.uniform(0, 6)) for _ in range(8))
        n = n - (n * muy[None, :]).sum(1, keepdims=True)                         # zero mean over y at every x
        n = sd * n / np.sqrt(((n**2) * muy[None, :] * nux[:, None]).sum() + 1e-30)
        dV = 1.5 * np.exp(-X**2 / (2 * 0.25**2)) + n
        par = np.log((muy / (nux[:, None] * np.exp(dV)).sum(0)).sum())
        ser = -np.log((nux / (np.exp(-dV) * muy[None, :]).sum(1)).sum())
        nu2 = nux[:, None] * muy[None, :]; flo, fup = -np.log((nu2 * np.exp(dV)).sum()), np.log((nu2 * np.exp(-dV)).sum())
        ex = [np.log(capacity(V + dV, r) / capacity(V, r)) for r in ratios]
        rows.append([sd, par, *ex, ser, flo, fup])
        print(f"        {sd:.1f}         |   {par:+.3f}     | " + " ".join(f"{e:+.3f}" for e in ex) + f" |   {ser:+.3f}    | {flo:+.3f} / {fup:+.3f}", flush=True)
    np.savetxt("data/anisotropic_limits.csv", rows, delimiter=",", comments="",
               header="noise_sd,parallel_lower," + ",".join(f"exact_r{r:g}" for r in ratios) + ",series_upper,frozen_lower,frozen_upper")
