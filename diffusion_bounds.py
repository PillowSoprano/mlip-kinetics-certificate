"""
Extension of the Dirichlet-Thomson bracket from rates to the EFFECTIVE DIFFUSION COEFFICIENT in a periodic potential
(the relevant observable for ion conductors; no single dominant barrier, no metastability asymptotics needed).

Overdamped dynamics on the unit cell, D0 = 1.  Homogenisation:  D_xx = (1/Z) min_f  int |e_x + grad f|^2 e^{-V},  f periodic.
With f* the reference corrector,  nu ~ |e_x + grad f*|^2 e^{-V}  (leverage density),  mu ~ e^{-V}:

      1 / ( <e^{+dV}>_nu <e^{-dV}>_mu )   <=   D_hat / D   <=   <e^{-dV}>_nu / <e^{-dV}>_mu

(upper: plug f* into the perturbed principle; lower: plug the reference flux into the dual principle).
1D: the lower bound is exact (Lifson-Jackson).
"""
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve

n = 72; h = 1.0 / n
xs = (np.arange(n) + 0.5) * h
X, Y = np.meshgrid(xs, xs, indexing="ij")
idx = np.arange(n * n).reshape(n, n)
EI = np.r_[idx.ravel(), idx.ravel()]
EJ = np.r_[np.roll(idx, -1, 0).ravel(), np.roll(idx, -1, 1).ravel()]          # +x neighbours, then +y neighbours
DX = np.r_[np.full(n * n, h), np.zeros(n * n)]


def solve_cell(V):
    w = np.exp(-(V - V.min()) / 2).ravel()
    c = w[EI] * w[EJ] / h**2
    W = sp.coo_matrix((np.r_[c, c], (np.r_[EI, EJ], np.r_[EJ, EI])), shape=(n * n, n * n)).tocsr()
    L = (sp.diags(np.asarray(W.sum(1)).ravel()) - W).tolil()
    b = np.bincount(EI, -c * DX, n * n) + np.bincount(EJ, c * DX, n * n)       # d/df of sum c (DX + f_j - f_i)^2
    L = L.tocsr(); keep = np.arange(1, n * n)
    f = np.zeros(n * n); f[keep] = spsolve(L[keep][:, keep].tocsc(), -b[keep])
    g = DX + f[EJ] - f[EI]
    Z = (w**2).sum()
    return (c * g**2).sum() / Z, c, g, w**2 / Z


def random_field(rng, amp):
    f = np.zeros_like(X)
    for _ in range(rng.integers(2, 6)):
        cx, cy, wd = rng.uniform(0, 1), rng.uniform(0, 1), rng.uniform(0.04, 0.15)
        dx = np.minimum(abs(X - cx), 1 - abs(X - cx)); dy = np.minimum(abs(Y - cy), 1 - abs(Y - cy))
        f += rng.normal() * np.exp(-(dx**2 + dy**2) / (2 * wd**2))
    return amp * f / np.abs(f).max()


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    # a 2D "conductor-like" landscape: two inequivalent sites per cell, several shallow barriers, two hop types
    V = 3.0 * (np.cos(2 * np.pi * X) * np.cos(2 * np.pi * Y)) + 1.5 * np.cos(4 * np.pi * X) + 1.0 * np.sin(2 * np.pi * (X + Y))
    D, c, g, mu = solve_cell(V)
    nu = c * g**2; nu /= nu.sum()
    print(f"reference: D_xx / D0 = {D:.4f}   (barriers ~ {V.max()-V.min():.1f} kT peak-to-peak, no single dominant saddle)")
    rows = []
    for amp in (0.3, 1.0, 3.0, 6.0):
        for k in range(30):
            dV = random_field(rng, amp)
            Dh = solve_cell(V + dV)[0]
            dVe = 0.5 * (dV.ravel()[EI] + dV.ravel()[EJ])
            mneg = (mu * np.exp(-dV.ravel())).sum()
            lo = -np.log((nu * np.exp(dVe)).sum()) - np.log(mneg)
            up = np.log((nu * np.exp(-dVe)).sum()) - np.log(mneg)
            first = -(nu * dVe).sum() + (mu * dV.ravel()).sum()
            fx, fy = np.gradient(dV, h); rmse = np.sqrt((mu * (fx**2 + fy**2).ravel()).sum())
            rows.append((amp, np.log(Dh / D), lo, up, first, rmse))
    R = np.array(rows)
    print("\n max|dV| | cases inside bracket | median width | median |first-order - exact| | range of ln(D_hat/D) | Spearman(force RMSE, |ln ratio|)")
    from scipy.stats import spearmanr
    for amp in (0.3, 1.0, 3.0, 6.0):
        r = R[R[:, 0] == amp]
        inside = np.mean((r[:, 1] >= r[:, 2] - 1e-6) & (r[:, 1] <= r[:, 3] + 1e-6)) * 100
        print(f"   {amp:4.1f}  |  {inside:5.1f}%  |  {np.median(r[:,3]-r[:,2]):.3f}  |  {np.median(abs(r[:,4]-r[:,1])):.4f}  |  [{r[:,1].min():+.2f}, {r[:,1].max():+.2f}]  |  {spearmanr(r[:,5], abs(r[:,1]))[0]:+.2f}")
    print(f"\n overall Spearman with |exact ln(D_hat/D)|:  force RMSE {spearmanr(R[:,5], abs(R[:,1]))[0]:+.2f}   |bracket centre| {spearmanr(abs(0.5*(R[:,2]+R[:,3])), abs(R[:,1]))[0]:+.2f}")
    np.savetxt("data/diffusion_bounds_2d.csv", R, delimiter=",", header="max_abs_dV,exact_ln_ratio,lower,upper,first_order,force_rmse", comments="")

    # 1D check: lower bound should be exact (Lifson-Jackson)
    x1 = (np.arange(4000) + 0.5) / 4000; V1 = 4 * np.cos(2 * np.pi * x1) + 2 * np.cos(6 * np.pi * x1)
    LJ = lambda U: 1.0 / (np.mean(np.exp(U)) * np.mean(np.exp(-U)))
    d1 = 3.0 * np.exp(-((x1 - 0.5) ** 2) / (2 * 0.03**2)) - 1.5 * np.exp(-((x1 - 0.2) ** 2) / (2 * 0.05**2))
    nu1 = np.exp(V1) / np.exp(V1).sum(); mu1 = np.exp(-V1) / np.exp(-V1).sum()
    print(f"\n 1D: exact ln(D_hat/D) = {np.log(LJ(V1 + d1) / LJ(V1)):+.6f}   lower-bound formula = {-np.log((nu1*np.exp(d1)).sum()) - np.log((mu1*np.exp(-d1)).sum()):+.6f}")
