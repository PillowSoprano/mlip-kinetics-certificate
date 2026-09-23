"""
How does the effect of a potential error on the rate depend on friction?   (1D underdamped Langevin, m = beta = 1)

Klein-Kramers generator in a Hermite basis in p:   (L f)_m = sqrt(m) f'_{m-1} + sqrt(m+1) f'_{m+1} - V' sqrt(m+1) f_{m+1} - gamma m f_m
The slowest non-zero eigenvalue is the relaxation rate. We perturb V by a bump on the saddle and compare ln(k_hat/k) with
    TST limit          : ratio = exp(-dV(x*)) / <e^{-dV}>_basin                      (surface ensemble, x* = top of V_hat)
    overdamped limit   : ratio = G / <e^{+dV}>_nu,  nu ~ e^{+V} on the barrier       (exact in 1D)
"""
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigs

B, NX, NH, XMAX = 6.0, 360, 36, 1.75
x = np.linspace(-XMAX, XMAX, NX); h = x[1] - x[0]


def slow_rate(V, gamma):
    dV = np.gradient(V, h)
    D = sp.diags([-np.ones(NX - 1), np.ones(NX - 1)], [-1, 1], format="lil") / (2 * h)
    D[0, :2] = [-1 / h, 1 / h]; D[-1, -2:] = [-1 / h, 1 / h]; D = D.tocsr()
    blocks = [[None] * NH for _ in range(NH)]
    for m in range(NH):
        blocks[m][m] = -gamma * m * sp.identity(NX, format="csr")
        if m + 1 < NH:
            blocks[m][m + 1] = np.sqrt(m + 1) * (D - sp.diags(dV))
        if m >= 1:
            blocks[m][m - 1] = np.sqrt(m) * D
    L = sp.bmat(blocks, format="csc")
    ev = eigs(L, k=6, sigma=1e-4, which="LM", return_eigenvectors=False)
    ev = ev[np.abs(ev.imag) < 1e-6 * (1 + np.abs(ev.real))].real
    ev = np.sort(-ev)                                        # decay rates, ascending; first ~ 0
    return ev[ev > 1e-9][0]


def overdamped_exact(V, d):
    nu = np.exp(V) * (np.abs(x) < 0.9); nu /= nu.sum(); mu = np.exp(-V); mu /= mu.sum()
    A = x < 0; aA = (mu[A] * np.exp(-d[A])).sum() / mu[A].sum(); aB = (mu[~A] * np.exp(-d[~A])).sum() / mu[~A].sum()
    return np.log(mu[A].sum() / aB + mu[~A].sum() / aA) - np.log((nu * np.exp(d)).sum())


def tst(V, d):
    mu = np.exp(-V); mu /= mu.sum(); A = x < 0
    aA = (mu[A] * np.exp(-d[A])).sum() / mu[A].sum()
    mid = np.abs(x) < 0.9
    return -((V + d)[mid].max() - V[mid].max()) - np.log(aA)  # variational TST: the surface sits on top of V_hat


if __name__ == "__main__":
    V = B * (x**2 - 1) ** 2
    wb = np.sqrt(4 * B)
    cases = {"narrow bump (w=0.06)": 2.0 * np.exp(-x**2 / (2 * 0.06**2)), "wide bump (w=0.40)": 2.0 * np.exp(-x**2 / (2 * 0.40**2)),
             "odd wiggle (w=0.12)": 2.0 * (x / 0.12) * np.exp(-x**2 / (2 * 0.12**2))}
    gammas = [0.3, 1, 3, 10, 30, 100, 300]
    k0 = {g: slow_rate(V, g) for g in gammas}
    print(f"barrier {B} kT, barrier frequency omega_b = {wb:.2f}; reference rates:", {g: f"{k:.2e}" for g, k in k0.items()})
    rows = []
    for name, d in cases.items():
        print(f"\n{name}:   TST limit {tst(V, d):+.3f}    overdamped limit {overdamped_exact(V, d):+.3f}")
        line = []
        for g in gammas:
            r = np.log(slow_rate(V + d, g) / k0[g]); line.append(r)
            rows.append((list(cases).index(name), g, r, tst(V, d), overdamped_exact(V, d)))
        print("   gamma/omega_b: " + "  ".join(f"{g/wb:6.2f}" for g in gammas))
        print("   ln(k_hat/k)  : " + "  ".join(f"{r:+6.3f}" for r in line))
    np.savetxt("data/friction_crossover_1d.csv", rows, delimiter=",", header="case,gamma,ln_ratio,tst_limit,overdamped_limit", comments="")
