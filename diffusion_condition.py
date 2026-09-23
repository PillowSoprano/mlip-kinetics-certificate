"""
Condition number of the effective diffusion coefficient with respect to the equilibrium force RMSE.

First order:   d ln D = -beta * sum_x dV(x) [nu(x) - mu(x)]          (leverage density minus population density)
Worst case over all dV with force RMSE eps:   sup |d ln D| = eps * beta * ||grad u||_{L2(mu)},   div(mu grad u) = nu - mu
(so the worst perturbation is dV* ~ u, the solution of a Poisson equation).  Conjectured closed form in 1D, deep barriers:

        cond_D = sqrt( beta / (12 D / D0) )               [compare the rate result  sqrt(beta / (3 kappa))]
"""
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve
import diffusion_bounds as db


def cond_1d(V, h):
    n = len(V); w = np.exp(-(V - V.min()) / 2)
    c = w * np.roll(w, -1) / h**2                                   # edge i -> i+1 (periodic)
    Z = (w**2).sum()
    # corrector in 1D: flux constant  => gradient g_e ~ 1/c_e
    g = (1 / c); g *= (n * h) / (g.sum() * h) * 1.0                 # sum g_e * h = cell length
    g = g * h
    D = (c * g**2).sum() / Z
    nu_e = c * g**2; nu_e /= nu_e.sum()
    nu = 0.5 * (nu_e + np.roll(nu_e, 1)); mu = w**2 / Z
    rhs = nu - mu
    L = sp.diags([c + np.roll(c, 1)], [0]) - sp.coo_matrix((np.r_[c, c], (np.r_[np.arange(n), (np.arange(n) + 1) % n], np.r_[(np.arange(n) + 1) % n, np.arange(n)])), shape=(n, n))
    L = L.tocsr(); keep = np.arange(1, n)
    u = np.zeros(n); u[keep] = spsolve(L[keep][:, keep].tocsc(), rhs[keep])
    return D, np.sqrt(Z * (rhs @ u)), u, mu, nu


def cond_2d(V):
    D, c, g, mu = db.solve_cell(V)
    n2 = db.n * db.n
    nu_e = c * g**2; nu_e /= nu_e.sum()
    nu = 0.5 * (np.bincount(db.EI, nu_e, n2) + np.bincount(db.EJ, nu_e, n2))
    rhs = nu - mu
    W = sp.coo_matrix((np.r_[c, c], (np.r_[db.EI, db.EJ], np.r_[db.EJ, db.EI])), shape=(n2, n2)).tocsr()
    L = sp.diags(np.asarray(W.sum(1)).ravel()) - W
    keep = np.arange(1, n2); u = np.zeros(n2); u[keep] = spsolve(L[keep][:, keep].tocsc(), rhs[keep])
    Z = np.exp(-(V - V.min())).sum()
    return D, np.sqrt(Z * (rhs @ u)), u


if __name__ == "__main__":
    print("1D, V = (B/2) cos(2 pi x):   B | D/D0 | cond (numerical) | sqrt(1/(12 D)) | ratio")
    n = 4000; h = 1.0 / n; x = (np.arange(n) + 0.5) * h
    rows = []
    for B in (1, 2, 4, 6, 8, 10, 12, 14):
        D, cnd, u, mu, nu = cond_1d(0.5 * B * np.cos(2 * np.pi * x), h)
        rows.append((B, D, cnd, np.sqrt(1 / (12 * D))))
        print(f"   {B:4d} | {D:.3e} | {cnd:9.3f} | {np.sqrt(1/(12*D)):9.3f} | {cnd/np.sqrt(1/(12*D)):.4f}")
    np.savetxt("data/diffusion_condition_1d.csv", rows, delimiter=",", header="B_kT,D_over_D0,cond_numeric,cond_closed_form", comments="")

    # is the worst case really attained?  perturb along dV* = u with small force RMSE and compare with the exact D_hat
    B = 8; V = 0.5 * B * np.cos(2 * np.pi * x); D, cnd, u, mu, nu = cond_1d(V, h)
    LJ = lambda U: 1.0 / (np.mean(np.exp(U)) * np.mean(np.exp(-U)))
    du = np.diff(np.r_[u, u[0]]) / h; rm = np.sqrt((mu * du**2).sum())
    for eps in (0.01, 0.05):
        dV = u * eps / rm
        print(f"   attainment check B=8, force RMSE={eps}: exact |ln(D_hat/D)| = {abs(np.log(LJ(V+dV)/LJ(V))):.5f}   cond*eps = {cnd*eps:.5f}")

    print("\n2D, V = (B/2) cos(2 pi x) cos(2 pi y) + 0.25 B cos(4 pi x):   B | D_xx/D0 | cond | cond*sqrt(D)")
    rows = []
    for B in (1, 2, 4, 6, 8, 10):
        V = 0.5 * B * np.cos(2 * np.pi * db.X) * np.cos(2 * np.pi * db.Y) + 0.25 * B * np.cos(4 * np.pi * db.X)
        D, cnd, _ = cond_2d(V); rows.append((B, D, cnd))
        print(f"   {B:4d} | {D:.3e} | {cnd:9.3f} | {cnd*np.sqrt(D):.4f}")
    np.savetxt("data/diffusion_condition_2d.csv", rows, delimiter=",", header="B_kT,Dxx_over_D0,cond_numeric", comments="")
    R = np.array(rows); print(f"   slope of ln(cond) vs ln(1/D) over B>=4: {np.polyfit(np.log(1/R[2:,1]), np.log(R[2:,2]), 1)[0]:.3f}   (1D theory: 0.5)")
