"""
Series / parallel refinement of the Dirichlet-Thomson bracket (no hand-made channels).

Flow coordinates of the REFERENCE dynamics: s = committor q in [0,1], sigma = stream function of the reactive flux.
  frozen lower :  C_hat/C >= 1 / E_nu[e^{+dV}]
  tube   lower :  C_hat/C >= sum_t w_t / E_{nu|tube t}[e^{+dV}]          (Thomson, flux tubes re-weighted: parallel law)
  slice  upper :  C_hat/C <= 1 / sum_k p_k / E_{nu|slice k}[e^{-dV}]     (Dirichlet, committor re-parametrised: series law)
  frozen upper :  C_hat/C <= E_nu[e^{-dV}]
Everything is computed on the resistor network that discretises the generator (conductance c_ij = sqrt(pi_i pi_j)/h^2),
for which the exact capacity is obtained from a sparse linear solve.
"""
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve
import twochannel_blindspot as t

N, h = t.N, t.h
idx = np.arange(N * N).reshape(N, N)
ex_i, ex_j = idx[:-1, :].ravel(), idx[1:, :].ravel()          # x-directed edges
ey_i, ey_j = idx[:, :-1].ravel(), idx[:, 1:].ravel()          # y-directed edges
EI, EJ = np.concatenate([ex_i, ey_i]), np.concatenate([ex_j, ey_j])


def network(V):
    w = np.exp(-(V - V.min()) / 2).ravel()
    return w[EI] * w[EJ] / h**2                                # edge conductances


def capacity(c, inA, inB):
    n = N * N
    W = sp.coo_matrix((np.r_[c, c], (np.r_[EI, EJ], np.r_[EJ, EI])), shape=(n, n)).tocsr()
    L = sp.diags(np.asarray(W.sum(1)).ravel()) - W
    free = ~(inA | inB)
    q = np.zeros(n); q[inB] = 1.0
    q[free] = spsolve(L[free][:, free].tocsc(), -(L[free][:, inB] @ np.ones(inB.sum())))
    return (c * (q[EI] - q[EJ]) ** 2).sum(), q


def bounds(c, q, dV, n_tubes, n_slices):
    dVe = 0.5 * (dV.ravel()[EI] + dV.ravel()[EJ])
    nu = c * (q[EI] - q[EJ]) ** 2; nu /= nu.sum()
    # stream function: cumulative x-directed flux integrated upwards in y, evaluated on nodes
    Jx = np.zeros((N, N)); Jx[:-1, :] = (c[:len(ex_i)] * (q[ex_j] - q[ex_i])).reshape(N - 1, N)
    psi = np.cumsum(Jx, axis=1); psi = psi / psi[:, -1].max()
    tube = np.clip((psi.ravel()[EI] * n_tubes).astype(int), 0, n_tubes - 1)
    slc = np.clip((0.5 * (q[EI] + q[EJ]) * n_slices).astype(int), 0, n_slices - 1)
    lo_frozen = -np.log((nu * np.exp(dVe)).sum())
    up_frozen = np.log((nu * np.exp(-dVe)).sum())
    wt = np.bincount(tube, nu, n_tubes); mt = np.bincount(tube, nu * np.exp(dVe), n_tubes)
    ok = wt > 0
    lo_tube = np.log((wt[ok] ** 2 / mt[ok]).sum())             # sum_t w_t / (m_t / w_t)
    # series law, done rigorously on the network: test function f = h(q), h piecewise linear with n_slices knots, optimised exactly
    knots = np.linspace(0, 1, n_slices + 1)
    def hat(qv):                                               # values of the hat basis at q, dense (n_edges x n_knots)
        Bm = np.zeros((len(qv), n_slices + 1)); k = np.clip((qv * n_slices).astype(int), 0, n_slices - 1)
        fr = qv * n_slices - k; Bm[np.arange(len(qv)), k] = 1 - fr; Bm[np.arange(len(qv)), k + 1] = fr
        return Bm
    act = nu > 1e-14
    Dm = hat(q[EI][act]) - hat(q[EJ][act])
    Mh = Dm.T @ (Dm * (c[act] * np.exp(-dVe[act]))[:, None])
    inner = slice(1, n_slices)
    theta = np.zeros(n_slices + 1); theta[-1] = 1.0
    theta[inner] = np.linalg.solve(Mh[inner, inner] + 1e-300 * np.eye(n_slices - 1), -Mh[inner, -1])
    up_slice = np.log(theta @ Mh @ theta / (c * (q[EI] - q[EJ]) ** 2).sum())
    return lo_frozen, lo_tube, up_slice, up_frozen


if __name__ == "__main__":
    V = t.true_potential()
    inA = ((V < 2.0) & (t.X < 0)).ravel(); inB = ((V < 2.0) & (t.X > 0)).ravel()
    c0 = network(V); C, q = capacity(c0, inA, inB)
    rows = []
    print("ln(C_hat/C):  block seed | frozen-lo   tube-lo(2)  tube-lo(16)  tube-lo(64) | EXACT | slice-up(64) slice-up(8) frozen-up")
    for block in (1, 3, 6, 9, 12, 18):
        for seed in (1, 2, 3):
            t.BLOCK = float(block); t.rng = np.random.default_rng(seed); dV = t.initial_error()
            Ch, _ = capacity(network(V + dV), inA, inB)
            # conductances are defined up to the partition function; compare unnormalised capacities
            scale = np.exp(-((V + dV).min() - V.min()))
            ex = np.log(Ch * scale / C)
            b2 = bounds(c0, q, dV, 2, 8); b16 = bounds(c0, q, dV, 16, 8); b64 = bounds(c0, q, dV, 64, 48)
            rows.append((block, seed, b2[0], b2[1], b16[1], b64[1], ex, b64[2], b2[2], b2[3]))
            if seed == 1:
                print(f"            {block:5d} {seed:4d} | {b2[0]:+8.3f}  {b2[1]:+8.3f}   {b16[1]:+8.3f}    {b64[1]:+8.3f}  | {ex:+.3f} | {b64[2]:+8.3f}   {b2[2]:+8.3f}  {b2[3]:+8.3f}")
    R = np.array(rows)
    viol_lo = (R[:, 5] - R[:, 6]).max(); viol_up = (R[:, 6] - R[:, 7]).max()
    print(f"\n18 cases: max (tube-lo(64) - exact) = {viol_lo:+.4f}  (<= 0 means the bound holds);  max (exact - slice-up(64)) = {viol_up:+.4f}")
    print(f"tube-lo(16): max (bound - exact) = {(R[:,4]-R[:,6]).max():+.4f};  |tube-lo(16) - exact| median {np.median(abs(R[:,4]-R[:,6])):.3f}")
    print(f"bracket width, median over cases:  frozen {np.median(R[:,9]-R[:,2]):.3f}  ->  tube(64)/slice(64) {np.median(R[:,7]-R[:,5]):.3f}")
    print(f"|tube-lo(64) - exact|: median {np.median(abs(R[:,5]-R[:,6])):.3f}, max {abs(R[:,5]-R[:,6]).max():.3f}")
    np.savetxt("data/bracket_2d_tubes.csv", R, delimiter=",", comments="",
               header="block_kT,seed,frozen_lower,tube2_lower,tube16_lower,tube64_lower,exact,slice64_upper,slice8_upper,frozen_upper")
