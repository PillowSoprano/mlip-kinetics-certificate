"""Rigorous per-path lower bound + frozen bounds on the alanine-dipeptide free-energy-surface network, vs the measured rate changes."""
import sys, json; sys.path.insert(0, "..")
import numpy as np, analyze as a
from path_bound_core import sample_paths

trajs, T = a.load("ref_g1"); kT = a.KJ_PER_KT(T)
C_lag, keep = a.msm(trajs); C = a.sqra(C_lag, keep); q = a.committor(C, keep); pi = C_lag.sum(1) / C_lag.sum(); n = len(keep)
I, J = np.nonzero(C); Jp = C[I, J] * np.maximum(q[J] - q[I], 0); k = Jp > 0; I, J, Jp = I[k], J[k], Jp[k]
o = np.argsort(I, kind="stable"); I, J, Jp = I[o], J[o], Jp[o]
indptr = np.zeros(n + 1, dtype=np.int64); np.add.at(indptr, I + 1, 1); indptr = np.cumsum(indptr)
out = np.bincount(I, Jp, n); cum = np.empty_like(Jp)
for i in np.unique(I): cum[indptr[i]:indptr[i + 1]] = np.cumsum(Jp[indptr[i]:indptr[i + 1]]) / out[i]
phi = np.degrees(a.centres()[0][keep]); inA = (phi > a.CORE_A[0]) & (phi < a.CORE_A[1]); inB = (phi > a.CORE_B[0]) & (phi < a.CORE_B[1])
sA = np.where(inA & (out > 0))[0]; scum = np.cumsum(out[sA]) / out[sA].sum()
meas = {"ts_up": (-0.712, 0.051), "ts_down": (1.257, 0.034)}
print("system  | frozen bracket          | per-path lower (rigorous on the network) | network exact | measured (gamma = 1)")
for name, bmp in json.load(open("design.json")).items():
    if name not in meas: continue
    d = a.gauss_field(bmp, kT)[keep]; de = 0.5 * (d[I] + d[J])
    mp, mm = sample_paths(indptr, J, cum, q, np.exp(de), np.exp(-de), sA, scum, inB, 40000, 3)
    lo, up, first, net, _ = a.predict(C, keep, q, a.gauss_field(bmp, kT), pi)
    A_ = q < 0.5; aA = (pi[A_] * np.exp(-d[A_])).sum() / pi[A_].sum(); aB = (pi[~A_] * np.exp(-d[~A_])).sum() / pi[~A_].sum()
    lnG = np.log(pi[A_].sum() / aB + pi[~A_].sum() / aA)
    print(f"{name:7s} | [{lo:+.3f}, {up:+.3f}]        |   {lnG + np.log(np.mean(1/mp)):+.3f}                                 |   {net:+.3f}      | {meas[name][0]:+.3f} +- {meas[name][1]:.3f}")

print("\ncapacity-level check (what the theorem is about):  ln(C_hat/C)  per-path lower | exact | Dirichlet upper")
def cap(Cm):
    L = np.diag(Cm.sum(1)) - Cm; free = ~(inA | inB); qq = np.zeros(n); qq[inB] = 1
    qq[free] = np.linalg.solve(L[np.ix_(free, free)], -L[np.ix_(free, inB)].sum(1)); ii, jj = np.nonzero(np.triu(Cm, 1))
    return (Cm[ii, jj] * (qq[ii] - qq[jj]) ** 2).sum()
C0 = cap(C)
for name, bmp in json.load(open("design.json")).items():
    if name not in meas: continue
    d = a.gauss_field(bmp, kT)[keep]; de = 0.5 * (d[I] + d[J])
    mp, mm = sample_paths(indptr, J, cum, q, np.exp(de), np.exp(-de), sA, scum, inB, 40000, 3)
    Ch = cap(C * np.exp(-0.5 * (d[:, None] + d[None, :])))
    print(f"{name:7s}   {np.log(np.mean(1/mp)):+.4f} | {np.log(Ch/C0):+.4f} | {np.log(mm.mean()):+.4f}")
