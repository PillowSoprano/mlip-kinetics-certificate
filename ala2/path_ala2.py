"""Path estimator of leverage averages on alanine dipeptide, from the REFERENCE trajectories only.
<g>_nu ~ mean over reactive segments of sum_t g(x_mid) * dq_t, q = committor of the free-energy-surface model (values only).
Compared with: the grid (SqRA network) leverage average, the TST surface ensemble, and the measured rate changes."""
import json, sys
import numpy as np
import analyze as a

TAG = sys.argv[1] if len(sys.argv) > 1 else ""
trajs, T = a.load("ref" + TAG); kT = a.KJ_PER_KT(T)
geo_trajs, _ = a.load("ref")                                   # geometry always from the gamma = 1 reference
C_lag, keep = a.msm(geo_trajs); C = a.sqra(C_lag, keep); q = a.committor(C, keep); pi = C_lag.sum(1) / C_lag.sum()
NB = a.NB
Q = np.full(NB * NB, np.nan); Q[keep] = q
P, S = a.centres()
phid = np.degrees(P)
Q[np.isnan(Q) & (phid > a.CORE_A[0]) & (phid < a.CORE_A[1])] = 0.0
Q[np.isnan(Q) & (phid > a.CORE_B[0]) & (phid < a.CORE_B[1])] = 1.0
Q[np.isnan(Q)] = 0.5                                           # never-visited bins between the cores (rarely touched)
wrap = lambda d: d - 2 * np.pi * np.floor((d + np.pi) / (2 * np.pi))


def dV_at(bumps, phi, psi):
    return sum(A / kT * np.exp(-(wrap(phi - p0) ** 2 + wrap(psi - s0) ** 2) / (2 * w**2)) for A, p0, s0, w in bumps)


def reactive_segments(x):
    phi = np.degrees(x[:, 0])
    st = np.where((phi > a.CORE_A[0]) & (phi < a.CORE_A[1]), 0, np.where((phi > a.CORE_B[0]) & (phi < a.CORE_B[1]), 1, -1))
    st[np.isnan(phi)] = -2
    segs, last, last_idx = [], -1, -1
    for k, s in enumerate(st):
        if s == -2:
            last = -1
        elif s >= 0:
            if last >= 0 and s != last:
                segs.append((last_idx, k, 1.0 if s == 1 else -1.0))
            last, last_idx = s, k
    return segs


design = json.load(open("design.json"))
acc = {n: [] for n in design}
lengths = []
for x in trajs:
    b = np.clip(((x + np.pi) / (2 * np.pi) * NB), 0, NB - 1e-9)
    for i0, i1, sgn in reactive_segments(x):
        seg = x[i0:i1 + 1]; bb = b[i0:i1 + 1].astype(int); qs = Q[bb[:, 0] * NB + bb[:, 1]]
        dq = np.diff(qs) * sgn
        mid_phi = seg[:-1, 0] + 0.5 * wrap(seg[1:, 0] - seg[:-1, 0]); mid_psi = seg[:-1, 1] + 0.5 * wrap(seg[1:, 1] - seg[:-1, 1])
        lengths.append((i1 - i0) * 0.1)
        for n, bmp in design.items():
            d = dV_at(bmp, mid_phi, mid_psi)
            acc[n].append(((np.exp(d) * dq).sum(), (np.exp(-d) * dq).sum(), dq.sum()))
print(f"{len(lengths)} reactive segments, median duration {np.median(lengths):.1f} ps, mean sum(dq) = {np.mean([v[2] for v in acc['ts_up']]):.3f}")

meas = {"": {"ts_up": -0.579, "ts_down": 1.411, "basinA": 0.077, "basinB": 0.319, "c5": 0.080}, "_g10": {"ts_up": -0.631, "ts_down": 1.160}}[TAG]
A_side = q < 0.5
rows = []
print("system   | grid (FES model) bracket | path bracket (frozen)   | per-path parallel law | measured")
for n, bmp in design.items():
    v = np.array(acc[n]); norm = v[:, 2].mean()
    ep, em = v[:, 0].mean() / norm, v[:, 1].mean() / norm
    d = a.gauss_field(bmp, kT)[keep]
    aA = (pi[A_side] * np.exp(-d[A_side])).sum() / pi[A_side].sum(); aB = (pi[~A_side] * np.exp(-d[~A_side])).sum() / pi[~A_side].sum()
    lnG = np.log(pi[A_side].sum() / aB + pi[~A_side].sum() / aA)
    lo_g, up_g, *_ = a.predict(C, keep, q, a.gauss_field(bmp, kT), pi)
    m = np.clip(v[:, 0], 1e-6, None)
    par = lnG + np.log(np.mean(1.0 / m))
    boot = [lnG + np.log(np.mean(1.0 / m[np.random.default_rng(r).integers(0, len(m), len(m))])) for r in range(200)]
    print(f"{n:8s} | [{lo_g:+.3f}, {up_g:+.3f}]         | [{lnG-np.log(ep):+.3f}, {lnG+np.log(em):+.3f}]        |  {par:+.3f} +- {np.std(boot):.3f}      | {meas.get(n, float('nan')):+.3f}")
    rows.append((lo_g, up_g, lnG - np.log(ep), lnG + np.log(em), par, np.std(boot), meas.get(n, np.nan)))
np.savetxt(f"../data/ala2_path_estimator{TAG}.csv", rows, delimiter=",", comments="",
           header="grid_lower,grid_upper,path_lower,path_upper,per_path_parallel,per_path_sd,measured")
