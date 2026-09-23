"""
Stage 1 on alanine dipeptide: can the bracket, computed from the REFERENCE trajectory only, contain the measured rate
change of a perturbed system?

Reference data -> reversible MSM on a (phi, psi) grid -> committor q, conductances c_ij, leverage density nu_ij ~ c_ij (q_i-q_j)^2.
A perturbation dV(phi, psi) acts on the chain as  c_ij -> c_ij exp(-(dV_i + dV_j)/2)   (square-root approximation),
for which Dirichlet / Thomson are exact theorems:
        lnG - ln<e^{+dV}>_nu  <=  ln(k_hat / k)  <=  lnG + ln<e^{-dV}>_nu ,      k = relaxation rate k_AB + k_BA
Ground truth: brute-force MD of the perturbed system, rates from core-to-core transition counts.
"""
import sys, glob, json
import numpy as np

NB, LAG = 30, 20                       # 12-degree bins, lag = 2 ps
KJ_PER_KT = lambda T: 0.0083144626 * T
CORE_A, CORE_B = (-150, -40), (30, 110)


def load(system):
    files = sorted(glob.glob(f"traj_{system}_s*.npz"))
    out = []
    for f in files:
        d = np.load(f); x = d["phipsi"].copy()
        if "omega" in d.files:                        # frames with a cis peptide bond are a different molecule state: mask them
            rep = int(np.ceil(len(x) / len(d["omega"])))
            cis = np.repeat((np.abs(d["omega"]) < np.pi / 2).any(1), rep)[:len(x)]
            x[cis] = np.nan
        out.append(x)
    return out, float(np.load(files[0])["temp"])


def md_rates(trajs, dt_ps=0.1):
    nAB = nBA = 0; tA = tB = 0.0
    for x in trajs:
        phi = np.degrees(x[:, 0])
        st = np.where((phi > CORE_A[0]) & (phi < CORE_A[1]), 0, np.where((phi > CORE_B[0]) & (phi < CORE_B[1]), 1, -1))
        st[np.isnan(phi)] = -2
        last = -1
        for s in st:                                   # last-visited-core assignment; a masked (cis) frame resets the memory
            if s == -2:
                last = -1; continue
            if s >= 0:
                if last == 0 and s == 1: nAB += 1
                if last == 1 and s == 0: nBA += 1
                last = s
            if last == 0: tA += dt_ps
            elif last == 1: tB += dt_ps
    return nAB, nBA, tA, tB


def msm(trajs):
    C = np.zeros((NB * NB, NB * NB))
    for x in trajs:
        x = x[~np.isnan(x[:, 0])]
        b = np.clip(((x + np.pi) / (2 * np.pi) * NB).astype(int), 0, NB - 1); s = b[:, 0] * NB + b[:, 1]
        np.add.at(C, (s[:-LAG], s[LAG:]), 1.0)
    C = C + C.T                                        # reversible (equilibrium) estimator
    keep = np.where(C.sum(1) > 20)[0]                  # drop barely visited bins
    C = C[np.ix_(keep, keep)]
    return C, keep


def sqra(C, keep):
    """Overdamped model on the free-energy surface: nearest-neighbour conductances c_ij = sqrt(pi_i pi_j) (periodic grid).
    Uses the reference data only through the histogram; the lag-time MSM smears the committor for low-friction dynamics."""
    pi = C.sum(1) / C.sum(); pos = {int(k): n for n, k in enumerate(keep)}
    Cn = np.zeros_like(C)
    for n, k in enumerate(keep):
        a, b = divmod(int(k), NB)
        for k2 in (((a + 1) % NB) * NB + b, a * NB + (b + 1) % NB):
            m = pos.get(k2)
            if m is not None:
                Cn[n, m] = Cn[m, n] = np.sqrt(pi[n] * pi[m])
    return Cn


def centres():
    c = (np.arange(NB) + 0.5) / NB * 2 * np.pi - np.pi
    P, S = np.meshgrid(c, c, indexing="ij")
    return P.ravel(), S.ravel()


def committor(C, keep):
    phi = np.degrees(centres()[0][keep])
    inA = (phi > CORE_A[0]) & (phi < CORE_A[1]); inB = (phi > CORE_B[0]) & (phi < CORE_B[1])
    L = np.diag(C.sum(1)) - C
    free = ~(inA | inB); q = np.zeros(len(keep)); q[inB] = 1
    Lff = L[np.ix_(free, free)]; Lff = Lff + 1e-12 * np.trace(Lff) / max(free.sum(), 1) * np.eye(free.sum())   # isolated cells would make it singular
    q[free] = np.linalg.solve(Lff, -L[np.ix_(free, inB)].sum(1))
    return q


def gauss_field(bumps, kT):
    P, S = centres(); f = np.zeros_like(P)
    wrap = lambda d: d - 2 * np.pi * np.floor((d + np.pi) / (2 * np.pi))
    for A, p0, s0, w in bumps:
        f += A / kT * np.exp(-(wrap(P - p0) ** 2 + wrap(S - s0) ** 2) / (2 * w**2))
    return f                                           # in units of kT


def relax_rate(C):
    pi = C.sum(1); T = C / pi[:, None]
    ev = np.sort(np.linalg.eigvals(T).real)[::-1]
    return -np.log(ev[1]) / (LAG * 0.1)               # 1/ps


def predict(C, keep, q, dV, pi):
    d = dV[keep]; i, j = np.nonzero(np.triu(C, 1)); c = C[i, j]
    nu = c * (q[i] - q[j]) ** 2; nu /= nu.sum(); de = 0.5 * (d[i] + d[j])
    A = q < 0.5
    aA = (pi[A] * np.exp(-d[A])).sum() / pi[A].sum(); aB = (pi[~A] * np.exp(-d[~A])).sum() / pi[~A].sum()
    lnG = np.log(pi[A].sum() / aB + pi[~A].sum() / aA)
    lo = lnG - np.log((nu * np.exp(de)).sum()); up = lnG + np.log((nu * np.exp(-de)).sum())
    first = -(nu * de).sum() + (pi[A] * d[A]).sum() / pi[A].sum() * pi[~A].sum() + (pi[~A] * d[~A]).sum() / pi[~A].sum() * pi[A].sum()
    Ch = C * np.exp(-0.5 * (d[:, None] + d[None, :]))
    sqra = np.log(relax_rate(Ch) / relax_rate(C))
    var = (nu * (de - (nu * de).sum()) ** 2).sum()
    return lo, up, first, sqra, var


def cv_force_rmse(pi_k, keep, dV):
    """sqrt < |d dV/d phi|^2 + |d dV/d psi|^2 >_pi  in kT/rad, finite differences on the grid."""
    g = dV.reshape(NB, NB); h = 2 * np.pi / NB
    gx = (np.roll(g, -1, 0) - np.roll(g, 1, 0)) / (2 * h); gy = (np.roll(g, -1, 1) - np.roll(g, 1, 1)) / (2 * h)
    pi = np.zeros(NB * NB); pi[keep] = pi_k
    return np.sqrt((pi * (gx**2 + gy**2).ravel()).sum())


if __name__ == "__main__":
    mode = sys.argv[1]; TAG = sys.argv[3] if len(sys.argv) > 3 else ""
    import os
    GEO = "ref_g1" if glob.glob("traj_ref_g1_s*.npz") else "ref"
    trajs, T = load(GEO); kT = KJ_PER_KT(T)               # geometry (FES, committor, leverage) from the gamma = 1 reference (long trajectories if present)
    C_lag, keep = msm(trajs)
    C = sqra(C_lag, keep) if (len(sys.argv) < 3 or sys.argv[2] == "sqra") else C_lag
    q = committor(C, keep)
    nAB, nBA, tA, tB = md_rates(trajs); k_ref = nAB / tA + nBA / tB
    P, S = centres(); pi = C_lag.sum(1) / C_lag.sum()
    i, j = np.nonzero(np.triu(C, 1)); nu_e = C[i, j] * (q[i] - q[j]) ** 2; nu_e /= nu_e.sum()
    nu = np.bincount(i, nu_e / 2, len(keep)) + np.bincount(j, nu_e / 2, len(keep))
    if mode == "design":
        print(f"reference: T={T:.0f} K, {len(trajs)} trajectories, transitions A->B {nAB}, B->A {nBA};  k_AB={nAB/tA*1e3:.3f}/ns  k_BA={nBA/tB*1e3:.3f}/ns")
        print(f"{len(keep)} bins; lag-MSM relaxation rate {relax_rate(C_lag)*1e3:.3f}/ns  (MD core counting: {k_ref*1e3:.3f}/ns),  population of B side: {pi[q>=0.5].sum():.3f}")
        print("leverage mass with 0.1<q<0.9:", round(float(nu[(q > 0.1) & (q < 0.9)].sum()), 3))
        deg = lambda k: (round(float(np.degrees(P[keep][k]))), round(float(np.degrees(S[keep][k]))))
        ts = int(np.argmax(np.where((q > 0.2) & (q < 0.8), nu, 0))); a = int(np.argmax(np.where(q < 0.5, pi, 0))); b = int(np.argmax(np.where(q >= 0.5, pi, 0)))
        phi_d, psi_d = np.degrees(P[keep]), np.degrees(S[keep])
        c5 = int(np.argmax(np.where((phi_d < -130) & (psi_d > 120), pi, 0)))
        print("leverage peak (TS):", deg(ts), f"pi={pi[ts]:.2e}; basin A:", deg(a), "; basin B:", deg(b), f"pi={pi[b]:.2e}; C5:", deg(c5))
        top = np.argsort(nu)[::-1][:8]; print("top leverage bins:", [(deg(k), round(float(nu[k]), 3)) for k in top])
        W = 0.35; amp_ts = 3.0 * kT
        eps = cv_force_rmse(pi, keep, gauss_field([[amp_ts, P[keep][ts], S[keep][ts], W]], kT))
        design = {}
        for name, k, sign in (("ts_up", ts, +1), ("ts_down", ts, -1), ("basinA", a, +1), ("basinB", b, +1), ("c5", c5, +1)):
            e1 = cv_force_rmse(pi, keep, gauss_field([[kT, P[keep][k], S[keep][k], W]], kT))
            design[name] = [[sign * kT * eps / e1, float(P[keep][k]), float(S[keep][k]), W]]
        json.dump(design, open("design.json", "w"), indent=1)
        print(f"iso CV-force-RMSE = {eps:.4f} kT/rad.  amplitudes (kT):", {n: round(v[0][0] / kT, 3) for n, v in design.items()})
        for n, bmp in design.items():
            lo, up, first, sqra, var = predict(C, keep, q, gauss_field(bmp, kT), pi)
            print(f"   {n:8s} predicted ln(k_hat/k): bracket [{lo:+.3f}, {up:+.3f}]  first-order {first:+.3f}  SqRA-MSM {sqra:+.3f}  beta^2Var_nu={var:.2f}")
    else:
        design = json.load(open("design.json")); rows = []
        if TAG:
            nAB, nBA, tA, tB = md_rates(load("ref" + TAG)[0]); k_ref = nAB / tA + nBA / tB
        print(f"reference: {nAB}+{nBA} transitions, k = {k_ref*1e3:.3f}/ns")
        print("system    | transitions | measured ln(k_hat/k) +- 1 sd | bracket from reference only | first order | SqRA-MSM | inside (within 2 sd)?")
        for n, bmp in design.items():
            try:
                tr, _ = load(n + TAG)
            except IndexError:
                continue
            a, b, ta, tb = md_rates(tr); k = a / ta + b / tb
            meas = np.log(k / k_ref); sd = np.sqrt(1 / max(a + b, 1) + 1 / (nAB + nBA))
            lo, up, first, sqra, var = predict(C, keep, q, gauss_field(bmp, kT), pi)
            ok = (meas + 2 * sd >= lo) and (meas - 2 * sd <= up)
            rows.append((bmp[0][0] / kT, a, b, meas, sd, lo, up, first, sqra))
            print(f"{n:9s} | {a:4d}+{b:<4d}  |   {meas:+.3f} +- {sd:.3f}          |   [{lo:+.3f}, {up:+.3f}]          |  {first:+.3f}    | {sqra:+.3f}  | {'yes' if ok else 'NO'}")
        np.savetxt(f"../data/ala2_stage1{TAG}.csv", rows, delimiter=",", comments="",
                   header="amp_kT,nAB,nBA,measured_ln_ratio,sd,bracket_lower,bracket_upper,first_order,sqra_msm")
