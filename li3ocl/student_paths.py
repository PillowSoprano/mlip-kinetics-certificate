"""
Estimates from the students' own hop paths with teacher single points (the roles of model and reference exchanged).

For a student S at 1000 K, with dV = E_student - E_teacher of the whole cell and x = beta dV:
    ln(k_S/k_T)  in  [ ln a - ln <e^{+x}>_nu_S ,  ln a + ln <e^{-x}>_nu_S ],   a = <e^{+x}> over the student's basin frames,
and to first order ln(k_S/k_T) = -M_S with M_S = <x>_nu_S - <x>_basins.
nu_S is proportional to exp(+beta F_S) along the hop coordinate s, with F_S the equilibrium free energy of the student:
the s values of the eight lithium ions next to the vacancy in stored frames outside reactive segments, fitted by a cosine
barrier.  Path frames are stratified in 40 bins of s; a bin carries the mass of nu_S and its frames are averaged.
Two ensembles: single-vacancy hops (no Li pair within 1.4 A along the segment), compared with the measured
single-vacancy rate, and all hops, compared with the measured total rate.  Basin frames: 0.2 ps before each hop.
    python student_paths.py S1 S2 S3 S4 S5 S6      ->  ../data/li3ocl_student_paths_1000K.csv
"""
import sys, numpy as np, torch
from scipy.optimize import minimize
torch.set_num_threads(4)
try: torch.serialization.add_safe_globals([slice])
except Exception: pass
sys.path.insert(0, "../colab")
from predict_students import site_energies
from analyze_hops import ideal

T = 1000; beta = 1 / (8.617333e-5 * T); SA, SB, NB, PER_BIN, NBASIN, FUSE = 0.25, 0.75, 40, 30, 150, 1.4
edges = np.linspace(SA, SB, NB + 1); mids = .5 * (edges[1:] + edges[:-1])
teacher = torch.load("../colab/li3ocl_teacher.model", map_location="cpu", weights_only=False)


def min_lili(X, li, L):
    out = np.empty(len(X))
    for k in range(0, len(X), 400):
        y = X[k:k + 400][:, li]; d = y[:, :, None, :] - y[:, None, :, :]; d -= L * np.round(d / L)
        out[k:k + 400] = (np.linalg.norm(d, axis=-1) + np.eye(len(li))[None] * 9).min((1, 2))
    return out


def s_equilibrium(X, Z, L, S):
    """s of the eight ions next to the vacancy, single-vacancy frames only (as equilibrium_coordinate.py)."""
    a = L[0] / 3; pos0, _ = ideal(); li, fw = np.where(Z == 3)[0], np.where(Z != 3)[0]; ref = pos0[fw].mean(0)
    nn = np.linalg.norm((S[:, None] - S[None]) - L * np.round((S[:, None] - S[None]) / L), axis=-1)
    neigh = [np.where((nn[i] > 0.1) & (nn[i] < a / np.sqrt(2) + 0.05))[0] for i in range(len(S))]
    sv = []
    for k0 in range(0, len(X), 500):
        Y = X[k0:k0 + 500][:, li] - (X[k0:k0 + 500][:, fw].mean(1) - ref)[:, None]
        dd = Y[:, :, None, :] - S[None, None]; dd -= L * np.round(dd / L); occ = np.linalg.norm(dd, axis=-1).argmin(-1)
        for k in range(len(Y)):
            vac = np.setdiff1d(np.arange(len(S)), occ[k])
            if len(vac) != 1: continue
            v = vac[0]
            for i in neigh[v]:
                ion = np.where(occ[k] == i)[0]
                if len(ion) != 1: continue
                e = S[v] - S[i]; e -= L * np.round(e / L); u = Y[k, ion[0]] - S[i]; u -= L * np.round(u / L)
                sv.append(u @ e / (e ** 2).sum())
    return np.array(sv)


def cosine_F(sv):
    ce = np.linspace(-0.25, 0.55, 33); cm = .5 * (ce[1:] + ce[:-1]); cnt = np.histogram(sv, bins=ce)[0]
    f = lambda x, B, s0: .5 * B * (1 - np.cos(2 * np.pi * (x - s0) / (1 - 2 * s0)))
    def nll(p):
        B, s0, lnA = p
        if not (0 < B < 60 and -.2 < s0 < .2): return 1e9
        g = f(cm, B, s0); return (np.exp(lnA - g) - cnt * (lnA - g)).sum()
    B, s0, _ = minimize(nll, [6., .04, np.log(max(cnt.max(), 1))], method="Nelder-Mead",
                        options=dict(xatol=1e-8, fatol=1e-8, maxiter=60000)).x
    return B, s0, f(mids, B, s0) - f(mids, B, s0).min()


def estimate(x, b, xb, nu):
    """x: path errors, b: their bin, xb: basin errors, nu: bin masses (normalized over occupied bins)."""
    occ = np.bincount(b, minlength=NB) > 0; m = nu * occ; m = m / m.sum()
    mean_bin = lambda v: np.array([v[b == j].mean() if occ[j] else 0.0 for j in range(NB)])
    c = x.mean()                                                         # constant offsets cancel; centre for stability
    ep, em, xm = mean_bin(np.exp(x - c)), mean_bin(np.exp(-(x - c))), mean_bin(x)
    ln_a = np.log(np.exp(xb - c).mean()) + c
    lo = ln_a - (np.log((m * ep).sum()) + c); up = ln_a + (np.log((m * em).sum()) - c)
    M = (m * xm).sum() - xb.mean()
    sd = np.sqrt((m * (mean_bin(x * x) - xm ** 2 + (xm - (m * xm).sum()) ** 2)).sum())
    return lo, up, -M, sd


rows = []
for name in sys.argv[1:]:
    H = np.load(f"../data/li3ocl_hops_{name}_{T}.npz"); D = np.load(f"../colab/step7_results/hops_{name}_{T}K.npz")
    Z, L, S = D["numbers"], D["cell"], D["sites"]; li = np.where(Z == 3)[0]
    fr, hop, s, pre = H["frame"], H["hop"], H["s"], H["pre_frame"]
    nhop_tot = int(H["hops_per_replica"].sum())
    # mechanism of every complete segment
    dmin = min_lili(D["frames"][fr].astype(np.float64), li, L)
    fused_hop = np.zeros(hop.max() + 1, bool); np.maximum.at(fused_hop, hop, dmin < FUSE)
    # equilibrium F of the student from stored frames outside reactive segments
    out_seg = np.setdiff1d(np.arange(len(D["frames"])), fr)
    rng = np.random.default_rng(3); pick = rng.choice(out_seg, size=min(len(out_seg), 6000), replace=False)
    sv = s_equilibrium(D["frames"][pick].astype(np.float64), Z, L, S); B, s0, F = cosine_F(sv)
    nu = np.exp(F); nu /= nu.sum()
    print(f"{name}: {nhop_tot} hops counted, {hop.max()+1} complete segments, {fused_hop.mean():.0%} not single-vacancy; "
          f"equilibrium s samples {len(sv)}, cosine barrier {B:.2f} kT", flush=True)
    res = {}
    for ens in ("single", "all"):
        okh = ~fused_hop if ens == "single" else np.ones_like(fused_hop)
        inwin = okh[hop] & (s >= SA) & (s < SB); bins = np.clip(np.digitize(s, edges) - 1, 0, NB - 1)
        sel = []
        for j in range(NB):
            i = np.where(inwin & (bins == j))[0]
            if len(i): sel.append(rng.choice(i, min(len(i), PER_BIN), replace=False))
        sel = np.concatenate(sel)
        pb = pre[(pre >= 0)]; pb = pb[okh[np.where(pre >= 0)[0]]] if ens == "single" else pb
        pb = rng.choice(pb, size=min(len(pb), NBASIN), replace=False)
        Xp, Xb = D["frames"][fr[sel]].astype(np.float64), D["frames"][pb].astype(np.float64)
        if ens == "single": Xb = Xb[min_lili(Xb, li, L) >= FUSE]
        res[ens] = (sel, Xp, Xb)
    student = torch.load(f"../colab/step6_results/li3ocl_{name}.model", map_location="cpu", weights_only=False)
    for ens, (sel, Xp, Xb) in res.items():
        dp = site_energies(student, Z, L, Xp).sum(1) - site_energies(teacher, Z, L, Xp).sum(1)
        db = site_energies(student, Z, L, Xb).sum(1) - site_energies(teacher, Z, L, Xb).sum(1)
        x, xb = beta * dp, beta * db; b = np.clip(np.digitize(s[sel], edges) - 1, 0, NB - 1); h = hop[sel]
        lo, up, fo, sd = estimate(x, b, xb, nu)
        boot = []; uh = np.unique(h)
        for _ in range(200):                                              # bootstrap over hops and basin frames
            ph = rng.choice(uh, size=len(uh)); idx = np.concatenate([np.where(h == q)[0] for q in ph])
            ib = rng.integers(0, len(xb), len(xb)); boot.append(estimate(x[idx], b[idx], xb[ib], nu)[:3])
        e = np.nanstd(np.array(boot), axis=0)
        print(f"  {ens:6s}: {len(uh)} hops, {len(x)} path frames, {len(xb)} basin frames; sd of beta dV on paths {x.std():.2f} "
              f"(nu-weighted {sd:.2f}); ln(k_S/k_T) in [{lo:+.2f}({e[0]:.2f}), {up:+.2f}({e[1]:.2f})], first order {fo:+.2f}({e[2]:.2f})", flush=True)
        rows.append((name, ens, len(uh), len(x), len(xb), B, sd, lo, up, fo, e[0], e[1], e[2]))
    import csv, os
    path = f"../data/li3ocl_student_paths_{T}K.csv"; head = ["student", "ensemble", "hops", "path_frames", "basin_frames",
        "barrier_kT", "sd_nu", "lower", "upper", "first_order", "sd_lower", "sd_upper", "sd_first_order"]
    old = [r for r in csv.reader(open(path))][1:] if os.path.exists(path) else []
    done = {(r[0], r[1]) for r in rows}; keep = [r for r in old if (r[0], r[1]) not in done]
    order = {f"S{i}": i for i in range(1, 7)}
    allrows = sorted(keep + [list(map(str, r)) for r in rows], key=lambda r: (order.get(r[0], 9), r[1] != "single"))
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(head); w.writerows(allrows)
