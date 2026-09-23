"""Local diffusion tensor in (phi, psi) from short-time displacements (Kramers-Moyal), then a nearest-neighbour network
c_ij = sqrt(pi_i pi_j) * D_edge / h^2  with the MEASURED position-dependent D_phiphi, D_psipsi.  Valid where the CV dynamics is diffusive,
i.e. at high friction (velocity memory 1/gamma = 20 fs at gamma = 50 /ps, frames every 100 fs)."""
import json, numpy as np, analyze as a

NB = a.NB; h = 2 * np.pi / NB
wrap = lambda d: d - 2 * np.pi * np.floor((d + np.pi) / (2 * np.pi))
meas = {"ts_up": {10: -0.639, 50: -0.634}, "ts_down": {10: 1.189, 50: 1.063}}
design = json.load(open("design.json"))
for g in (50, 10):
    trajs, T = a.load(f"ref_g{g}"); kT = a.KJ_PER_KT(T)
    a.LAG = 20; C_lag, keep = a.msm(trajs); pi = C_lag.sum(1) / C_lag.sum()
    print(f"\n=== gamma = {g} /ps     measured: barrier raised {meas['ts_up'][g]:+.3f} | lowered {meas['ts_down'][g]:+.3f}")
    for lag in (1, 2, 4):
        S = np.zeros((NB * NB, 3)); Nn = np.zeros(NB * NB); M = np.zeros((NB * NB, 2))
        for x in trajs:
            ok = ~np.isnan(x[:-lag, 0]) & ~np.isnan(x[lag:, 0]); x0 = x[:-lag][ok]; d = wrap(x[lag:][ok] - x0)
            b = np.clip(((x0 + np.pi) / (2 * np.pi) * NB).astype(int), 0, NB - 1); s = b[:, 0] * NB + b[:, 1]
            np.add.at(Nn, s, 1); np.add.at(M, s, d); np.add.at(S, s, np.column_stack([d[:, 0] ** 2, d[:, 1] ** 2, d[:, 0] * d[:, 1]]))
        m = M / np.maximum(Nn, 1)[:, None]; tau = lag * 0.1
        Dpp = (S[:, 0] / np.maximum(Nn, 1) - m[:, 0] ** 2) / (2 * tau); Dss = (S[:, 1] / np.maximum(Nn, 1) - m[:, 1] ** 2) / (2 * tau)
        Dp, Ds = Dpp[keep], Dss[keep]
        barrier = np.abs(np.degrees(a.centres()[0][keep])) < 25; basin = pi > 1e-3
        pos = {int(k): n for n, k in enumerate(keep)}; Cn = np.zeros_like(C_lag)
        for n, k in enumerate(keep):
            ia, ib = divmod(int(k), NB)
            for k2, D in ((((ia + 1) % NB) * NB + ib, Dp), (ia * NB + (ib + 1) % NB, Ds)):
                mm = pos.get(k2)
                if mm is not None: Cn[n, mm] = Cn[mm, n] = np.sqrt(pi[n] * pi[mm]) * 0.5 * (D[n] + D[mm])
        out = []
        for nme in ("ts_up", "ts_down"):
            d = a.gauss_field(design[nme], kT)[keep]; out.append(np.log(a.relax_rate(Cn * np.exp(-0.5 * (d[:, None] + d[None, :]))) / a.relax_rate(Cn)))
        print(f"  KM lag {tau:.1f} ps: D_phiphi barrier/basin = {np.median(Dp[barrier]):.2f}/{np.median(Dp[basin]):.2f} rad^2/ps, D_psipsi = {np.median(Ds[barrier]):.2f}/{np.median(Ds[basin]):.2f};"
              f"  D_psi/D_phi on the barrier = {np.median(Ds[barrier])/np.median(Dp[barrier]):.2f}   ->  predicted {out[0]:+.3f} | {out[1]:+.3f}", flush=True)
    Ciso = a.sqra(C_lag, keep); out = []
    for nme in ("ts_up", "ts_down"):
        d = a.gauss_field(design[nme], kT)[keep]; out.append(np.log(a.relax_rate(Ciso * np.exp(-0.5 * (d[:, None] + d[None, :]))) / a.relax_rate(Ciso)))
    print(f"  isotropic, uniform D:                                                                                                  ->  predicted {out[0]:+.3f} | {out[1]:+.3f}")
    # how anisotropy alone moves the answer: uniform D but D_psi / D_phi = r
    for r in (0.1, 1, 10, 100):
        Cr = np.zeros_like(C_lag)
        for n, k in enumerate(keep):
            ia, ib = divmod(int(k), NB)
            for k2, D in ((((ia + 1) % NB) * NB + ib, 1.0), (ia * NB + (ib + 1) % NB, r)):
                mm = pos.get(k2)
                if mm is not None: Cr[n, mm] = Cr[mm, n] = np.sqrt(pi[n] * pi[mm]) * D
        out = []
        for nme in ("ts_up", "ts_down"):
            d = a.gauss_field(design[nme], kT)[keep]; out.append(np.log(a.relax_rate(Cr * np.exp(-0.5 * (d[:, None] + d[None, :]))) / a.relax_rate(Cr)))
        print(f"  uniform D with D_psi/D_phi = {r:5.1f}:                                                                                  ->  predicted {out[0]:+.3f} | {out[1]:+.3f}")
