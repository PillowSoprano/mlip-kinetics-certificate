"""Is the high-friction discrepancy in alanine dipeptide explained by non-uniform / anisotropic diffusion in (phi, psi)?
Replace the isotropic nearest-neighbour FES network by the DATA-DRIVEN reversible network  c_ij = pi_i T_ij(lag)  estimated from the
gamma = 50 /ps reference trajectories, apply the perturbation as c_ij -> c_ij exp(-(dV_i + dV_j)/2), and re-solve for the slow rate."""
import json, numpy as np, analyze as a

meas = {"ts_up": {1: -0.712, 10: -0.639, 50: -0.634}, "ts_down": {1: 1.257, 10: 1.189, 50: 1.063}}
design = json.load(open("design.json"))
for g in (50, 10, 1):
    trajs, T = a.load(f"ref_g{g}"); kT = a.KJ_PER_KT(T)
    print(f"\n=== network estimated from the gamma = {g} /ps reference ({len(trajs)} trajectories)")
    print("lag (ps) | relaxation rate of the network (1/ns) | predicted ln(k_hat/k): barrier raised | barrier lowered      [measured: %+.3f | %+.3f]" % (meas["ts_up"][g], meas["ts_down"][g]))
    for lag in (2, 5, 10, 20, 50):
        a.LAG = lag; C, keep = a.msm(trajs)
        k0 = a.relax_rate(C); out = []
        for n in ("ts_up", "ts_down"):
            d = a.gauss_field(design[n], kT)[keep]
            out.append(np.log(a.relax_rate(C * np.exp(-0.5 * (d[:, None] + d[None, :]))) / k0))
        print(f"  {lag*0.1:4.1f}   |   {k0*1e3:7.3f}                              |   {out[0]:+.3f}                              |   {out[1]:+.3f}", flush=True)
    a.LAG = 20; C_lag, keep = a.msm(trajs); Ciso = a.sqra(C_lag, keep); out = []
    for n in ("ts_up", "ts_down"):
        d = a.gauss_field(design[n], kT)[keep]; out.append(np.log(a.relax_rate(Ciso * np.exp(-0.5 * (d[:, None] + d[None, :]))) / a.relax_rate(Ciso)))
    print(f"  isotropic nearest-neighbour FES network:                 |   {out[0]:+.3f}                              |   {out[1]:+.3f}")
