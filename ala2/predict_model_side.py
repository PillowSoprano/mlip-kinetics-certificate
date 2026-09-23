"""
MODEL-SIDE analysis: bracket ln(k_student / k_teacher) from the STUDENT's own reactive paths and teacher single points on those frames.

Exchange the roles of model and reference in the bracket: with nu_hat the student's leverage density (path representation with the student's
committor q_hat, itself from the student's free-energy surface) and dV = E_student - E_teacher,

      -ln G_hat - ln < e^{+dV} >_nu_hat   <=   ln(k_student / k_teacher)   <=   -ln G_hat + ln < e^{-dV} >_nu_hat ,

G_hat = mu_hat(A) / a_hat_B + mu_hat(B) / a_hat_A,  a_hat_X = < e^{+dV} >_{mu_hat|X} = 1 / < e^{-dV} >_{mu|X}   (exact FEP identity, so teacher
equilibrium frames suffice for the basin factors).
"""
import sys, glob
import numpy as np, torch
import analyze as a
from student import Student, teacher_context
from run_md_common import dihedral
import openmm.unit as u

KT = a.KJ_PER_KT(600.0); NSL = 8
ctx, _, _ = teacher_context()
L = np.load("labels.npz"); Xb, Eb = L["X"], L["E"]; phib = np.degrees(np.array([dihedral(x, [4, 6, 8, 14]) for x in Xb]))
inA = (phib > a.CORE_A[0]) & (phib < a.CORE_A[1]); inB = (phib > a.CORE_B[0]) & (phib < a.CORE_B[1])
meas = {"B1": 1.327, "A1": 2.042, "A2": 2.963}
prereg = {"B1": (0.60, -0.12, 0.62), "A1": (0.95, -0.09, 0.76), "A2": (1.30, 0.05, 1.36)}
rows = []
print("student | path frames | dV on STUDENT paths: mean / sd (kT) | model-side bracket        | series law | per-path | MEASURED | reference-side (predicted)")
for name in ("B1", "A1", "A2"):
    ck = torch.load(f"student_{name}.pt"); m = Student(ck["hidden"]); m.load_state_dict(ck["state"]); m.eval()
    # student free-energy surface and committor from its long trajectories, no cis mask
    long_tr = [np.load(f)["phipsi"] for f in sorted(glob.glob(f"traj_stu-{name}_s*.npz"))]
    C_lag, keep = a.msm(long_tr); C = a.sqra(C_lag, keep); q = a.committor(C, keep); pi = C_lag.sum(1) / C_lag.sum()
    Q = np.full(a.NB**2, np.nan); Q[keep] = q; P, _ = a.centres(); pd = np.degrees(P)
    Q[np.isnan(Q) & (pd > a.CORE_A[0]) & (pd < a.CORE_A[1])] = 0.0; Q[np.isnan(Q) & (pd > a.CORE_B[0]) & (pd < a.CORE_B[1])] = 1.0; Q[np.isnan(Q)] = 0.5
    qval = lambda x: Q[np.clip((x[:, 0] + np.pi) / (2 * np.pi) * a.NB, 0, a.NB - 1e-9).astype(int) * a.NB + np.clip((x[:, 1] + np.pi) / (2 * np.pi) * a.NB, 0, a.NB - 1e-9).astype(int)]
    muA, muB = pi[q < 0.5].sum(), pi[q >= 0.5].sum()
    # student reactive paths with coordinates
    ts_x, ts_dq, ts_path, ts_phi = [], [], [], []; npath = ncap = 0
    for f in sorted(glob.glob(f"traj_stutsx-{name}_s*.npz")):
        d = np.load(f); x = d["phipsi"]; phi = np.degrees(x[:, 0]); pos = {int(k): n for n, k in enumerate(d["ts_idx"])}; TSX = d["ts_xyz"]
        st = np.where((phi > a.CORE_A[0]) & (phi < a.CORE_A[1]), 0, np.where((phi > a.CORE_B[0]) & (phi < a.CORE_B[1]), 1, -1)); last, li = -1, -1
        for k, s in enumerate(st):
            if s >= 0:
                if last >= 0 and s != last:
                    qs = qval(x[li:k + 1]); sgn = 1.0 if s == 1 else -1.0; dq = 0.5 * sgn * (np.r_[qs[1:], qs[-1]] - np.r_[qs[0], qs[:-1]]); got = 0.0
                    for j in range(li, k + 1):
                        if j in pos and dq[j - li] != 0:
                            ts_x.append(TSX[pos[j]]); ts_dq.append(dq[j - li]); ts_path.append(npath); ts_phi.append(phi[j]); got += dq[j - li]
                    npath += 1; ncap += got > 0.5
                last, li = s, k
    ts_x = np.array(ts_x, dtype=np.float64); ts_dq = np.array(ts_dq); ts_path = np.array(ts_path); ts_phi = np.array(ts_phi)
    Et = np.empty(len(ts_x))
    for k, xx in enumerate(ts_x):
        ctx.setPositions(xx); Et[k] = ctx.getState(getEnergy=True).getPotentialEnergy().value_in_unit(u.kilojoule_per_mole)
    with torch.no_grad():
        dts = (m.energy(torch.tensor(ts_x)).numpy() - Et) / KT; db = (m.energy(torch.tensor(Xb)).numpy() - Eb) / KT
    ref0 = db[inA].mean(); dts -= ref0; db -= ref0
    ahA, ahB = 1 / np.exp(-db[inA]).mean(), 1 / np.exp(-db[inB]).mean(); lnG = np.log(muA / ahB + muB / ahA)
    w = ts_dq / ts_dq.sum(); wa = np.abs(ts_dq)
    lo = -lnG - np.log((w * np.exp(np.clip(dts, -50, 50))).sum()); up = -lnG + np.log((w * np.exp(np.clip(-dts, -50, 50))).sum())
    o = np.argsort(ts_phi); cw = np.cumsum(wa[o]) / wa.sum(); sl = np.empty(len(o), int); sl[o] = np.minimum((cw * NSL).astype(int), NSL - 1)
    pk = np.array([wa[sl == k].sum() for k in range(NSL)]); pk /= pk.sum()
    bk = np.array([(wa[sl == k] * np.exp(np.clip(dts[sl == k], -50, 50))).sum() / wa[sl == k].sum() for k in range(NSL)])   # <e^{+dV}> per slice
    ser = -lnG + np.log((pk / bk).sum())            # model-side series law: teacher capacity / student capacity <= 1/sum(p/b)  ->  ln(k_s/k_t) >= ...
    mp = np.array([(np.exp(np.clip(-dts[ts_path == p], -50, 50)) * ts_dq[ts_path == p]).sum() / ts_dq[ts_path == p].sum() for p in np.unique(ts_path) if ts_dq[ts_path == p].sum() > 0.5])
    par = -lnG - np.log(np.mean(1.0 / mp))          # per-path (parallel) estimate, model side
    pr = prereg[name]
    print(f"  {name}    | {len(ts_x):5d} ({ncap}/{npath} paths) |      {dts.mean():+.2f} / {dts.std():.2f}                   | [{lo:+.2f}, {up:+.2f}]          |  {ser:+.2f}     | {par:+.2f}    | {meas[name]:+.2f}    | {pr[0]:+.2f} in [{pr[1]:+.2f}, {pr[2]:+.2f}]", flush=True)
    rows.append((ck["rmse_f_meVA"], len(ts_x), npath, dts.mean(), dts.std(), lo, up, ser, par, meas[name], lnG))
np.savetxt("../data/ala2_students_model_side.csv", rows, delimiter=",", comments="",
           header="force_rmse_meVA,n_frames,n_paths,dV_mean_kT,dV_sd_kT,lower,upper,series_law,per_path,measured,lnG_hat")
