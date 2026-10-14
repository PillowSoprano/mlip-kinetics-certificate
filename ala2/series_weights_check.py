"""
Series estimate and interval of the alanine students with consistent frame weights.

predict_students.py evaluates the interval with signed committor increments dq on every path frame, but the
series estimate with |dq| inside each committor slice. The two therefore average over different frame weights,
and the series estimate need not lie inside the interval. This script evaluates, from the same frames:
  interval_signed  : the interval of predict_students.py (signed weights)
  series_abs       : the series estimate of predict_students.py (|dq| inside slices)
  interval_abs     : the interval with the slice masses and |dq| weights of series_abs (contains series_abs)
  series_signed    : the series estimate with signed weights inside slices (lies below interval_signed's upper end)
Output: ../data/ala2_students_series_weights.csv
"""
import glob, os
import numpy as np, torch
import analyze as a
from student import Student, teacher_context

KT = a.KJ_PER_KT(600.0)
ref, _ = a.load("ref"); C_lag, keep = a.msm(ref); C = a.sqra(C_lag, keep); q = a.committor(C, keep)
Q = np.full(a.NB**2, np.nan); Q[keep] = q; P, _ = a.centres(); pd = np.degrees(P)
Q[np.isnan(Q) & (pd < -40)] = 0.0; Q[np.isnan(Q) & (pd > 30)] = 1.0; Q[np.isnan(Q)] = 0.5
qval = lambda x: Q[(np.clip((x[:, 0] + np.pi) / (2 * np.pi) * a.NB, 0, a.NB - 1e-9).astype(int)) * a.NB + np.clip((x[:, 1] + np.pi) / (2 * np.pi) * a.NB, 0, a.NB - 1e-9).astype(int)]

CACHE = "series_weights_cache.npz"
if os.path.exists(CACHE):
    c = np.load(CACHE); ts_x, ts_dq, ts_path, ts_phi, E_ts = c["ts_x"], c["ts_dq"], c["ts_path"], c["ts_phi"], c["E_ts"]
else:
    import openmm.unit as u
    ctx, _, _ = teacher_context()
    ts_x, ts_dq, ts_path, ts_phi = [], [], [], []; npath = 0
    for f in sorted(glob.glob("traj_tsx_s*.npz")):
        d = np.load(f); x = d["phipsi"]; phi = np.degrees(x[:, 0]); idx = d["ts_idx"]; pos = {int(k): n for n, k in enumerate(idx)}
        TSX = d["ts_xyz"]
        st = np.where((phi > a.CORE_A[0]) & (phi < a.CORE_A[1]), 0, np.where((phi > a.CORE_B[0]) & (phi < a.CORE_B[1]), 1, -1))
        last, li = -1, -1
        for k, s in enumerate(st):
            if s >= 0:
                if last >= 0 and s != last:
                    qs = qval(x[li:k + 1]); sgn = 1.0 if s == 1 else -1.0
                    dq = 0.5 * sgn * (np.r_[qs[1:], qs[-1]] - np.r_[qs[0], qs[:-1]])
                    for j in range(li, k + 1):
                        if j in pos and dq[j - li] != 0:
                            ts_x.append(TSX[pos[j]]); ts_dq.append(dq[j - li]); ts_path.append(npath); ts_phi.append(phi[j])
                    npath += 1
                last, li = s, k
        print(f"{f}: {npath} paths so far", flush=True)
    ts_x = np.array(ts_x, dtype=np.float64); ts_dq = np.array(ts_dq); ts_path = np.array(ts_path); ts_phi = np.array(ts_phi)
    E_ts = np.empty(len(ts_x))
    for k, xx in enumerate(ts_x):
        ctx.setPositions(xx); E_ts[k] = ctx.getState(getEnergy=True).getPotentialEnergy().value_in_unit(u.kilojoule_per_mole)
        if k % 2000 == 0:
            print(f"teacher energies {k}/{len(ts_x)}", flush=True)
    np.savez(CACHE, ts_x=ts_x, ts_dq=ts_dq, ts_path=ts_path, ts_phi=ts_phi, E_ts=E_ts)
print(f"{len(np.unique(ts_path))} reactive paths, {len(ts_x)} path frames, negative-dq frames {np.mean(ts_dq < 0):.1%}", flush=True)

NSL = 8; o = np.argsort(ts_phi); cw = np.cumsum(ts_dq[o]) / ts_dq.sum()
sl = np.empty(len(o), dtype=int); sl[o] = np.minimum((cw * NSL).astype(int), NSL - 1)

L = np.load("labels.npz"); Xb = L["X"]; Eb = L["E"]
from run_md_common import dihedral
phib = np.degrees(np.array([dihedral(x, [4, 6, 8, 14]) for x in Xb]))
inA = (phib > a.CORE_A[0]) & (phib < a.CORE_A[1]); inB = (phib > a.CORE_B[0]) & (phib < a.CORE_B[1])
piA, piB = inA.mean() / (inA.mean() + inB.mean()), inB.mean() / (inA.mean() + inB.mean())


def estimates(dts, dq, slc, lnG):
    w = dq / dq.sum(); wa = np.abs(dq)
    pk = np.clip(np.array([w[slc == k].sum() for k in range(NSL)]), 1e-6, None); pk /= pk.sum()
    lo_s = lnG - np.log((w * np.exp(dts)).sum()); up_s = lnG + np.log((w * np.exp(-dts)).sum())
    a_abs = np.array([(wa[slc == k] * np.exp(-dts[slc == k])).sum() / wa[slc == k].sum() for k in range(NSL)])
    b_abs = np.array([(wa[slc == k] * np.exp(dts[slc == k])).sum() / wa[slc == k].sum() for k in range(NSL)])
    a_sig = np.array([(dq[slc == k] * np.exp(-dts[slc == k])).sum() / dq[slc == k].sum() for k in range(NSL)])
    ser_abs = lnG - np.log((pk / a_abs).sum())
    lo_abs = lnG - np.log((pk * b_abs).sum()); up_abs = lnG + np.log((pk * a_abs).sum())
    ser_sig = lnG - np.log((pk / a_sig).sum()) if np.all(a_sig > 0) else np.nan
    return lo_s, up_s, ser_abs, lo_abs, up_abs, ser_sig, a_sig.min()


rows, names = [], []
for f in sorted(glob.glob("student_[A-G][0-9].pt")):
    name = f[8:-3]; names.append(name)
    ck = torch.load(f); m = Student(ck["hidden"]); m.load_state_dict(ck["state"]); m.eval()
    with torch.no_grad():
        dts = (m.energy(torch.tensor(ts_x)).numpy() - E_ts) / KT
        db = (m.energy(torch.tensor(Xb)).numpy() - Eb) / KT
    ref0 = db[inA].mean(); dts, db = dts - ref0, db - ref0
    aA, aB = np.exp(-db[inA]).mean(), np.exp(-db[inB]).mean(); lnG = np.log(piA / aB + piB / aA)
    est = estimates(dts, ts_dq, sl, lnG)
    rb = np.random.default_rng(0); ids = np.unique(ts_path); boot = []
    for _ in range(300):
        pick = rb.choice(ids, size=len(ids)); m_ = np.concatenate([np.where(ts_path == p)[0] for p in pick])
        boot.append(estimates(dts[m_], ts_dq[m_], sl[m_], lnG)[:6])
    sd = np.nanstd(np.array(boot), axis=0)
    rows.append((ck["rmse_f_meVA"],) + est[:6] + tuple(sd) + (est[6],))
    lo_s, up_s, ser_abs, lo_abs, up_abs, ser_sig, amin = est
    print(f"{name}: signed [{lo_s:+.2f},{up_s:+.2f}] series_abs {ser_abs:+.2f}({sd[2]:.2f}) | abs [{lo_abs:+.2f},{up_abs:+.2f}] | "
          f"series_signed {ser_sig:+.2f}({sd[5]:.2f}) min slice mean {amin:.3g}", flush=True)

np.savetxt("../data/ala2_students_series_weights.csv", rows, delimiter=",", comments="",
           header="force_rmse_meVA,lower_signed,upper_signed,series_abs,lower_abs,upper_abs,series_signed,"
                  "sd_lower_signed,sd_upper_signed,sd_series_abs,sd_lower_abs,sd_upper_abs,sd_series_signed,min_signed_slice_mean")
open("../data/ala2_students_series_weights_names.txt", "w").write("\n".join(names))
