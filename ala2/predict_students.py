"""
Predict ln(k_student / k_teacher) for every trained student WITHOUT running the student:
   dV = E_student - E_teacher  on (i) equilibrium frames of the two basins, (ii) Cartesian frames along teacher transition paths,
   leverage averages from the path representation  <g>_nu = E_reactive[ sum_t g(x_t) dq_t ],  q = committor values from the teacher's FES model.
Output: ../data/ala2_students_predicted.csv   (to be compared with direct student MD)
"""
import glob, sys
import numpy as np, torch
import analyze as a
from student import Student, teacher_context

KT = a.KJ_PER_KT(600.0)
# ---- committor values on the (phi, psi) grid, from the teacher reference only
ref, _ = a.load("ref"); C_lag, keep = a.msm(ref); C = a.sqra(C_lag, keep); q = a.committor(C, keep)
Q = np.full(a.NB**2, np.nan); Q[keep] = q; P, _ = a.centres(); pd = np.degrees(P)
Q[np.isnan(Q) & (pd < -40)] = 0.0; Q[np.isnan(Q) & (pd > 30)] = 1.0; Q[np.isnan(Q)] = 0.5
qval = lambda x: Q[(np.clip((x[:, 0] + np.pi) / (2 * np.pi) * a.NB, 0, a.NB - 1e-9).astype(int)) * a.NB + np.clip((x[:, 1] + np.pi) / (2 * np.pi) * a.NB, 0, a.NB - 1e-9).astype(int)]

# ---- transition-path frames with coordinates (teacher runs with ALA_TSXYZ=1) and their committor increments
import openmm.unit as u
ctx, _, _ = teacher_context()
ts_x, ts_dq, ts_path, ts_phi = [], [], [], []; npath = 0
for f in sorted(glob.glob("traj_tsx_s*.npz")):
    d = np.load(f); x = d["phipsi"]; phi = np.degrees(x[:, 0]); idx = d["ts_idx"]; pos = {int(k): n for n, k in enumerate(idx)}
    TSX = d["ts_xyz"]                                  # load ONCE: indexing the NpzFile decompresses the whole array every time
    st = np.where((phi > a.CORE_A[0]) & (phi < a.CORE_A[1]), 0, np.where((phi > a.CORE_B[0]) & (phi < a.CORE_B[1]), 1, -1))
    last, li = -1, -1
    for k, s in enumerate(st):
        if s >= 0:
            if last >= 0 and s != last:
                qs = qval(x[li:k + 1]); sgn = 1.0 if s == 1 else -1.0
                dq = 0.5 * sgn * (np.r_[qs[1:], qs[-1]] - np.r_[qs[0], qs[:-1]])          # centred increments, sum = +-(q_end - q_start)
                for j in range(li, k + 1):
                    if j in pos and dq[j - li] != 0:
                        ts_x.append(TSX[pos[j]]); ts_dq.append(dq[j - li]); ts_path.append(npath); ts_phi.append(phi[j])
                npath += 1
            last, li = s, k
ts_x = np.array(ts_x, dtype=np.float64); ts_dq = np.array(ts_dq); ts_path = np.array(ts_path); ts_phi = np.array(ts_phi)
# slices along the reaction coordinate with equal leverage mass (for the series law: FEP over orthogonal dof inside a slice, harmonic across slices)
NSL = 8; o = np.argsort(ts_phi); cw = np.cumsum(ts_dq[o]) / ts_dq.sum(); sl = np.empty(len(o), dtype=int); sl[o] = np.minimum((cw * NSL).astype(int), NSL - 1)
E_ts = np.empty(len(ts_x))
for k, xx in enumerate(ts_x):
    ctx.setPositions(xx); E_ts[k] = ctx.getState(getEnergy=True).getPotentialEnergy().value_in_unit(u.kilojoule_per_mole)
print(f"{npath} reactive paths, {len(ts_x)} path frames with coordinates, captured committor change per path = {ts_dq.sum()/npath:.3f}")

# ---- basin frames (equilibrium, already labelled)
L = np.load("labels.npz"); Xb = L["X"]; Eb = L["E"]
from run_md_common import dihedral
PHI = [4, 6, 8, 14]
phib = np.degrees(np.array([dihedral(x, PHI) for x in Xb]))
inA = (phib > a.CORE_A[0]) & (phib < a.CORE_A[1]); inB = (phib > a.CORE_B[0]) & (phib < a.CORE_B[1])
piA, piB = inA.mean() / (inA.mean() + inB.mean()), inB.mean() / (inA.mean() + inB.mean())
print(f"basin frames: A {inA.sum()}, B {inB.sum()}")

rows = []
print("student | force RMSE meV/A | dV spread on path frames (kT) | bracket [lower, upper] | first order | per-path estimate")
PATTERN = sys.argv[1] if len(sys.argv) > 1 else "student_[A-G][0-9].pt"; OUT = sys.argv[2] if len(sys.argv) > 2 else "ala2_students_predicted"
NAMES = []
for f in sorted(glob.glob(PATTERN)):
    name = f[8:-3]; NAMES.append(name)
    ck = torch.load(f); m = Student(ck["hidden"]); m.load_state_dict(ck["state"]); m.eval()
    with torch.no_grad():
        dts = (m.energy(torch.tensor(ts_x)).numpy() - E_ts) / KT
        db = (m.energy(torch.tensor(Xb)).numpy() - Eb) / KT
    ref0 = db[inA].mean(); dts, db = dts - ref0, db - ref0
    aA, aB = np.exp(-db[inA]).mean(), np.exp(-db[inB]).mean()
    lnG = np.log(piA / aB + piB / aA)
    w = ts_dq / ts_dq.sum()
    lo, up = lnG - np.log((w * np.exp(dts)).sum()), lnG + np.log((w * np.exp(-dts)).sum())
    first = -(w * dts).sum() + piB * db[inA].mean() + piA * db[inB].mean()
    mp = np.array([(np.exp(dts[ts_path == p]) * ts_dq[ts_path == p]).sum() / max(ts_dq[ts_path == p].sum(), 1e-9) for p in np.unique(ts_path) if ts_dq[ts_path == p].sum() > 0.5])
    par = lnG + np.log(np.mean(1.0 / mp))
    # inside a slice use POSITIVE weights |dq| (signed increments are only unbiased in expectation and make slice averages unstable)
    wa = np.abs(ts_dq); pk = np.array([w[sl == k].sum() for k in range(NSL)]); pk = np.clip(pk, 1e-6, None); pk /= pk.sum()
    ak = np.array([(wa[sl == k] * np.exp(-dts[sl == k])).sum() / wa[sl == k].sum() for k in range(NSL)])
    ser = lnG - np.log((pk / ak).sum())                          # series law = the answer if the orthogonal dof are fast
    up_abs = lnG + np.log((pk * ak).sum())                       # frozen upper bound with the same weights (must be >= series law)
    # bootstrap over reactive paths (frames inside a path are correlated, paths are independent)
    rb = np.random.default_rng(0); ids = np.unique(ts_path); boot = []
    for _ in range(300):
        pick = rb.choice(ids, size=len(ids)); m_ = np.concatenate([np.where(ts_path == p)[0] for p in pick]); wb = ts_dq[m_] / ts_dq[m_].sum(); wab = np.abs(ts_dq[m_]); db_ = dts[m_]; slb = sl[m_]
        pkb = np.clip(np.array([wb[slb == k].sum() for k in range(NSL)]), 1e-6, None); pkb /= pkb.sum()
        akb = np.array([(wab[slb == k] * np.exp(-db_[slb == k])).sum() / max(wab[slb == k].sum(), 1e-12) for k in range(NSL)])
        boot.append((lnG - np.log((wb * np.exp(db_)).sum()), lnG + np.log((wb * np.exp(-db_)).sum()), lnG - np.log((pkb / np.maximum(akb, 1e-12)).sum())))
    boot = np.array(boot); sd_lo, sd_up, sd_ser = np.nanstd(boot, axis=0); neff = (np.abs(ts_dq).sum() ** 2) / (ts_dq**2).sum()
    rows.append((ck["rmse_f_meVA"], ck["rmse_e_kJ"] / KT, dts.std(), lo, up, first, par, db[inB].mean() - db[inA].mean(), ser, sd_lo, sd_up, sd_ser))
    print(f"{name:7s} | {ck['rmse_f_meVA']:8.1f}         |  {dts.std():.2f}   | [{lo:+.2f}, {up:+.2f}] | {first:+.2f} | {par:+.2f} | series law {ser:+.2f} +- {sd_ser:.2f} (<= {up_abs:+.2f}) | sd of bounds {sd_lo:.2f}/{sd_up:.2f}")
np.savetxt(f"../data/{OUT}.csv", rows, delimiter=",", comments="",
           header="force_rmse_meVA,energy_rmse_kT,dV_sd_on_paths_kT,bracket_lower,bracket_upper,first_order,per_path,basinB_minus_basinA_offset_kT,series_law,sd_lower,sd_upper,sd_series")
open(f"../data/{OUT}_names.txt", "w").write("\n".join(NAMES))
