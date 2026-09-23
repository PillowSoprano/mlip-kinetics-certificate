"""
Measured rates of the neural-network students, with the decomposition into teacher-like and shortcut transitions.

    python measure_students.py            (all students that have trajectories traj_stu-NAME_s*.npz; no cis mask)

teacher-like transition: core-to-core in phi through phi ~ 0 (|phi| <= 150 deg along the segment) with every |omega| >= OM_CUT within +-1 ps.
The criterion was fixed on B1/A1/A2 (post hoc for those three); it is applied UNCHANGED to the students measured later.
Writes ../data/ala2_students_measured_all.csv and ../figs/data/fig10_students.csv
"""
import numpy as np, glob, re, os
OM_CUT = 120.0

PRE = "../data/ala2_students_predicted_with_errorbars.csv"; NAMES = open("../data/ala2_students_predicted_with_errorbars_names.txt").read().split()

def rates(files, om_cut=OM_CUT):
    n = {("AB", True): 0, ("AB", False): 0, ("BA", True): 0, ("BA", False): 0}; tA = tB = 0.0; ns = 0.0; blown = 0
    for f in files:
        d = np.load(f); phi = np.degrees(d["phipsi"][:, 0]); om = np.degrees(np.abs(d["omega"]))
        if len(phi) < 1000 or len(om) == 0 or not np.isfinite(phi).all(): blown += 1; continue      # diverged trajectory (short or NaN)
        rep = int(np.ceil(len(phi) / len(om))); ns += len(phi) * 1e-4
        st = np.where((phi > -150) & (phi < -40), 0, np.where((phi > 30) & (phi < 110), 1, -1)); last, li = -1, -1
        for k, s in enumerate(st):
            if s >= 0:
                if last >= 0 and s != last:
                    seg = phi[li:k + 1]; j0, j1 = max(li // rep - 1, 0), min(k // rep + 2, len(om))
                    n[("AB" if s == 1 else "BA", (np.abs(seg).max() <= 150) and (om[j0:j1].min() >= om_cut))] += 1
                last, li = s, k
            if last == 0: tA += 0.1
            elif last == 1: tB += 0.1
    return n, tA, tB, ns, blown

def k_of(n, tA, tB, which):
    ab = sum(n[("AB", w)] for w in which); ba = sum(n[("BA", w)] for w in which); return ab / tA + ba / tB, ab + ba

if __name__ == "__main__":
    P = np.loadtxt(PRE, delimiter=",", skiprows=1); pre = {nm: P[i] for i, nm in enumerate(NAMES)}     # cols: rmse, .., 3 lower, 4 upper, .., 8 series, 9 sd_lo, 10 sd_up, 11 sd_series
    n0, tA0, tB0, ns0, _ = rates(sorted(glob.glob("traj_ref_g1_s*.npz"))); k0, N0 = k_of(n0, tA0, tB0, (True, False)); k0n, N0n = k_of(n0, tA0, tB0, (True,))
    print(f"teacher: {ns0:.0f} ns, {N0} transitions, teacher-like fraction {N0n / N0:.3f}")
    students = sorted({re.search(r"traj_stu-(\w+?)_s", f).group(1) for f in glob.glob("traj_stu-*_s*.npz")}, key=lambda s: pre[s][0])
    rows = []; done = []; print("student rmse | ns  trans blown | total        teacher-like   shortcut | predicted series [bracket]       | cut 100 / 140")
    for s in students:
        files = sorted(glob.glob(f"traj_stu-{s}_s*.npz")); n, tA, tB, ns, blown = rates(files)
        if tA == 0 or tB == 0 or sum(n.values()) < 20 or 2 * blown >= len(files): print(f"  {s}  {pre[s][0]:6.1f} | {ns:4.0f} ns, {sum(n.values())} transitions, {blown}/{len(files)} trajectories diverged: no rate"); continue
        kt, N = k_of(n, tA, tB, (True, False)); kn, Nn = k_of(n, tA, tB, (True,)); ks = kt - kn
        sdt, sdn = np.sqrt(1 / N + 1 / N0), np.sqrt(1 / max(Nn, 1) + 1 / N0n); sens = []
        for cut in (100.0, 140.0):
            m, a, b, _, _ = rates(files, cut); sens.append(np.log(k_of(m, a, b, (True,))[0] / k0n))
        p = pre[s]; print(f"  {s}  {p[0]:6.1f} | {ns:4.0f} {N:5d} {blown}/{len(files)} | {np.log(kt/k0):+.2f}({sdt:.2f})  {np.log(kn/k0n):+.2f}({sdn:.2f})   {np.log(ks/k0) if ks > 0 else float('nan'):+.2f}  | {p[8]:+.2f}({p[11]:.2f}) [{p[3]:+.2f},{p[4]:+.2f}] | {sens[0]:+.2f} / {sens[1]:+.2f}", flush=True)
        done.append(s); rows.append((p[0], np.log(kt / k0), sdt, np.log(kn / k0n), sdn, np.log(ks / k0) if ks > 0 else np.nan, Nn / N, p[8], p[11], p[3], p[4], sens[0], sens[1], ns, N, blown))
    hdr = "rmse,total,sd_total,teacherlike,sd_teacherlike,shortcut,frac_teacherlike,pred,pred_sd,lower,upper,tl_cut100,tl_cut140,ns,transitions,diverged"
    np.savetxt("../data/ala2_students_measured_all.csv", rows, delimiter=",", header=hdr, comments="")
    with open("../figs/data/fig10_students.csv", "w") as f:
        f.write("name,rmse,pred,pred_sd,upper,total,teacherlike,tl_sd\n")
        for s, r in zip(done, rows): f.write(f"{s},{r[0]:.1f},{r[7]:.4f},{r[8]:.4f},{r[10]:.4f},{r[1]:.4f},{r[3]:.4f},{r[4]:.4f}\n")
    R = np.array(rows)
    if len(R) >= 4:
        rk = lambda v: np.argsort(np.argsort(v)); sp = lambda a, b: np.corrcoef(rk(a), rk(b))[0, 1]
        print(f"\nrank correlation with measured TOTAL ln ratio:   force RMSE {sp(R[:,0],R[:,1]):+.2f}   predicted series law {sp(R[:,7],R[:,1]):+.2f}")
        print(f"rank correlation with measured TEACHER-LIKE:      force RMSE {sp(R[:,0],R[:,3]):+.2f}   predicted series law {sp(R[:,7],R[:,3]):+.2f}")
        print(f"inside predicted bracket (2 sd): total {int(((R[:,1] <= R[:,10] + 2*R[:,2]) & (R[:,1] >= R[:,9] - 2*R[:,2])).sum())}/{len(R)},  teacher-like {int(((R[:,3] <= R[:,10] + 2*R[:,4]) & (R[:,3] >= R[:,9] - 2*R[:,4])).sum())}/{len(R)}")
