"""Measured rates of the active-learning students (same analysis and the same, unchanged channel criterion as measure_students.py).  python measure_AL.py"""
import numpy as np, glob, re
from measure_students import rates, k_of
P = np.loadtxt("../data/ala2_AL_predicted.csv", delimiter=",", skiprows=1); NAMES = open("../data/ala2_AL_predicted_names.txt").read().split()
pre = {nm: P[i] for i, nm in enumerate(NAMES)}
for f in ["../data/ala2_ALtwo23_predicted.csv", "../data/ala2_ALtwo_predicted.csv"]:
    Q = np.loadtxt(f, delimiter=",", skiprows=1, ndmin=2); nm_ = open(f.replace(".csv", "_names.txt")).read().split()
    for i, nm in enumerate(nm_): pre.setdefault(nm, Q[i])
for f in ["../data/ala2_ALseeds23_predicted.csv", "../data/ala2_ALtwo_predicted.csv"]:
    Q = np.loadtxt(f, delimiter=",", skiprows=1, ndmin=2); nm_ = open(f.replace(".csv", "_names.txt")).read().split()
    for i, nm in enumerate(nm_): pre.setdefault(nm, Q[i])
n0, tA0, tB0, _, _ = rates(sorted(glob.glob("traj_ref_g1_s*.npz"))); k0, N0 = k_of(n0, tA0, tB0, (True, False)); k0n, N0n = k_of(n0, tA0, tB0, (True,))
names = sorted({re.search(r"traj_stu-(AL-\w+-r\d-s\d)_s", f).group(1) for f in glob.glob("traj_stu-AL-*_s*.npz")}); rows = []
print("student           rmse | ns  trans | total        teacher-like  frac t-l | dwell B (ps) | predicted series [bracket]")
for s in names:
    key = s if s in pre else s.replace("lev-r0", "random-r0").replace("unc-r0", "random-r0"); p = pre.get(key, pre.get(s.replace("-s1", ""), None))
    files = sorted(glob.glob(f"traj_stu-{s}_s*.npz")); n, tA, tB, ns, blown = rates(files)
    if ns < 0.5 * 15 * len(files):
        print(f"  {s:16s} | {ns:4.0f} ns of {15*len(files):4d}: numerically unstable, no rate"); rows.append((s, *([np.nan] * 6), *(pre.get(s, [np.nan]*9)[[0,3,4,8]] if s in pre else [np.nan]*4), len(files))); continue
    kt, N = k_of(n, tA, tB, (True, False)); kn, Nn = k_of(n, tA, tB, (True,))
    nBA = n[("BA", True)] + n[("BA", False)]; sdt, sdn = np.sqrt(1 / N + 1 / N0), np.sqrt(1 / max(Nn, 1) + 1 / N0n)
    ptxt = f"{p[8]:+.2f} [{p[3]:+.2f},{p[4]:+.2f}]  rmse {p[0]:.0f}" if p is not None else "(no prediction found)"
    print(f"  {s:16s} | {ns:4.0f} {N:6d} | {np.log(kt/k0):+.2f}({sdt:.2f})  {np.log(kn/k0n):+.2f}({sdn:.2f})   {Nn/N:.2f}   | {tB/max(nBA,1):7.1f}     | {ptxt}", flush=True)
    rows.append((s, np.log(kt / k0), sdt, np.log(kn / k0n), sdn, Nn / N, tB / max(nBA, 1), *(p[[0, 3, 4, 8]] if p is not None else [np.nan] * 4), blown))
with open("../data/ala2_AL_measured.csv", "w") as f:
    f.write("name,total,sd_total,teacherlike,sd_teacherlike,frac_teacherlike,dwellB_ps,rmse,lower,upper,pred,diverged\n")
    for r in rows: f.write(",".join(str(x) if isinstance(x, str) else f"{x:.4f}" for x in r) + "\n")
