"""Measured rates of the control students (pool half by committee disagreement alone), with the analysis and channel
criterion of measure_AL.py, next to the two-sided strategy for comparison.   python measure_twoc.py"""
import numpy as np, glob
from measure_students import rates, k_of
n0, tA0, tB0, _, _ = rates(sorted(glob.glob("traj_ref_g1_s*.npz"))); k0, N0 = k_of(n0, tA0, tB0, (True, False)); k0n, N0n = k_of(n0, tA0, tB0, (True,))
rows = []
for strat in ("twoc", "two"):
    for seed in (1, 2, 3):
        s = f"AL-{strat}-r3-s{seed}"; files = sorted(glob.glob(f"traj_stu-{s}_s*.npz"))
        if not files: continue
        n, tA, tB, ns, blown = rates(files); kt, N = k_of(n, tA, tB, (True, False)); kn, Nn = k_of(n, tA, tB, (True,))
        nBA = n[("BA", True)] + n[("BA", False)]; sdt, sdn = np.sqrt(1 / N + 1 / N0), np.sqrt(1 / max(Nn, 1) + 1 / N0n)
        tot, ref = np.log(kt / k0), np.log(kn / k0n)
        print(f"  {s:16s} | {ns:5.0f} ns {N:5d} transitions | total {tot:+.2f}({sdt:.2f})  reference pathway {ref:+.2f}({sdn:.2f})  "
              f"fraction reference-like {Nn/N:.2f}  minor-basin dwell {tB/max(nBA,1):.1f} ps", flush=True)
        rows.append((s, tot, sdt, ref, sdn, Nn / N, tB / max(nBA, 1), ns))
with open("../data/ala2_AL_twoc_measured.csv", "w") as f:
    f.write("student,ln_total,sd_total,ln_reference,sd_reference,fraction_reference,dwell_minor_ps,ns\n")
    for r in rows: f.write(",".join(str(x) for x in r) + "\n")
