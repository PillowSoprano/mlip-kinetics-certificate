"""
Where does the teacher's contrast against DFT come from?  Splits M(teacher - DFT) on the 211 single points into
M(teacher - MACE-MP-0) + M(MACE-MP-0 - DFT), with MACE-MP-0 (small), the model whose labels the teacher was
distilled from.

    python mp0_vs_dft.py            (base env with mace-torch; reads ../data/li3ocl_teacher_vs_dft.npz and ../dft)

Output: ../data/li3ocl_mp0_vs_dft.csv (one row per frame) and the three contrasts with bootstrap errors.
"""
import numpy as np, os
from ase.io import read
from mace.calculators import mace_mp

D = "../dft"; KB = 8.617333e-5; T = 1000.0; beta = 1 / (KB * T)
d = np.load("../data/li3ocl_teacher_vs_dft.npz"); Et, Ed, kind, frame = d["E_teacher"], d["E_dft"], d["kind"], d["frame"]
calc = mace_mp(model="small", device="cpu", default_dtype="float64")          # the labels of the teacher came from this model
Em = []
for k, f in enumerate(frame):
    a = read(f"{D}/frames/{f}/POSCAR"); a.calc = calc; Em.append(a.get_potential_energy())
    if (k + 1) % 50 == 0: print(f"  {k+1}/{len(frame)}", flush=True)
Em = np.array(Em); P, E = kind == "path", kind == "eq"
np.savetxt("../data/li3ocl_mp0_vs_dft.csv", np.c_[np.arange(len(frame)), P.astype(int), Et, Em, Ed], delimiter=",",
           header="row,is_path,E_teacher_eV,E_mp0_eV,E_dft_eV", comments="", fmt=["%d", "%d", "%.6f", "%.6f", "%.6f"])

rng = np.random.default_rng(0); B = 2000
idx = [(rng.integers(0, P.sum(), P.sum()), rng.integers(0, E.sum(), E.sum())) for _ in range(B)]
def contrast(dv):
    xp, xe = dv[P], dv[E]; m = xp.mean() - xe.mean()
    bs = np.array([xp[i].mean() - xe[j].mean() for i, j in idx])
    return 1e3 * m, 1e3 * bs.std(), 1e3 * xe.std(), 1e3 * xp.std()
for name, dv in (("teacher - DFT", Et - Ed), ("MACE-MP-0 - DFT", Em - Ed), ("teacher - MACE-MP-0", Et - Em)):
    m, s, se, sp = contrast(dv)
    print(f"{name:20s} M = {m:+6.1f} +- {s:4.1f} meV   (beta M = {beta*m/1e3:+.2f});  frame-to-frame sd: basins {se:5.1f}, paths {sp:5.1f} meV")
