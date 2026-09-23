"""
Paired far-field test against density-functional theory (Proposition "far field" (iv)).

Each hop-path frame f#### of the teacher-versus-DFT comparison is paired with the configuration of the same replica 0.2 ps
before its reactive segment (p####, the basin frame).  With x = beta (E_teacher - E_DFT) on both, the far field cancels
from the difference x_path - x_pre only if the two errors are correlated; the script reports that correlation, the spreads,
and the barrier error from the paired difference.
    python paired_dft.py
"""
import numpy as np, json, csv, torch
torch.set_num_threads(4)
try: torch.serialization.add_safe_globals([slice])
except Exception: pass
from ase.io import read
from mace.calculators import MACECalculator

T = 1000.0; KB = 8.617333e-5; beta = 1 / (KB * T)
D, P = "../dft", "../dft/paired"
E_path = {r["frame"]: float(r["E0_eV"]) for r in csv.DictReader(open(f"{D}/energies.csv"))}
E_pre = {r["frame"]: float(r["E0_eV"]) for r in csv.DictReader(open(f"{P}/energies.csv"))}
partner = {f"p{m['id']:04d}": m["partner"] for m in json.load(open(f"{P}/frames.json"))}
calc = MACECalculator(model_paths="../colab/li3ocl_teacher.model", device="cpu", default_dtype="float32")

def teacher(path):
    a = read(path); a.calc = calc; return a.get_potential_energy()

pre = sorted(E_pre); x_path, x_pre, frames = [], [], []
for k, p in enumerate(pre):
    f = partner[p]
    x_path.append(beta * (teacher(f"{D}/frames/{f}/POSCAR") - E_path[f]))
    x_pre.append(beta * (teacher(f"{P}/frames/{p}/POSCAR") - E_pre[p]))
    frames.append(f)
    if (k + 1) % 10 == 0: print(f"  {k+1}/{len(pre)} pairs", flush=True)
x_path, x_pre = np.array(x_path), np.array(x_pre)
np.savez("../data/li3ocl_paired.npz", x_path=x_path, x_pre=x_pre, frames=np.array(frames))

d = x_path - x_pre; rng = np.random.default_rng(0)
boot = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(2000)])
print(f"\n{len(d)} pairs, T = {T:.0f} K, x = beta (E_teacher - E_DFT)")
print(f"  correlation of x_path and x_pre   r = {np.corrcoef(x_path, x_pre)[0, 1]:.2f}")
print(f"  spread: x_path {x_path.std():.2f}, x_pre {x_pre.std():.2f}, x_path - x_pre {d.std():.2f}")
print(f"  barrier error from the paired difference: {d.mean() / beta * 1e3:+.0f} +- {boot.std() / beta * 1e3:.0f} meV")
