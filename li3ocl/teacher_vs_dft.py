"""
Bracket the hop rate of density-functional theory from the TEACHER's own hop paths.
Model = teacher (MACE), reference = DFT, dV = E_teacher - E_DFT.  Model-side form of Theorem 1:

    ln a_hat - ln <e^{+beta dV}>_nu_hat  <=  ln(Gamma_teacher / Gamma_DFT)  <=  ln a_hat + ln <e^{-beta dV}>_nu_hat,
    a_hat = <e^{+beta dV}>_{equilibrium frames of the teacher}.

Any constant offset between the two energy zeros cancels between the two terms.
The 131 path frames are one frame per hop, drawn with probability |dq| inside the hop: an unbiased, equally weighted sample of nu_hat.
    python teacher_vs_dft.py
"""
import numpy as np, json, csv, torch
torch.set_num_threads(4)
try: torch.serialization.add_safe_globals([slice])
except Exception: pass
from ase.io import read
from mace.calculators import MACECalculator

T = 1000.0; KB = 8.617333e-5; beta = 1 / (KB * T); NSL = 8
D = "../dft"
rows = list(csv.DictReader(open(f"{D}/energies.csv"))); meta = {f"f{r['id']:04d}": r for r in json.load(open(f"{D}/frames.json"))}
calc = MACECalculator(model_paths="../colab/li3ocl_teacher.model", device="cpu", default_dtype="float32")
Et, Ed, kind, s = [], [], [], []
for k, r in enumerate(rows):
    a = read(f"{D}/frames/{r['frame']}/POSCAR"); a.calc = calc
    Et.append(a.get_potential_energy()); Ed.append(float(r["E0_eV"])); m = meta[r["frame"]]; kind.append(m["kind"]); s.append(m.get("s", np.nan))
    if (k + 1) % 50 == 0: print(f"  {k+1}/{len(rows)} frames", flush=True)
Et, Ed, kind, s = np.array(Et), np.array(Ed), np.array(kind), np.array(s, dtype=float)
dV = Et - Ed; P, E = kind == "path", kind == "eq"
np.savez("../data/li3ocl_teacher_vs_dft.npz", E_teacher=Et, E_dft=Ed, kind=kind, s=s, frame=np.array([r["frame"] for r in rows]))

off = dV[E].mean(); x = beta * (dV - off)                       # constant offset removed for readability; it cancels in the bracket
print(f"\n{P.sum()} hop-path frames (one per hop), {E.sum()} equilibrium frames, T = {T:.0f} K")
print(f"beta*(E_teacher - E_DFT), relative to the mean over equilibrium frames:")
print(f"  equilibrium : mean {x[E].mean():+.2f}, sd {x[E].std():.2f}   ({x[E].std()/beta*1e3:.0f} meV per cell)")
print(f"  hop paths   : mean {x[P].mean():+.2f}, sd {x[P].std():.2f}   ({x[P].std()/beta*1e3:.0f} meV per cell)")
print(f"  at the top of the hop (|s-0.5|<0.15, {(P & (abs(s-0.5)<0.15)).sum()} frames): mean {x[P & (abs(s-0.5)<0.15)].mean():+.2f}")

def estimates(xp, xe, sp):
    la = np.log(np.exp(xe).mean())
    lo = la - np.log(np.exp(xp).mean()); up = la + np.log(np.exp(-xp).mean())
    o = np.argsort(sp); sl = np.minimum((np.arange(len(o)) * NSL // len(o)), NSL - 1); lab = np.empty(len(o), int); lab[o] = sl
    ak = np.array([np.exp(-xp[lab == k]).mean() for k in range(NSL)])
    ser = la - np.log(np.mean(1 / ak)); first = xe.mean() - xp.mean()      # first order on both sides: -(<x>_nu - <x>_mu)
    return lo, up, ser, first

c = estimates(x[P], x[E], s[P]); rng = np.random.default_rng(0)
B = np.array([estimates(x[P][i], x[E][j], s[P][i]) for i, j in ((rng.integers(0, P.sum(), P.sum()), rng.integers(0, E.sum(), E.sum())) for _ in range(400))])
sd = B.std(0)
print(f"\nln(Gamma_teacher / Gamma_DFT), single-vacancy mechanism at {T:.0f} K:")
print(f"  bracket      [{c[0]:+.2f} +- {sd[0]:.2f}, {c[1]:+.2f} +- {sd[1]:.2f}]   width {c[1]-c[0]:.2f}")
print(f"  series law    {c[2]:+.2f} +- {sd[2]:.2f}")
print(f"  first order   {c[3]:+.2f} +- {sd[3]:.2f}")
print(f"  validity of first order: beta^2 Var_nu(dV) = {x[P].var():.2f}  (first order is valid when this is << 1)")
np.savetxt("../data/li3ocl_teacher_vs_dft_bracket.csv", [[T, P.sum(), E.sum(), x[P].mean(), x[P].std(), x[E].std(), *c, *sd]], delimiter=",", comments="",
           header="T_K,n_path,n_eq,mean_x_path,sd_x_path,sd_x_eq,lower,upper,series,first,sd_lower,sd_upper,sd_series,sd_first")
