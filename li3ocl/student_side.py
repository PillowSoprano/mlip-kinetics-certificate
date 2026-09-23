"""Student-side check: teacher single points on the STUDENT's own hop-path frames.   python student_side.py S6 1000"""
import sys, numpy as np, torch
torch.set_num_threads(4)
try: torch.serialization.add_safe_globals([slice])
except Exception: pass
sys.path.insert(0, "../colab")
from predict_students import site_energies
name, T = sys.argv[1], int(sys.argv[2]); beta = 1 / (8.617333e-5 * T); rng = np.random.default_rng(0)
H = np.load(f"../data/li3ocl_hops_{name}_{T}.npz"); F = np.load(f"../colab/step7_results/hops_{name}_{T}K.npz"); Z, L = F["numbers"], F["cell"]
hops = np.unique(H["hop"]); sel = rng.choice(hops, size=min(120, len(hops)), replace=False); m = np.isin(H["hop"], sel); fr, s, dq, hop = H["frame"][m], H["s"][m], H["dq"][m], H["hop"][m]
pre = H["pre_frame"][sel]; pre = pre[pre >= 0]; X, Xp = F["frames"][fr].astype(np.float64), F["frames"][pre].astype(np.float64)
teacher = torch.load("../colab/li3ocl_teacher.model", map_location="cpu", weights_only=False); student = torch.load(f"../colab/step6_results/li3ocl_{name}.model", map_location="cpu", weights_only=False)
d = (site_energies(student, Z, L, X).sum(1) - site_energies(teacher, Z, L, X).sum(1)); dp = (site_energies(student, Z, L, Xp).sum(1) - site_energies(teacher, Z, L, Xp).sum(1)); off = np.median(dp)
x = beta * (d - off); xp = beta * (dp - off); w = np.abs(dq) / np.abs(dq).sum()
print(f"{name} {T} K, {len(sel)} of the student's own hops, {len(X)} path frames, {len(Xp)} basin frames 0.2 ps before a hop")
print(f"  beta*(E_student - E_teacher), relative to the student's basin frames:  basin sd {xp.std():.2f};  hop paths: leverage-weighted mean {np.sum(w*x):+.2f}, sd {x.std():.2f}, 5%/50%/95% {np.quantile(x,.05):+.1f}/{np.median(x):+.1f}/{np.quantile(x,.95):+.1f}")
top = np.abs(s - 0.5) < 0.1; print(f"  at the top of the hop (|s-0.5|<0.1, {top.sum()} frames): mean {x[top].mean():+.2f}, sd {x[top].std():.2f}   -> negative = the student's barrier is too LOW by that many kT")
np.savez(f"../data/li3ocl_student_side_{name}_{T}.npz", x=x, xp=xp, s=s, dq=dq, hop=hop)
