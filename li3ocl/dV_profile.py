"""
The energy error of every student on the teacher's single-vacancy hop paths, resolved along s.

Frames are stratified by s over the reactive window and subsampled within each bin: nu weights bins, and frames
inside one bin are exchangeable, so this is unbiased and costs a quarter of the full evaluation.  The teacher's
site energies are taken from the cache written by predict_students.py.
    python dV_profile.py
"""
import numpy as np, torch, sys
torch.set_num_threads(7)
try: torch.serialization.add_safe_globals([slice])
except Exception: pass
sys.path.insert(0, ".")
from predict_students import site_energies

KB, T = 8.617333e-5, 1000.0; beta = 1 / (KB * T)
SA, SB, NB, PER_BIN, NEQ = 0.25, 0.75, 40, 40, 256
F = np.load("../colab/li3ocl_teacher_1000K_censored.npz"); H = np.load("../data/li3ocl_hops_T1000.npz")
C = np.load("../data/li3ocl_teacher_site_energies_li3ocl_hops_T1000.npz")
Z, L = F["numbers"], F["cell"]; li = np.where(Z == 3)[0]
fr, hop, s, dq = H["frame"], H["hop"], H["s"], H["dq"]

def fused(X):
    out = np.zeros(len(X), bool)
    for k in range(0, len(X), 400):
        y = X[k:k + 400][:, li]; d = y[:, :, None, :] - y[:, None, :, :]; d -= L * np.round(d / L)
        out[k:k + 400] = (np.linalg.norm(d, axis=-1) + np.eye(len(li))[None] * 9).min((1, 2)) < 1.4
    return out

Xall = F["frames"][fr].astype(np.float64)
keep = ~np.isin(hop, list(set(hop[fused(Xall)])))
edges = np.linspace(SA, SB, NB + 1)
rng = np.random.default_rng(11); sel = []; nbin = np.zeros(NB, int)
for j in range(NB):
    i = np.where(keep & (s >= edges[j]) & (s < edges[j + 1]))[0]; nbin[j] = len(i)
    if len(i): sel.append(rng.choice(i, min(len(i), PER_BIN), replace=False))
sel = np.concatenate(sel)
print(f"{keep.sum()} single-vacancy frames, {len(set(hop[keep]))} hops; {nbin.sum()} in the window, "
      f"{len(sel)} evaluated ({(nbin>0).sum()} of {NB} bins occupied)", flush=True)

X = Xall[sel]; Ep_t = C["ts"][sel].sum(1)
Xe = np.load("../colab/step6_results/eq_1000K.npz")["X"].astype(np.float64)
ok_e = ~fused(Xe); ie = np.where(ok_e)[0][:NEQ]
Xe, Ee_t = Xe[ie], C["eq"][ie].sum(1)
out = {"s": s[sel], "dq": dq[sel], "hop": hop[sel], "bin_count": nbin, "edges": edges}
for n in ("S1", "S2", "S3", "S4", "S5", "S6"):
    st = torch.load(f"../colab/step6_results/li3ocl_{n}.model", map_location="cpu", weights_only=False)
    out[f"path_{n}"] = beta * (site_energies(st, Z, L, X).sum(1) - Ep_t)
    out[f"eq_{n}"] = beta * (site_energies(st, Z, L, Xe).sum(1) - Ee_t)
    print(f"{n}: <beta dV>_mu = {out['eq_'+n].mean():+.3f}, path mean {out['path_'+n].mean():+.3f}, "
          f"near the top (|s-0.5|<0.05) {out['path_'+n][np.abs(out['s']-0.5)<0.05].mean():+.3f}", flush=True)
np.savez_compressed("../data/li3ocl_dV_profile_1000K.npz", **out)
print("saved", flush=True)
