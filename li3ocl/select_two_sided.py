"""Two-sided acquisition for Li3OCl student S6 (the student with the widest bracket on the pooled ensemble): choose 200 new configurations, no labels yet.
   100 reference side: frames of the TEACHER's hop paths (1000 K and 900 K), drawn with probability ~ |dq| (leverage density in the path representation)
   100 model side    :  50 frames of S6's OWN hop paths (~ |dq|), 50 frames of S6 trajectories in which a site is doubly occupied (the extra-defect state), all three temperatures
   Teacher labels are computed afterwards (colab/li3ocl_step8_acquisition.ipynb). Control S6c: 200 further EQUILIBRIUM frames (the next 200 of the fixed pool permutation)."""
import numpy as np
from analyze_hops import ideal
rng = np.random.default_rng(20260921); pos0, Z0 = ideal(); out = []; src = []
def take(F, idx, tag):
    Z, L = F["numbers"], F["cell"]; fw = np.where(Z != 3)[0]; x = F["frames"][idx].astype(np.float64); x = x - (x[:, fw].mean(1) - pos0[fw].mean(0))[:, None]; out.append(x); src.extend([tag] * len(idx))
for T, f, n in ((1000, "../colab/li3ocl_teacher_1000K_censored.npz", 50), (900, "../colab/li3ocl_teacher_900K.npz", 50)):
    H = np.load(f"../data/li3ocl_hops_T{T}.npz"); w = np.abs(H["dq"]); take(np.load(f), H["frame"][rng.choice(len(w), n, replace=False, p=w / w.sum())], f"teacher_path_{T}")
for T in (1000, 900, 800):
    F = np.load(f"../colab/step7_results/hops_S6_{T}K.npz"); H = np.load(f"../data/li3ocl_hops_S6_{T}.npz"); w = np.abs(H["dq"]); n = (17, 17, 16)[(1000 - T) // 100]
    take(F, H["frame"][rng.choice(len(w), n, replace=False, p=w / w.sum())], f"S6_path_{T}")
    Z, L, S = F["numbers"], F["cell"], F["sites"]; li = np.where(Z == 3)[0]; fw = np.where(Z != 3)[0]; cand = rng.choice(len(F["frame_step"]), 4000, replace=False); X = F["frames"][cand].astype(np.float64)
    X = X - (X[:, fw].mean(1) - pos0[fw].mean(0))[:, None]; keep = []
    for k in range(0, len(X), 500):
        dd = X[k:k + 500][:, li][:, :, None, :] - S[None, None]; dd -= L * np.round(dd / L); a = np.linalg.norm(dd, axis=-1).argmin(-1)
        keep += [k + j for j, r in enumerate(a) if (np.bincount(r, minlength=len(S)) >= 2).any()]
    print(f"S6 {T} K: {len(keep)} of 4000 sampled frames are in the extra-defect state"); take(F, cand[rng.choice(keep, n, replace=False)], f"S6_defect_{T}")
X = np.concatenate(out); print(len(X), "frames:", {t: src.count(t) for t in sorted(set(src))})
np.savez_compressed("../colab/li3ocl_S6_two_sided_frames.npz", X=X.astype(np.float32), source=np.array(src), numbers=Z0, cell=np.array([3 * 3.926] * 3))
