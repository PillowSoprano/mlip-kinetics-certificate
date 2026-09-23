"""
The O-Cl framework of the teacher stays crystalline over the simulated time.

For every stored frame of the teacher runs used in the paper, the displacement of each framework atom (O, Cl) from its
ideal site is taken with the minimum-image convention and the rigid displacement of the framework is removed.  The
root-mean-square displacement per frame, and the largest single-atom displacement, are binned in time.
    python lattice_integrity.py
"""
import numpy as np
from analyze_hops import ideal

RUNS = {1000: "../colab/li3ocl_teacher_1000K_censored.npz", 900: "../colab/li3ocl_teacher_900K.npz",
        800: "../colab/li3ocl_teacher_800K_cluster.npz", 700: "../colab/li3ocl_teacher_700K_cluster.npz"}
NBIN = 12
pos0, num0 = ideal(); fw = np.where(num0 != 3)[0]; a = 3.926; d_OCl = a * np.sqrt(3) / 2

rows = []
for T, path in RUNS.items():
    d = np.load(path); X = d["frames"][:, fw].astype(np.float64); L = d["cell"]; t = d["frame_step"] * 2e-6    # ns
    u = X - pos0[fw]; u -= L * np.round(u / L); u -= u.mean(1, keepdims=True)
    r = np.linalg.norm(u, axis=-1); rms = np.sqrt((r**2).mean(1)); mx = r.max(1)
    edges = np.linspace(0, t.max() * (1 + 1e-9), NBIN + 1); k = np.digitize(t, edges) - 1
    for b in range(NBIN):
        s = k == b
        if s.sum(): rows.append((T, 0.5 * (edges[b] + edges[b + 1]), np.median(rms[s]), np.quantile(rms[s], 0.95), mx[s].max(), s.sum()))
    print(f"{T} K: {len(t)} frames over {t.max():.2f} ns per replica; framework rms displacement median {np.median(rms):.3f} A, "
          f"95% {np.quantile(rms, .95):.3f} A, first vs last bin {rows[-NBIN][2]:.3f} -> {rows[-1][2]:.3f} A; largest single-atom {mx.max():.2f} A "
          f"(Lindemann ratio of the median: {np.median(rms) / d_OCl:.3f})", flush=True)
np.savetxt("../figs/data/figS1_lattice.csv", rows, delimiter=",", comments="", header="T_K,t_ns,rms_median_A,rms_q95_A,max_single_A,n_frames")
