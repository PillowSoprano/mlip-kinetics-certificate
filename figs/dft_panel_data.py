"""
Values for Figure 5d: teacher - DFT energy error of the whole Li3OCl cell on basin and hop-path frames.

    python dft_panel_data.py        (run from figs/; reads ../data/li3ocl_teacher_vs_dft.npz)

beta*(E_teacher - E_DFT) at 1000 K, relative to the mean over basin frames. Writes a kernel density estimate per
group, every frame with a deterministic vertical jitter, and the two means with bootstrap standard errors.
"""
import numpy as np

KB = 8.617333e-5; beta = 1 / (KB * 1000.0)
d = np.load("../data/li3ocl_teacher_vs_dft.npz"); dV = d["E_teacher"] - d["E_dft"]; kind = d["kind"]
E, P = kind == "eq", kind == "path"; x = beta * (dV - dV[E].mean())
grid = np.linspace(-8, 9.5, 351)

def kde(v):
    h = 1.06 * v.std() * len(v) ** -0.2
    return np.exp(-0.5 * ((grid[:, None] - v[None]) / h) ** 2).sum(1) / (len(v) * h * np.sqrt(2 * np.pi))

np.savetxt("data/fig_dft_kde.csv", np.c_[grid, kde(x[E]), kde(x[P])], delimiter=",", header="x,eq,path", comments="", fmt="%.5f")
rng = np.random.default_rng(1)
for name, sel in (("eq", E), ("path", P)):
    np.savetxt(f"data/fig_dft_points_{name}.csv", np.c_[x[sel], rng.uniform(-1, 1, sel.sum())], delimiter=",", header="x,y", comments="", fmt="%.4f")
B = np.array([[x[E][rng.integers(0, E.sum(), E.sum())].mean(), x[P][rng.integers(0, P.sum(), P.sum())].mean()] for _ in range(2000)])
m = [x[E].mean(), x[P].mean()]; se = B.std(0)
np.savetxt("data/fig_dft_means.csv", np.c_[m, se], delimiter=",", header="mean,se", comments="", fmt="%.4f")
diff = np.subtract(*B.T[::-1]); print(f"basin mean {m[0]:+.3f}({se[0]:.3f}), path mean {m[1]:+.3f}({se[1]:.3f}); "
                                     f"M = {m[1]-m[0]:.3f} +- {diff.std():.3f} kT = {1e3*(m[1]-m[0])/beta:.0f} +- {1e3*diff.std()/beta:.0f} meV; "
                                     f"sd {x[E].std():.2f} / {x[P].std():.2f} kT")
