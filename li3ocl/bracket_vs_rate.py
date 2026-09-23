"""
The two-sided bracket against the measured rate, both on the channel the certificate refers to.

nu is proportional to exp(+beta F) along the hop coordinate.  Three estimates of F are compared, because it is
the least certain input:
  path   the histogram of stored reactive frames, which is what step 2 of the protocol uses.  It is the density of
         reactive paths, p(s|TP) proportional to exp(-beta F) q(1-q), and the factor q(1-q) fills the barrier in.
  emp    minus the log of the equilibrium density of s, binned, folded about s = 1/2, with no assumed shape.
  cos    a cosine barrier fitted to the same equilibrium density by Poisson likelihood.
For a symmetric pair of sites the basin factor is G = 1 / <exp(-beta dV)>_mu, so
    lower = ln G - ln<e^{+b dV}>_nu  <=  ln(k_hat/k)  <=  ln G + ln<e^{-b dV}>_nu = upper.
    python bracket_vs_rate.py
"""
import numpy as np
from scipy.optimize import minimize
SA, SB = 0.25, 0.75
R = {"teacher": (527, 4.396), "S1": (215, 1.450), "S2": (402, 2.803), "S3": (223, 1.634), "S4": (383, 2.794),
     "S5": (196, 1.138), "S6": (156, 0.530), "S6b": (412, 2.560), "S6c": (403, 1.556)}
SIX = ["S1", "S2", "S3", "S4", "S5", "S6"]
lr = lambda n: (np.log((R[n][0] / R[n][1]) / (R["teacher"][0] / R["teacher"][1])),
                np.sqrt(1 / R[n][0] + 1 / R["teacher"][0]))

D = np.load("../data/li3ocl_dV_profile_1000K.npz"); s, edges = D["s"], D["edges"]
NB = len(edges) - 1; m2 = .5 * (edges[1:] + edges[:-1]); occ = D["bin_count"].astype(float)
ib = np.clip(np.digitize(s, edges) - 1, 0, NB - 1); nsamp = np.bincount(ib, minlength=NB).astype(float)

sv = np.load("../data/li3ocl_s_equilibrium_1000K.npy")
ce = np.linspace(-0.25, 0.55, 33); cm = .5 * (ce[1:] + ce[:-1]); cnt = np.histogram(sv, bins=ce)[0]
def bFcos(x, B, s0): return .5 * B * (1 - np.cos(2 * np.pi * (x - s0) / (1 - 2 * s0)))
def nll(p):
    B, s0, lnA = p
    if not (0 < B < 60 and -.2 < s0 < .2): return 1e9
    f = bFcos(cm, B, s0); return (np.exp(lnA - f) - cnt * (lnA - f)).sum()
Bc, s0c, _ = minimize(nll, [7., .04, 7.5], method="Nelder-Mead", options=dict(xatol=1e-8, fatol=1e-8, maxiter=60000)).x

NC = 8; ec = np.linspace(SA, 0.5, NC // 2 + 1)                       # coarse bins on the rising half
half = np.minimum(m2, 1 - m2)                                         # fold about s = 1/2
def F_emp():
    x = np.minimum(sv, 1 - sv); h = np.histogram(x, bins=ec)[0].astype(float)
    h = np.maximum(h, 0.5); f = -np.log(h / h.max())
    j = np.clip(np.digitize(half, ec) - 1, 0, len(ec) - 2)
    return f[j] - f[j].min()
FS = {"path (protocol)": np.interp(m2, .5 * (np.load("../data/li3ocl_hops_T1000.npz")["edges"][1:] +
                                               np.load("../data/li3ocl_hops_T1000.npz")["edges"][:-1]),
                                     np.load("../data/li3ocl_hops_T1000.npz")["F"]),
      "equilibrium, binned": F_emp(),
      "equilibrium, cosine": bFcos(m2, Bc, s0c)}
for k in FS: FS[k] = FS[k] - FS[k].min()

def bracket(n, w):
    d, g = D["path_" + n], D["eq_" + n]
    lnG = g.mean() - np.log(np.exp(-(g - g.mean())).mean())
    return (lnG - np.log((w * np.exp(d - d.mean())).sum()) - d.mean(),
            lnG + np.log((w * np.exp(-(d - d.mean()))).sum()) - d.mean())
def wts(F):
    nu = np.exp(F) * occ; nu /= nu.sum()
    w = np.where(nsamp[ib] > 0, nu[ib] / np.maximum(nsamp[ib], 1), 0); return w / w.sum()

y, sy = map(np.array, zip(*[lr(n) for n in SIX]))
print(f"cosine fit: barrier {Bc:.2f} kT, minima at s = {s0c:+.3f} and {1-s0c:.3f}")
print(f"rise over the reactive window: path {FS['path (protocol)'].max():.2f}, "
      f"binned {FS['equilibrium, binned'].max():.2f}, cosine {FS['equilibrium, cosine'].max():.2f} kT\n")
best = None
for lab, F in FS.items():
    w = wts(F); up = np.array([bracket(n, w)[1] for n in SIX])
    sl = (up * y).sum() / (up * up).sum(); se = np.sqrt((up ** 2 * sy ** 2).sum()) / (up * up).sum()
    print(f"{lab:22s}: slope {sl:.2f} +- {se:.2f}, correlation {np.corrcoef(up,y)[0,1]:.3f}, "
          f"mean |difference| {np.abs(up-y).mean():.3f}")
    if lab == "equilibrium, binned": best = w
print()
print("student  bracket (equilibrium nu, binned)   measured        (measured-upper)/se")
rows = []
for n in SIX + ["S6b", "S6c"]:
    lo, up = bracket(n, best); mv, se = lr(n)
    sd = np.sqrt((best * (D["path_" + n] - (best * D["path_" + n]).sum()) ** 2).sum())
    print(f"  {n:4s}   [{lo:+.3f}, {up:+.3f}]                 {mv:+.2f} +- {se:.2f}      {(mv-up)/se:+5.1f}")
    rows.append((n, lo, up, sd, mv, se))
np.savetxt("../data/li3ocl_bracket_vs_rate.csv", np.array(rows, dtype=object), delimiter=",", fmt="%s",
           header="student,lower,upper,sd_nu,measured,se", comments="")
print("\nsensitivity: the slope as the barrier of the cosine form is varied")
for B in (0.38, 2, 3, 4.58, 6.32, 9, 12):
    w = wts(bFcos(m2, B, s0c) - bFcos(m2, B, s0c).min())
    up = np.array([bracket(n, w)[1] for n in SIX])
    print(f"   B = {B:5.2f} kT -> slope {(up*y).sum()/(up*up).sum():.2f}")
