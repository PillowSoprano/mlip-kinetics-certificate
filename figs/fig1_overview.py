"""Fig. 1: problem (a) -> measurement (b) -> payoff (c).

(a) exact 1D double well, beta*B = 12, Gaussian bumps of width 0.15 scaled to force error 0.1 kT/length
    (same cases as doublewell_check.py / data/dw_E2_named_cases.csv).
(b) two-channel ring model of twochannel_blindspot.py (barriers 5 and 7 kT); a model error that raises the
    dominant channel by 1 kT and lowers the rare one by 1.5 kT, plus smooth background. Committor, leverage
    density, bounds and exact rate all from the resistor network of tube_bounds.py.
(c) alanine dipeptide students A1/A2, values from Table 2.
"""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from scipy.linalg import eigh_tridiagonal

import os, glob
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)
import tube_bounds as tb            # noqa: E402
import twochannel_blindspot as t    # noqa: E402

# ------------------------------------------------------------------ style
LM_DIRS = ["/usr/share/texmf/fonts/opentype/public/lm", os.path.expanduser("~/Library/Caches/Tectonic/bundles")]
for d in LM_DIRS:
    for f in glob.glob(os.path.join(d, "**", "lmroman*.otf"), recursive=True):
        fm.fontManager.addfont(f)
plt.rcParams.update({
    "font.family": "Latin Modern Roman", "mathtext.fontset": "cm", "font.size": 7,
    "axes.linewidth": 0.6, "axes.edgecolor": "#5d6d7e", "axes.labelcolor": "#2c3e50",
    "xtick.color": "#5d6d7e", "ytick.color": "#5d6d7e", "xtick.major.width": 0.6,
    "ytick.major.width": 0.6, "xtick.major.size": 2.2, "ytick.major.size": 2.2,
    "axes.spines.top": False, "axes.spines.right": False, "text.color": "#2c3e50",
})
REF, MODEL, GREY = "#2c3e50", "#c0392b", "#8a96a3"
CYAN, CYANL = "#2a9db5", "#d4eef4"
AMBER, AMBERL, AMBERD = "#e39b2d", "#f7d9a8", "#b8730f"
nu_cmap = LinearSegmentedColormap.from_list("nu", ["#ffffff", "#fbe6c4", "#f1b858", "#d9861a"])


def title(ax, s, x=-0.02, y=1.0):
    ax.text(x, y, s, transform=ax.transAxes, fontsize=8, fontweight="bold", va="bottom", ha="left",
            color=REF)


# ------------------------------------------------------------------ (a) data: exact 1D rates
BW, EPS, WD = 12.0, 0.1, 0.15
x = np.linspace(-1.8, 1.8, 1500); h = x[1] - x[0]
V1 = BW * (x**2 - 1) ** 2


def slow_rate(V):
    dV = np.diff(V)
    diag = np.zeros_like(V)
    diag[:-1] -= np.exp(-dV / 2) / h**2
    diag[1:] -= np.exp(dV / 2) / h**2
    w = eigh_tridiagonal(diag, np.full(len(V) - 1, 1 / h**2), select="i",
                         select_range=(len(V) - 2, len(V) - 1))[0]
    return -w[0]


pi1 = np.exp(-(V1 - V1.min())); pi1 /= pi1.sum()


def scaled_bump(c):
    g = np.exp(-(x - c) ** 2 / (2 * WD**2))
    dF = -np.diff(g) / h
    pe = np.sqrt(pi1[:-1] * pi1[1:])
    return g * EPS / np.sqrt((pe * dF**2).sum() / pe.sum())


k0 = slow_rate(V1)
bump_s, bump_w = scaled_bump(0.0), scaled_bump(1.0)
r_s, r_w = slow_rate(V1 + bump_s) / k0, slow_rate(V1 + bump_w) / k0
print(f"(a) saddle bump {bump_s.max():.2f} kT -> rate x{r_s:.4f} (1/{1/r_s:.0f}); "
      f"well bump {bump_w.max():.3f} kT -> rate x{r_w:.3f}")

# ------------------------------------------------------------------ (b) data: two-channel network
Vr = t.true_potential(); Vr -= Vr.min()
inA = ((Vr < 2) & (t.X < 0)).ravel(); inB = ((Vr < 2) & (t.X > 0)).ravel()
c0 = tb.network(Vr); C0, q0 = tb.capacity(c0, inA, inB)
wE = c0 * (q0[tb.EI] - q0[tb.EJ]) ** 2; wE /= wE.sum()            # leverage density on edges
rng = np.random.default_rng(0)
bg = sum(rng.normal() * np.cos(rng.uniform(1, 4) * t.X + rng.uniform(0, 6))
         * np.cos(rng.uniform(1, 4) * t.Y + rng.uniform(0, 6)) for _ in range(6))
dVr = 1.0 * t.gauss(0, -1, 0.3) - 1.5 * t.gauss(0, 1, 0.3) + 0.2 * bg
Vh = Vr + dVr
cH = tb.network(Vh); CH, qH = tb.capacity(cH, inA, inB); CH *= np.exp(-Vh.min())
wH = cH * (qH[tb.EI] - qH[tb.EJ]) ** 2; wH /= wH.sum()
up_E = t.Y.ravel()[tb.EI] > 0
print(f"(b) upper-channel share of reactive dissipation: reference {wE[up_E].sum():.0%}, model {wH[up_E].sum():.0%}")

dE = 0.5 * (dVr.ravel()[tb.EI] + dVr.ravel()[tb.EJ])
mu = np.exp(-Vr).ravel(); d = dVr.ravel()
aA = (mu[inA] * np.exp(-d[inA])).sum() / mu[inA].sum()
aB = (mu[inB] * np.exp(-d[inB])).sum() / mu[inB].sum()
pA = mu[inA].sum() / (mu[inA].sum() + mu[inB].sum()); pB = 1 - pA
lnG = np.log(pA / aB + pB / aA)
L = lnG - np.log((wE * np.exp(dE)).sum()); U = lnG + np.log((wE * np.exp(-dE)).sum())
basin_mean = pB * (mu[inA] * d[inA]).sum() / mu[inA].sum() + pA * (mu[inB] * d[inB]).sum() / mu[inB].sum()
nu_mean = (wE * dE).sum()
M = nu_mean - basin_mean
sig = np.sqrt((wE * (dE - nu_mean) ** 2).sum())


def rate(Vx, cap):
    m = np.exp(-Vx).ravel(); Z = m.sum()
    return cap / (Z * (m[inA].sum() / Z) * (m[inB].sum() / Z))


exact = np.log(rate(Vh, CH) / rate(Vr, C0))
print(f"(b) M={M:.3f}  sigma_nu={sig:.3f}  interval=[{L:.3f},{U:.3f}]  exact={exact:.3f}")

# basin sample: mu-weighted grid points inside A and B
bas_vals = np.r_[d[inA], d[inB]]
bas_w = np.r_[pB * mu[inA] / mu[inA].sum(), pA * mu[inB] / mu[inB].sum()]

# nodal leverage density for the map
nu_node = np.zeros(t.N * t.N)
np.add.at(nu_node, tb.EI, wE / 2); np.add.at(nu_node, tb.EJ, wE / 2)
nu_node = nu_node.reshape(t.N, t.N)

# ------------------------------------------------------------------ figure
from matplotlib.patches import ConnectionPatch

fig = plt.figure(figsize=(7.0, 3.3))
gs = fig.add_gridspec(1, 3, width_ratios=[2.1, 3.0, 1.75], wspace=0.26,
                      left=0.055, right=0.985, bottom=0.115, top=0.855)
TY = 0.955
SY = 0.905


def ptitle(xf, s, sub):
    fig.text(xf, TY, s, fontsize=8.2, fontweight="bold", color=REF, ha="left", va="center")
    fig.text(xf, SY, sub, fontsize=6.3, color=GREY, ha="left", va="center")


# ============================== (a)
ga = gs[0].subgridspec(2, 1, height_ratios=[1.6, 1.0], hspace=0.10)
axE = fig.add_subplot(ga[0]); axW = fig.add_subplot(ga[1], sharex=axE)

axE.plot(x, V1, color=REF, lw=1.2, zorder=3)
m = bump_s > 0.02
axE.fill_between(x[m], V1[m], (V1 + bump_s)[m], color=MODEL, alpha=0.13, lw=0, zorder=2)
axE.plot(x[m], (V1 + bump_s)[m], color=MODEL, lw=1.1, ls=(0, (3, 1.6)), zorder=4)
axE.text(0.0, BW + bump_s.max() + 0.5, f"+{bump_s.max():.0f}" + r"$\,k_BT$", color=MODEL,
         ha="center", va="bottom", fontsize=6.5)
axE.annotate("", xy=(1.0, 0.3), xytext=(1.0, 4.1),
             arrowprops=dict(arrowstyle="-|>", color=MODEL, lw=0.7, mutation_scale=6))
chip = dict(boxstyle="round,pad=0.28,rounding_size=0.3", lw=0.6)
axE.text(0.48, 17.0, rf"rate $\times\,1/{1/r_s:.0f}$", color="white", fontsize=7, ha="left", va="center",
         bbox=dict(chip, fc=MODEL, ec=MODEL))
axE.text(0.95, 5.6, f"+{bump_w.max():.2f}" + r"$\,k_BT$" + "\n" + rf"rate $\times\,{r_w:.2f}$", color=MODEL,
         fontsize=6.2, ha="center", va="center", linespacing=1.15,
         bbox=dict(chip, fc="white", ec=MODEL))
axE.text(-1.06, 4.2, r"$V$", color=REF, fontsize=7.5)
axE.text(-0.42, 15.2, r"$\hat V$", color=MODEL, fontsize=7.5)
axE.set_ylim(0, 21); axE.set_yticks([0, 12]); axE.set_yticklabels(["0", r"$B$"])
axE.set_ylabel(r"energy ($k_BT$)", labelpad=1)
plt.setp(axE.get_xticklabels(), visible=False); axE.tick_params(axis="x", length=0)

mu1 = np.exp(-V1); mu1 /= mu1.max()
inner = np.abs(x) <= 1
nu1 = np.where(inner, np.exp(V1 - BW), 0.0); nu1 /= nu1.max()
dFs = np.abs(np.gradient(bump_s, h)); dFs /= dFs.max()
axW.fill_between(x, mu1, color=CYANL, lw=0); axW.plot(x, mu1, color=CYAN, lw=0.9)
axW.fill_between(x, nu1, color=AMBERL, lw=0, alpha=0.9); axW.plot(x[inner], nu1[inner], color=AMBER, lw=0.9)
axW.plot(x, 0.6 * dFs, color=MODEL, lw=0.7, ls=(0, (2, 1.2)))
axW.text(-1.12, 1.06, r"$\mu$: force error", color=CYAN, ha="center", va="bottom", fontsize=6.5)
axW.text(1.0, 1.06, r"$\mu$", color=CYAN, ha="center", va="bottom", fontsize=6.5)
axW.text(0.0, 1.06, r"$\nu$: rate", color=AMBERD, ha="center", va="bottom", fontsize=6.5)
axW.annotate(r"$|\delta F|$", xy=(0.19, 0.5), xytext=(0.36, 0.70), fontsize=6.2,
             color=MODEL, ha="left", va="center", arrowprops=dict(arrowstyle="-", color=MODEL, lw=0.5))
axW.set_ylim(0, 1.32); axW.set_yticks([]); axW.set_xlim(-1.75, 1.75); axW.set_xticks([])
axW.set_ylabel("weight", labelpad=4); axW.set_xlabel("reaction coordinate", labelpad=2)

# ============================== (b)
gb = gs[1].subgridspec(2, 2, width_ratios=[1.08, 1.0], height_ratios=[1.6, 0.55], wspace=0.22, hspace=0.75)
axM = fig.add_subplot(gb[:, 0]); axH = fig.add_subplot(gb[0, 1]); axI = fig.add_subplot(gb[1, 1])

ext = [t.xs[0], t.xs[-1], t.xs[0], t.xs[-1]]
axM.contourf(t.xs, t.xs, Vr.T, levels=[-1, 9], colors=["#eef1f4"])
axM.contour(t.xs, t.xs, Vr.T, levels=[9], colors=["#c3cad2"], linewidths=0.5)
nus = np.where(Vr > 10, 0.0, np.sqrt(nu_node / nu_node.max()))
axM.contourf(t.xs, t.xs, nus.T, levels=np.linspace(0.07, 1.0, 16), cmap=nu_cmap, vmin=0, vmax=1, zorder=1.5)
Qm = np.ma.masked_where(Vr > 9, q0.reshape(t.N, t.N))
axM.contour(t.xs, t.xs, Qm.T, levels=[0.2, 0.35, 0.5, 0.65, 0.8], colors="white", linewidths=0.55, zorder=2)
for side in (-1, 1):
    Vs = np.where(side * t.X > 0, Vr, 99.0)
    axM.contourf(t.xs, t.xs, Vs.T, levels=[-1, 2], colors=[CYANL], zorder=3)
    axM.contour(t.xs, t.xs, Vs.T, levels=[2], colors=[CYAN], linewidths=0.8, zorder=3)
axM.text(-1.0, 0.0, r"$A$", color=CYAN, ha="center", va="center", fontsize=9.5, zorder=4)
axM.text(1.0, 0.0, r"$B$", color=CYAN, ha="center", va="center", fontsize=9.5, zorder=4)

prng = np.random.default_rng(7)


def arc(th0, th1, r0, amp, n=160):
    th = np.linspace(th0, th1, n)
    jit = sum(prng.normal() * np.sin((k + 1) * np.pi * (th - th0) / (th1 - th0)) / (k + 1) for k in range(5))
    r = r0 + amp * jit
    return r * np.cos(th), r * np.sin(th)


for r0 in (0.95, 1.05, 1.0):
    px, py = arc(np.pi * 1.12, np.pi * 1.88, r0, 0.02)
    axM.plot(px, py, color=REF, lw=0.6, zorder=5, alpha=0.85)
px, py = arc(np.pi * 1.12, np.pi * 1.88, 1.0, 0.0, n=7)
axM.plot(px[1:-1], py[1:-1], "o", ms=2.5, color=REF, zorder=6)
for r0 in (0.97, 1.03):
    mx, my = arc(np.pi * 0.88, np.pi * 0.12, r0, 0.025)
    axM.plot(mx, my, color=MODEL, lw=0.85, ls=(0, (3, 1.5)), zorder=5)
mx, my = arc(np.pi * 0.88, np.pi * 0.12, 1.0, 0.0, n=7)
axM.plot(mx[1:-1], my[1:-1], "o", ms=2.5, color=MODEL, zorder=6)

axM.text(0, -1.40, "reference paths", color=REF, ha="center", va="top", fontsize=6.5)
axM.text(0, -1.62, rf"carry ${wE[~up_E].sum()*100:.0f}\%$ of the current", color=GREY, ha="center",
         va="top", fontsize=5.8)
axM.text(0, 1.62, "model paths", color=MODEL, ha="center", va="bottom", fontsize=6.5)
axM.text(0, 1.40, rf"route at ${wE[up_E].sum()*100:.0f}\%\ \to\ {wH[up_E].sum()*100:.0f}\%$",
         color=MODEL, ha="center", va="bottom", fontsize=5.8)
axM.text(0.0, 0.0, r"$\nu$", color=AMBERD, ha="center", va="center", fontsize=8.5)
axM.annotate("", xy=(0.0, -0.80), xytext=(0.0, -0.25),
             arrowprops=dict(arrowstyle="-|>", color=AMBERD, lw=0.5, mutation_scale=5))
axM.set_xlim(-1.62, 1.62); axM.set_ylim(-1.9, 1.95); axM.set_aspect("equal"); axM.axis("off")

con = ConnectionPatch(xyA=(1.15, -1.05), coordsA=axM.transData, xyB=(-0.06, 0.18), coordsB=axH.transAxes,
                      arrowstyle="-|>", color=GREY, lw=0.6, mutation_scale=6,
                      connectionstyle="arc3,rad=0.25")
fig.add_artist(con)
fig.text(0.0, 0.0, "")

bins = np.linspace(-2.0, 1.6, 31)
axH.hist(bas_vals, bins=bins, weights=bas_w, color=CYANL, edgecolor=CYAN, lw=0.5, histtype="stepfilled")
axH.hist(dE, bins=bins, weights=wE, color=AMBERL, edgecolor=AMBER, lw=0.6, histtype="stepfilled", alpha=0.85)
yt = axH.get_ylim()[1]; ymax = yt * 1.32
axH.plot([basin_mean] * 2, [0, ymax * 0.84], color=CYAN, lw=0.7, ls=(0, (2, 1)))
axH.plot([nu_mean] * 2, [0, ymax * 0.84], color=AMBERD, lw=0.7, ls=(0, (2, 1)))
yM = ymax * 0.84
axH.annotate("", xy=(nu_mean, yM), xytext=(basin_mean, yM),
             arrowprops=dict(arrowstyle="-|>", color=REF, lw=0.6, mutation_scale=5, shrinkA=0, shrinkB=0))
axH.text((nu_mean + basin_mean) / 2, yM * 1.02, r"$M$", ha="center", va="bottom", fontsize=7.5)
axH.text(-1.95, ymax * 0.80, "basins", color=CYAN, fontsize=6, ha="left")
axH.text(-1.95, ymax * 0.66, r"paths, $\nu$", color=AMBERD, fontsize=6, ha="left")
axH.text(-1.25, ymax * 0.17, "opened\nroute", color=MODEL, fontsize=5.5, ha="center", va="bottom")
axH.text(1.05, yt * 0.36, "raised\nbarrier", color=MODEL, fontsize=5.5, ha="center", va="bottom")
ys = -ymax * 0.10
axH.annotate("", xy=(nu_mean + sig, ys), xytext=(nu_mean - sig, ys), annotation_clip=False,
             arrowprops=dict(arrowstyle="<->", color=AMBERD, lw=0.55, mutation_scale=4, shrinkA=0, shrinkB=0))
axH.text(nu_mean - sig - 0.06, ys, r"$\sigma_\nu$", color=AMBERD, fontsize=6.5, ha="right", va="center")
axH.set_ylim(0, ymax); axH.set_yticks([]); axH.set_xlim(-2.0, 1.6)
axH.spines["left"].set_visible(False)
axH.spines["bottom"].set_position(("axes", -0.20))
axH.set_xticks([-1, 0, 1])
axH.set_xlabel(r"$\beta\,\delta V$", labelpad=1)

axI.plot([L, U], [0, 0], color=AMBER, lw=4.5, solid_capstyle="butt", alpha=0.5)
axI.plot([L, L], [-0.25, 0.25], color=AMBERD, lw=0.8); axI.plot([U, U], [-0.25, 0.25], color=AMBERD, lw=0.8)
axI.plot(-M, 0, "D", ms=4, mfc="white", mec=REF, mew=0.8, zorder=5)
axI.plot(exact, 0, "o", ms=3.4, color=MODEL, zorder=6)
axI.text(-M, 0.42, r"$-M$", ha="center", va="bottom", fontsize=6.5)
axI.text(exact, 0.42, "exact", ha="center", va="bottom", fontsize=6, color=MODEL)
axI.text(U + 0.04, 0, "bounds", ha="left", va="center", fontsize=6, color=AMBERD)
axI.set_xlim(-0.7, 0.32); axI.set_ylim(-0.5, 1.1); axI.set_yticks([])
axI.spines["left"].set_visible(False)
axI.set_xticks([-0.5, 0]); axI.set_xlabel(r"$\ln(\hat k/k)$", labelpad=1)

# ============================== (c)
gc = gs[2].subgridspec(3, 1, height_ratios=[0.55, 1.35, 0.62], hspace=0.95)
axF = fig.add_subplot(gc[0]); axR = fig.add_subplot(gc[1])
axCard = fig.add_subplot(gc[2]); axCard.axis("off")
names = ["A2", "A1"]; yy = [0, 1]
rmse = {"A1": 87.2, "A2": 86.9}
axF.barh(yy, [rmse[n] for n in names], height=0.55, color="#d5dbe1", edgecolor=GREY, lw=0.5)
for y_, n in zip(yy, names):
    axF.text(rmse[n] + 3, y_, f"{rmse[n]:.1f}", va="center", fontsize=6.3)
axF.set_yticks(yy); axF.set_yticklabels(names); axF.set_xlim(0, 112); axF.set_xticks([0, 50, 100])
axF.set_xlabel("force RMSE (meV/Å)", labelpad=1); axF.tick_params(axis="y", length=0)

pred = {"A1": (0.95, 0.10), "A2": (1.30, 0.08)}
refm = {"A1": (0.82, 0.04), "A2": (1.14, 0.04)}
allm = {"A1": 2.04, "A2": 2.96}
for y_, n in zip(yy, names):
    axR.annotate("", xy=(allm[n] - 0.07, y_), xytext=(refm[n][0] + 0.1, y_ - 0.17),
                 arrowprops=dict(arrowstyle="-|>", color=MODEL, lw=0.55, alpha=0.7, mutation_scale=5,
                                 ls=(0, (2, 1.2)), connectionstyle="arc3,rad=-0.15"))
    axR.plot(allm[n], y_, "o", ms=4.2, mfc="white", mec=MODEL, mew=0.8)
    axR.errorbar(pred[n][0], y_ + 0.17, xerr=pred[n][1], fmt="D", ms=3.4, mfc="white", mec=REF, mew=0.8,
                 ecolor=REF, elinewidth=0.6, capsize=1.3)
    axR.errorbar(refm[n][0], y_ - 0.17, xerr=refm[n][1], fmt="o", ms=3.2, color=REF, elinewidth=0.6,
                 capsize=1.3)
axR.text(pred["A1"][0], 1.40, "predicted", fontsize=5.6, ha="center", va="bottom")
axR.text(refm["A1"][0], 0.60, "measured", fontsize=5.6, ha="center", va="top")
axR.text(allm["A1"], 1.28, "all\ntransitions", fontsize=5.6, ha="center", va="bottom", color=MODEL)
axR.text(1.78, -0.42, "extra = route the\nmodel invented", fontsize=5.5, ha="left", va="center", color=MODEL)
axR.set_yticks(yy); axR.set_yticklabels(names); axR.set_ylim(-0.85, 1.85); axR.set_xlim(0, 3.3)
axR.set_xticks([0, 1, 2, 3]); axR.set_xlabel(r"$\ln(\hat k/k)$", labelpad=1); axR.tick_params(axis="y", length=0)

card = FancyBboxPatch((0.0, 0.0), 1.0, 1.0, boxstyle="round,pad=0.0,rounding_size=0.06",
                      transform=axCard.transAxes, fc="#f6f7f9", ec=GREY, lw=0.5, clip_on=False)
axCard.add_patch(card)
axCard.text(0.06, 0.76, "report for kinetics", fontsize=6.3, fontweight="bold", va="center")
axCard.text(0.06, 0.47, r"force RMSE $+\ M,\ \sigma_\nu$", fontsize=6.8, va="center")
axCard.text(0.06, 0.19, r"$T$, event, path ensemble", fontsize=6.0, color=GREY, va="center")

# panel titles aligned on one baseline
pa, pb, pc = axE.get_position(), axM.get_position(), axF.get_position()
ptitle(0.012, "(a) The problem", "same force error, two places")
ptitle(pb.x0 - 0.005, "(b) The measurement", "weight energy errors by the reactive current")
ptitle(pc.x0 - 0.045, "(c) The payoff", "two seeds, one force error")

fig.savefig(os.path.join(HERE, "fig1_overview.pdf")); fig.savefig(os.path.join(HERE, "fig1_overview.png"), dpi=300)
print("saved")
