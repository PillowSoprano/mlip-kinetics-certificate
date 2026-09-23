"""Robustness scans for twochannel_blindspot.py: (a) how hard must the channel be blocked before
leverage goes blind, (b) seed-robustness of the final plateau, (c) what the residual error is made of."""
import numpy as np, twochannel_blindspot as t

V = t.true_potential(); lam, phi, pi, _ = t.slow_mode(V); m = t.lower_channel_mask()
gx, gy = np.gradient(phi, t.h); g2 = gx**2 + gy**2

print("(a) blocking height scan  [upper-channel model barrier = 8 kT]")
rows = []
for block in (3, 6, 9, 12, 15, 18):
    t.BLOCK = float(block); t.rng = np.random.default_rng(1)
    dV0 = t.initial_error(); lam_hat, sc, _, _ = t.scores(V, dV0)
    first = {}
    for s in ("unc_md", "lev_md", "levE_md", "unc_free", "lev_free"):
        _, p = t.run_al(V, dV0, lam, s)
        hit = [r + 1 for r, (px, py) in enumerate(p) if abs(px) < 0.6 and py < -0.5]
        first[s] = hit[0] if hit else 0
    fr = {k: sc[k][m].sum() / sc[k].sum() for k in ("unc_md", "lev_md", "lev_free")}
    rows.append((block, lam_hat / lam, fr["unc_md"], fr["lev_md"], fr["lev_free"], *first.values()))
    print(f"  block={block:2d}kT rate_hat/rate={lam_hat/lam:.3f}  lower-channel mass: unc_md={fr['unc_md']:.1e} lev_md={fr['lev_md']:.1e} lev_free={fr['lev_free']:.2f}"
          f" | first hit round (0=never in {t.ROUNDS}): " + " ".join(f"{k}={v}" for k, v in first.items()))
np.savetxt("data/twochannel_blockscan.csv", rows, delimiter=",", comments="",
           header="block_kT,rate_ratio,mass_unc_md,mass_lev_md,mass_lev_free,first_unc_md,first_lev_md,first_levE_md,first_unc_free,first_lev_free")

print("\n(b) seed robustness at block=6: |ln rate error| after 3 / 6 / 14 queries, median over 8 background seeds")
t.BLOCK = 6.0
res = {s: [] for s in ("unc_md", "lev_md", "levE_md", "unc_free", "lev_free")}
resid = []
for seed in range(8):
    t.rng = np.random.default_rng(100 + seed); dV0 = t.initial_error()
    for s in res:
        c, _ = t.run_al(V, dV0, lam, s); res[s].append(np.abs(c))
for s, c in res.items():
    c = np.array(c); md = np.median(c, axis=0)
    print(f"  {s:9s} start={md[0]:.2f}  q3={md[3]:.2f}  q6={md[6]:.2f}  q14={md[14]:.2f}   (min..max at q14: {c[:,14].min():.2f}..{c[:,14].max():.2f})")
np.savez("data/twochannel_seedscan.npz", **{k: np.array(v) for k, v in res.items()})

print("\n(c) what is the residual made of? background-only error (no bumps), exact vs first-order energy formula")
for seed in range(4):
    t.rng = np.random.default_rng(100 + seed)
    save = (t.BLOCK, t.UPPER_ERR, t.DECOY); t.BLOCK = t.UPPER_ERR = t.DECOY = 0.0
    bg = t.initial_error(); t.BLOCK, t.UPPER_ERR, t.DECOY = save
    lam_bg = t.slow_mode(V + bg)[0]
    first_order = (bg * (g2 - abs(lam) * phi**2) * pi).sum() / lam      # d lambda / lambda from energy-error formula
    fx, fy = np.gradient(-bg, t.h)
    print(f"  seed {seed}: force RMSE of background={np.sqrt((pi*(fx**2+fy**2)).sum()):.3f}  exact ln(rate ratio)={np.log(lam_bg/lam):+.3f}  energy-formula prediction={first_order:+.3f}")
