"""
2D two-channel blind-spot experiment (overdamped, beta = 1).

Ring-shaped potential, wells at (+-1, 0), two channels:
    upper saddle (0,+1) barrier B_UP, lower saddle (0,-1) barrier B_DN  (true PES: lower is the fast one)
"MLIP" = true PES + dV, where dV contains
    * a large error that BLOCKS the lower channel            (the blind-spot candidate)
    * a moderate error on the upper saddle                   (visible, dynamically relevant)
    * large decoy errors in dynamically irrelevant places    (visible to uncertainty, irrelevant)
    * a small smooth background error
A "DFT query" at point q repairs dV in a Gaussian neighbourhood of q.
Acquisition strategies differ in (score) x (candidate pool):
    pool "md"   : candidates weighted by the MODEL's equilibrium density mu_hat (what MLIP-MD visits)
    pool "free" : any grid point with model energy below E_CUT (e.g. enhanced / uniform sampling)
    score "unc" : |dF|^2            (oracle-calibrated uncertainty = best case for uncertainty sampling)
    score "lev" : (dF . grad phi_hat)^2    (uncertainty x dynamical leverage, MODEL's own slow mode)
    score "levE": |dV| * | |grad phi_hat|^2 - beta |lambda_hat| phi_hat^2 |   (energy-form leverage, sees basin offsets)
All raw results are written to data/ as .npz and .csv.
"""
import os
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

N, XMAX = 121, 1.8
KR, B_UP, B_DN = 30.0, 7.0, 5.0
BLOCK, UPPER_ERR, DECOY = 6.0, 1.0, 6.0
RHO, E_CUT, ROUNDS = 0.3, 16.0, 14
rng = np.random.default_rng(1)

xs = np.linspace(-XMAX, XMAX, N)
h = xs[1] - xs[0]
X, Y = np.meshgrid(xs, xs, indexing="ij")
R2 = X**2 + Y**2
R = np.sqrt(R2)


def gauss(cx, cy, w):
    return np.exp(-((X - cx) ** 2 + (Y - cy) ** 2) / (2 * w**2))


def true_potential():
    s2 = Y**2 / (R2 + 0.01)
    s = Y / np.sqrt(R2 + 0.01)
    return KR * (R - 1) ** 2 + s2 * (0.5 * (B_UP + B_DN) + 0.5 * (B_UP - B_DN) * s)


def initial_error():
    dV = BLOCK * gauss(0, -1, 0.3)                       # blocks the lower channel
    dV += UPPER_ERR * gauss(0, 1, 0.3)                   # moderate error on the upper saddle
    for cx, cy in ((1.45, 0.0), (-1.45, 0.0), (0.0, 0.0)):   # decoys: outside the ring / centre hill
        dV += DECOY * gauss(cx, cy, 0.25)
    bg = sum(rng.normal() * np.cos(rng.uniform(1, 4) * X + rng.uniform(0, 6))
             * np.cos(rng.uniform(1, 4) * Y + rng.uniform(0, 6)) for _ in range(6))
    return dV + 0.15 * bg


def slow_mode(V):
    """lambda_1 (<0), phi_1, stationary pi for the reversible jump discretisation of the generator."""
    diag = np.zeros_like(V)
    dx = np.diff(V, axis=0); dy = np.diff(V, axis=1)
    diag[:-1, :] -= np.exp(-dx / 2); diag[1:, :] -= np.exp(dx / 2)
    diag[:, :-1] -= np.exp(-dy / 2); diag[:, 1:] -= np.exp(dy / 2)
    idx = np.arange(N * N).reshape(N, N)
    rows = np.concatenate([idx[:-1, :].ravel(), idx[:, :-1].ravel()])
    cols = np.concatenate([idx[1:, :].ravel(), idx[:, 1:].ravel()])
    off = sp.coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(N * N, N * N))
    A = -(sp.diags(diag.ravel()) + off + off.T) / h**2          # PSD
    w, U = eigsh(A.tocsc(), k=3, sigma=-1e-3, which="LM")
    o = np.argsort(w)
    pi = np.exp(-(V - V.min())); pi /= pi.sum()
    phi = (U[:, o[1]] / np.sqrt(pi.ravel())).reshape(N, N)
    return -w[o[1]], phi, pi, -w[o[2]]


def scores(V_true, dV):
    lam_hat, phi_hat, pi_hat, _ = slow_mode(V_true + dV)
    gx, gy = np.gradient(phi_hat, h)
    fx, fy = np.gradient(-dV, h)
    unc = fx**2 + fy**2
    lev = (fx * gx + fy * gy) ** 2
    pool_free = (V_true + dV) - (V_true + dV).min() < E_CUT
    # energy-form leverage: first-order d(lambda) = int dV (|grad phi|^2 - beta|lambda| phi^2) dmu, so basin offsets count too
    dVc = dV - (pi_hat * dV).sum()
    levE = pi_hat * np.abs(dVc) * np.abs(gx**2 + gy**2 - abs(lam_hat) * phi_hat**2)
    return lam_hat, {
        "levE_md": levE,
        "unc_md": pi_hat * unc, "lev_md": pi_hat * lev,
        "unc_free": np.where(pool_free, unc, 0), "lev_free": np.where(pool_free, lev, 0),
    }, pi_hat, phi_hat


def lower_channel_mask():
    return (np.abs(X) < 0.6) & (Y < -0.5)


def run_al(V_true, dV0, lam_true, strategy):
    dV = dV0.copy()
    hist, picks = [], []
    for r in range(ROUNDS + 1):
        lam_hat, sc, pi_hat, _ = scores(V_true, dV)
        hist.append(np.log(lam_hat / lam_true))
        if r == ROUNDS:
            break
        if strategy == "random_md":
            k = rng.choice(N * N, p=pi_hat.ravel())
        else:
            k = np.argmax(sc[strategy])
        i, j = np.unravel_index(k, (N, N))
        picks.append((xs[i], xs[j]))
        dV = dV * (1 - gauss(xs[i], xs[j], RHO))               # the DFT query repairs this neighbourhood
    return np.array(hist), np.array(picks)


if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    V = true_potential()
    dV0 = initial_error()
    lam, phi, pi, lam2 = slow_mode(V)
    lam_hat, sc, pi_hat, phi_hat = scores(V, dV0)
    print(f"true  lambda_1 = {lam:.4e}   (next eigenvalue {lam2:.3e}, gap ok)")
    print(f"model lambda_1 = {lam_hat:.4e}   rate_hat/rate = {lam_hat/lam:.4f}")

    # what would a TRUE-dynamics leverage look like (not available in practice)
    gx, gy = np.gradient(phi, h); fx, fy = np.gradient(-dV0, h)
    sc["lev_true_md (oracle)"] = pi * (fx * gx + fy * gy) ** 2
    m = lower_channel_mask()
    print("\nfraction of acquisition mass inside the blocked lower channel:")
    frac = {}
    for k, s in sc.items():
        frac[k] = s[m].sum() / s.sum()
        print(f"   {k:22s} {frac[k]:.3e}")
    g2 = gx**2 + gy**2; g2h = np.gradient(phi_hat, h); g2h = g2h[0]**2 + g2h[1]**2
    print(f"\n|grad phi|^2 at lower saddle:  true {g2[N//2, np.argmin(abs(xs+1))]:.3f}   model {g2h[N//2, np.argmin(abs(xs+1))]:.3f}")
    print(f"mu at lower saddle:            true {pi[N//2, np.argmin(abs(xs+1))]:.3e}   model {pi_hat[N//2, np.argmin(abs(xs+1))]:.3e}")

    print(f"\nactive learning, {ROUNDS} queries, |ln(rate_hat/rate)| per round:")
    strategies = ["random_md", "unc_md", "lev_md", "levE_md", "unc_free", "lev_free"]
    curves, picks = {}, {}
    for s in strategies:
        if s == "random_md":
            reps = [run_al(V, dV0, lam, s)[0] for _ in range(20)]
            curves[s] = np.median(np.abs(reps), axis=0); picks[s] = np.zeros((0, 2))
        else:
            c, p = run_al(V, dV0, lam, s)
            curves[s] = np.abs(c); picks[s] = p
        print(f"   {s:10s} " + " ".join(f"{v:5.2f}" for v in curves[s]))
    for s in strategies[1:]:
        hit = [r for r, (px, py) in enumerate(picks[s]) if abs(px) < 0.6 and py < -0.5]
        print(f"   {s:10s} first query in lower channel: round {hit[0]+1 if hit else 'never'}")

    # ---------------- raw data
    np.savez_compressed("data/twochannel_fields.npz", x=xs, V=V, dV0=dV0, phi=phi, phi_hat=phi_hat,
                        pi=pi, pi_hat=pi_hat, **{"score_" + k.split(" ")[0]: v for k, v in sc.items()})
    np.savetxt("data/twochannel_al_curves.csv",
               np.column_stack([np.arange(ROUNDS + 1)] + [curves[s] for s in strategies]),
               delimiter=",", header="round," + ",".join(strategies), comments="")
    for s in strategies[1:]:
        np.savetxt(f"data/twochannel_picks_{s}.csv", picks[s], delimiter=",", header="x,y", comments="")
    with open("data/twochannel_summary.csv", "w") as f:
        f.write("quantity,value\n")
        f.write(f"lambda_true,{lam}\nlambda_model,{lam_hat}\n")
        for k, v in frac.items():
            f.write(f"lower_channel_mass_{k.split(' ')[0]},{v}\n")

    # ---------------- figure
    fig, ax = plt.subplots(2, 3, figsize=(14, 8.5))
    ext = [-XMAX, XMAX, -XMAX, XMAX]
    def show(a, F, title, **kw):
        im = a.imshow(F.T, origin="lower", extent=ext, **kw); a.set_title(title, fontsize=10)
        a.contour(X, Y, V, levels=[2, 5, 7, 10], colors="w", linewidths=0.4); plt.colorbar(im, ax=a, shrink=0.8)
    show(ax[0, 0], np.minimum(V, 20), "true PES (kT)")
    show(ax[0, 1], dV0, "model error dV (kT)", cmap="RdBu_r", vmin=-6, vmax=6)
    show(ax[0, 2], np.log10(sc["lev_true_md (oracle)"] + 1e-30), "log10 leverage, TRUE dynamics (oracle)", vmin=-8)
    show(ax[1, 0], np.log10(sc["lev_md"] + 1e-30), "log10 leverage, model's own dynamics (MD pool)", vmin=-8)
    show(ax[1, 1], np.log10(sc["lev_free"] + 1e-30), "log10 leverage, model phi_hat, free pool", vmin=-3)
    a = ax[1, 2]
    for s in strategies:
        a.plot(curves[s], "o-", ms=3, label=s)
    a.set_yscale("log"); a.set_xlabel("DFT queries"); a.set_ylabel("|ln(rate_hat / rate)|"); a.legend(fontsize=8)
    a.set_title("active learning: kinetic error vs budget", fontsize=10)
    plt.tight_layout(); plt.savefig("twochannel_blindspot.png", dpi=130)
    print("\nsaved twochannel_blindspot.png and data/")
