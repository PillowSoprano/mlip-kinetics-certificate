"""
1D overdamped double-well checks for "force RMSE vs slow kinetics".

Dynamics: dX = -V'(X) dt + sqrt(2/beta) dW,  V = B (x^2-1)^2,  beta = 1.
Generator discretised as a reversible nearest-neighbour jump process
(rate i->i+-1 = h^-2 exp(-beta (V_{i+-1}-V_i)/2)), symmetrised and diagonalised.

E0  correlation-function error vs time, against the Girsanov/Pinsker bound
E1  worst-case condition number  sup |dlam/lam| / forceRMSE  vs barrier B
E2  iso-RMSE perturbations: which metric predicts the rate error?
E3  same first-order prediction but using the MODEL's own eigenfunction
"""
import numpy as np
from scipy.linalg import eigh_tridiagonal
from scipy.stats import spearmanr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BETA = 1.0
rng = np.random.default_rng(0)


def grid(n=1500, xmax=1.8):
    x = np.linspace(-xmax, xmax, n)
    return x, x[1] - x[0]


def solve(V, h, k=2, full=False):
    """Return eigenvalues (descending, lam0=0), right eigenfunctions phi, stationary pi."""
    dV = np.diff(V)
    r_up = np.exp(-BETA * dV / 2) / h**2          # i -> i+1
    r_dn = np.exp(+BETA * dV / 2) / h**2          # i+1 -> i
    diag = np.zeros_like(V)
    diag[:-1] -= r_up
    diag[1:] -= r_dn
    off = np.full(len(V) - 1, 1.0 / h**2)         # sqrt(r_up r_dn), exactly
    n = len(V)
    if full:
        w, U = eigh_tridiagonal(diag, off)
    else:
        w, U = eigh_tridiagonal(diag, off, select="i", select_range=(n - k, n - 1))
    w, U = w[::-1], U[:, ::-1]
    pi = np.exp(-BETA * (V - V.min()))
    pi /= pi.sum()
    phi = U / np.sqrt(pi)[:, None]                # <phi_i,phi_j>_pi = delta_ij
    return w, phi, pi


def edge(pi, phi, h):
    """Edge-centred quantities consistent with the discrete Dirichlet form."""
    pe = np.sqrt(pi[:-1] * pi[1:])
    return pe, 0.5 * (phi[:-1] + phi[1:]), np.diff(phi) / h


def force_rmse(dVp, pi, h):
    dF = -np.diff(dVp) / h
    pe = np.sqrt(pi[:-1] * pi[1:])
    return np.sqrt((pe * dF**2).sum() / pe.sum()), dF


def metrics(dVp, pi, phi1, h):
    """signed first-order dlam, squared 'L_dyn' metric, force RMSE (all under pi)."""
    pe, ph, dph = edge(pi, phi1, h)
    eps, dF = force_rmse(dVp, pi, h)
    signed = (pe * ph * dF * dph).sum()                      # <phi, dF phi'>_mu
    ldyn = np.sqrt((pe * (dF * dph) ** 2).sum())             # sqrt E[(dF phi')^2]
    return signed, ldyn, eps


# ---------------------------------------------------------------- E1
def exp1():
    x, h = grid()
    Bs = np.arange(3, 17)
    kappa, kappa_bump, lam = [], [], []
    for B in Bs:
        V = B * (x**2 - 1) ** 2
        w, phi, pi = solve(V, h)
        l1, p1 = w[1], phi[:, 1]
        pe, ph, dph = edge(pi, p1, h)
        # sup over ||dF||_{L2(mu)} = eps of |<phi dF phi'>| / eps  = ||phi phi'||_{L2(mu)}
        kappa.append(np.sqrt((pe * (ph * dph) ** 2).sum()) / abs(l1))
        # concrete smooth bump at the saddle, small height (linear regime), exact re-diagonalisation
        dVp = 0.05 * np.exp(-x**2 / (2 * 0.15**2))
        w2, _, _ = solve(V + dVp, h)
        eps, _ = force_rmse(dVp, pi, h)
        kappa_bump.append(abs(np.log(w2[1] / l1)) / eps)
        lam.append(abs(l1))
    return Bs, np.array(kappa), np.array(kappa_bump), np.array(lam)


# ---------------------------------------------------------------- E2 / E3
def random_perturbation(x):
    dVp = np.zeros_like(x)
    for _ in range(rng.integers(1, 4)):
        c, wd = rng.uniform(-1.3, 1.3), rng.uniform(0.08, 0.5)
        kind = rng.integers(0, 3)
        g = np.exp(-(x - c) ** 2 / (2 * wd**2))
        if kind == 0:
            dVp += rng.normal() * g                          # bump
        elif kind == 1:
            dVp += rng.normal() * (x - c) / wd * g           # odd wiggle (no net height change)
        else:
            dVp += rng.normal() * np.cos(rng.uniform(5, 25) * x) * g   # ripple
    return dVp


def exp2(B=8.0, eps_target=0.02, n=400):
    x, h = grid()
    V = B * (x**2 - 1) ** 2
    w, phi, pi = solve(V, h)
    l1, p1 = w[1], phi[:, 1]
    rows = []
    for _ in range(n):
        dVp = random_perturbation(x)
        eps, _ = force_rmse(dVp, pi, h)
        dVp *= eps_target / eps                               # ---- iso-RMSE
        w2, phi2, pi2 = solve(V + dVp, h)
        exact = np.log(w2[1] / l1)
        signed, ldyn, eps = metrics(dVp, pi, p1, h)
        # E3: same formula evaluated with the perturbed model's own (phi_hat, mu_hat)
        p2 = phi2[:, 1] * np.sign((pi * phi2[:, 1] * p1).sum())
        signed_hat, _, _ = metrics(dVp, pi2, p2, h)
        rows.append((exact, signed / l1, ldyn / abs(l1), eps, signed_hat / w2[1],
                     np.abs(dVp).max()))
    return np.array(rows), l1


def named_cases(B=8.0, eps_target=0.02):
    x, h = grid()
    V = B * (x**2 - 1) ** 2
    w, phi, pi = solve(V, h)
    l1, p1 = w[1], phi[:, 1]
    g = lambda c, wd: np.exp(-(x - c) ** 2 / (2 * wd**2))
    cases = {
        "bump at saddle": g(0, 0.15),
        "bump at well bottom (x=+1)": g(1, 0.15),
        "bump at both wells": g(1, 0.15) + g(-1, 0.15),
        "odd wiggle at saddle": x / 0.15 * g(0, 0.15),
        "ripple everywhere (k=20)": np.cos(20 * x),
        "bump on the slope (x=0.5)": g(0.5, 0.15),
    }
    out = []
    for name, dVp in cases.items():
        eps, _ = force_rmse(dVp, pi, h)
        dVp = dVp * eps_target / eps
        w2, _, _ = solve(V + dVp, h)
        signed, ldyn, eps = metrics(dVp, pi, p1, h)
        out.append((name, eps, np.abs(dVp).max(), w2[1] / l1, np.exp(signed / l1)))
    return out


# ---------------------------------------------------------------- E0
def exp0(B=8.0, delta=1.0):
    x, h = grid(n=500)
    V = B * (x**2 - 1) ** 2
    dVp = delta * np.exp(-x**2 / (2 * 0.15**2))
    w, phi, pi = solve(V, h, full=True)
    w2, phi2, pi2 = solve(V + dVp, h, full=True)
    f = np.sign(x)
    eps_hat, _ = force_rmse(dVp, pi2, h)                     # under the MODEL's measure
    ts = np.logspace(-2, np.log10(30 / abs(w[1])), 80)

    def corr(wv, ph, piv, t):                                # E_{pi2}[ f(X0) (P_t f)(X0) ]
        coef = ph.T @ (piv * f)                              # <phi_k, f>_piv
        Ptf = ph @ (np.exp(wv * t) * coef)
        return (pi2 * f * Ptf).sum()

    err = np.array([abs(corr(w2, phi2, pi2, t) - corr(w, phi, pi, t)) for t in ts])
    bound = np.minimum(2.0, 2 * np.sqrt(ts * BETA * eps_hat**2 / 8))   # 2*TV, Pinsker, KL=t*beta*eps^2/4
    return ts, err, bound, abs(w[1]), eps_hat


if __name__ == "__main__":
    print("=== E1: condition number vs barrier ===")
    Bs, kap, kapb, lam = exp1()
    sl = np.polyfit(Bs[4:], np.log(kap[4:]), 1)[0]
    slb = np.polyfit(Bs[4:], np.log(kapb[4:]), 1)[0]
    sll = np.polyfit(Bs[4:], np.log(lam[4:]), 1)[0]
    for B, k1, k2, l in zip(Bs, kap, kapb, lam):
        print(f"B={B:2d}  |lam1|={l:.3e}  kappa_sup={k1:.3e}  kappa_bump={k2:.3e}")
    print(f"fitted slope d ln(kappa_sup)/dB  = {sl:.3f}   (theory beta/2 = 0.5, minus log corrections)")
    print(f"fitted slope d ln(kappa_bump)/dB = {slb:.3f}")
    print(f"fitted slope d ln|lam1|/dB       = {sll:.3f}   (Kramers: -1 plus log corrections)")

    print("\n=== E2: named iso-RMSE perturbations (B=8, force RMSE = 0.02) ===")
    for name, eps, amp, ratio, pred in named_cases():
        print(f"{name:30s} RMSE={eps:.4f}  max|dV|={amp:8.4f}  rate_hat/rate={ratio:8.4f}  1st-order pred={pred:8.4f}")

    print("\n=== E2/E3: random iso-RMSE ensemble ===")
    res = {}
    for eps_t in (0.005, 0.02):
        R, l1 = exp2(eps_target=eps_t)
        ex = R[:, 0]
        print(f"-- force RMSE fixed at {eps_t}: rate ratio ranges {np.exp(ex.min()):.3f} .. {np.exp(ex.max()):.3f}")
        print(f"   signed (true phi)   : Spearman vs exact = {spearmanr(R[:,1], ex)[0]:.3f},  median |pred-exact| = {np.median(abs(R[:,1]-ex)):.4f}")
        print(f"   signed (model phi^) : Spearman vs exact = {spearmanr(R[:,4], ex)[0]:.3f},  median |pred-exact| = {np.median(abs(R[:,4]-ex)):.4f}")
        print(f"   squared L_dyn       : Spearman vs |exact| = {spearmanr(R[:,2], abs(ex))[0]:.3f}")
        big = abs(ex) > 0.1
        if big.sum() > 5:
            overs = R[:, 2] / np.maximum(abs(ex), 1e-12)
            print(f"   squared L_dyn / |exact| : median {np.median(overs):.1f}x, 90th pct {np.percentile(overs,90):.1f}x  (how loose the squared upper bound is)")
        res[eps_t] = R

    print("\n=== E0: correlation error vs Pinsker bound (B=8, saddle bump 1 kT) ===")
    ts, err, bound, l1abs, eps_hat = exp0()
    t_vac = 8 / (BETA * eps_hat**2)
    print(f"force RMSE under model measure = {eps_hat:.4f};  bound becomes vacuous at t = {t_vac:.3e};  1/|lam1| = {1/l1abs:.3e}")
    print(f"max correlation error = {err.max():.3f} at t = {ts[err.argmax()]:.3e}")

    # ---------------- raw data (CSV so pgfplots can read it directly)
    import os
    os.makedirs("data", exist_ok=True)
    np.savetxt("data/dw_E1_condition_number.csv", np.column_stack([Bs, lam, kap, kapb]), delimiter=",",
               header="B_kT,abs_lambda1,kappa_sup,kappa_bump", comments="")
    np.savetxt("data/dw_E0_correlation_error.csv", np.column_stack([ts, err, bound]), delimiter=",",
               header="t,abs_corr_error,pinsker_bound", comments="")
    cols = "exact_ln_ratio,signed_pred_true_phi,squared_Ldyn_over_lambda,force_rmse,signed_pred_model_phi,max_abs_dV"
    for eps_t, Rr in res.items():
        np.savetxt(f"data/dw_E2_ensemble_B8_eps{eps_t}.csv", Rr, delimiter=",", header=cols, comments="")
    Rbig, _ = exp2(B=12.0, eps_target=0.1)
    np.savetxt("data/dw_E2_ensemble_B12_eps0.1.csv", Rbig, delimiter=",", header=cols, comments="")
    with open("data/dw_E2_named_cases.csv", "w") as fh:
        fh.write("B_kT,case,force_rmse,max_abs_dV,rate_ratio_exact,rate_ratio_first_order\n")
        for Bv, ev in ((8.0, 0.02), (12.0, 0.1)):
            for name, e_, amp, ratio, pred in named_cases(Bv, ev):
                fh.write(f"{Bv},{name},{e_},{amp},{ratio},{pred}\n")

    fig, ax = plt.subplots(2, 2, figsize=(11, 8.5))
    a = ax[0, 0]
    a.loglog(ts, err, label="actual |C_hat(t) - C(t)|, f = sign(x)")
    a.loglog(ts, bound, "--", label="Girsanov + Pinsker bound")
    a.axvline(1 / l1abs, color="gray", ls=":", label="1/|lambda_1| (hopping time)")
    a.set_xlabel("t"); a.set_title("E0: force RMSE guarantees dynamics only for t << 1/eps^2"); a.legend(fontsize=8)

    a = ax[0, 1]
    a.semilogy(Bs, kap, "o-", label=f"sup over all dF (slope {sl:.2f})")
    a.semilogy(Bs, kapb, "s-", label=f"smooth saddle bump (slope {slb:.2f})")
    a.semilogy(Bs, kap[5] * np.exp(0.5 * (Bs - Bs[5])), "k--", label="exp(beta B / 2)")
    a.set_xlabel("barrier B / kT"); a.set_ylabel("|d ln rate| / force RMSE")
    a.set_title("E1: condition number of kinetics w.r.t. force RMSE"); a.legend(fontsize=8)

    R = res[0.02]
    a = ax[1, 0]
    a.scatter(R[:, 1], R[:, 0], s=8)
    lim = max(abs(R[:, :2]).max(), 1e-3)
    a.plot([-lim, lim], [-lim, lim], "k--", lw=0.8)
    a.set_xlabel("signed first-order prediction  <phi, dF phi'> / lambda")
    a.set_ylabel("exact ln(rate_hat / rate)")
    a.set_title("E2: all points have IDENTICAL force RMSE = 0.02")

    a = ax[1, 1]
    a.scatter(R[:, 2], abs(R[:, 0]) + 1e-6, s=8)
    a.set_xscale("log"); a.set_yscale("log")
    a.set_xlabel("squared metric  sqrt E[(dF phi')^2] / |lambda|")
    a.set_ylabel("|exact ln(rate_hat / rate)|")
    a.set_title("E2: squared L_dyn is an upper bound, but loose")
    plt.tight_layout()
    plt.savefig("doublewell_check.png", dpi=140)
    print("\nsaved doublewell_check.png")
