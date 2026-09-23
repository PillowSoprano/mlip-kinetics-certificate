"""
A bracket valid at every friction (above the energy-diffusion regime), tested against the exact 1D Klein-Kramers spectrum.

Kramers-Langer regime: k = k_TST * Lambda(gamma/omega_b),  Lambda(x) = sqrt(1 + x^2/4) - x/2,  omega_b = barrier frequency.
For a perturbation dV that is smooth on the thermal width of the barrier top the perturbed barrier is again parabolic, so

    ln(k_hat/k) = R_TST + ln Lambda(gamma/omega_b_hat) - ln Lambda(gamma/omega_b),        R_OD = R_TST + ln(omega_b_hat/omega_b)

Claim to test: ln(k_hat/k) is MONOTONE in gamma and therefore bracketed by its two friction-independent limits R_TST and R_OD,
both of which are equilibrium (reference-only) quantities.  In 1D:
    R_TST = -(max V_hat - max V) + basin term                      (variational TST; arithmetic/parallel law)
    R_OD  = ln G - ln < e^{dV} >_nu                                 (exact overdamped formula; harmonic/series law)
"""
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigs
import friction_crossover as fc


def setup(B, nx=420, nh=40):
    fc.B, fc.NX, fc.NH = B, nx, nh
    fc.x = np.linspace(-fc.XMAX, fc.XMAX, nx); fc.h = fc.x[1] - fc.x[0]
    return fc.x


def slow_rate(V, gamma, guess):
    dV = np.gradient(V, fc.h); NX, NH = fc.NX, fc.NH
    D = sp.diags([-np.ones(NX - 1), np.ones(NX - 1)], [-1, 1], format="lil") / (2 * fc.h)
    D[0, :2] = [-1 / fc.h, 1 / fc.h]; D[-1, -2:] = [-1 / fc.h, 1 / fc.h]; D = D.tocsr()
    blocks = [[None] * NH for _ in range(NH)]
    for m in range(NH):
        blocks[m][m] = -gamma * m * sp.identity(NX, format="csr")
        if m + 1 < NH: blocks[m][m + 1] = np.sqrt(m + 1) * (D - sp.diags(dV))
        if m >= 1: blocks[m][m - 1] = np.sqrt(m) * D
    L = sp.bmat(blocks, format="csc")
    for shift in (0.3, 1.0, 3.0, 0.1, 10.0):                       # the Kramers guess can be off by a factor of a few
        ev = eigs(L, k=6, sigma=shift * guess, which="LM", return_eigenvectors=False)
        ev = np.sort(-ev[np.abs(ev.imag) < 1e-3 * np.abs(ev.real) + 1e-9 * guess].real)
        ev = ev[(ev > 0.05 * guess) & (ev < 20 * guess)]             # the slow mode must be near the Kramers guess; fast modes are 1e3+ times larger
        if len(ev): return ev[0]
    raise RuntimeError("slow eigenvalue not found")


def limits(V, d):
    x = fc.x; mid = np.abs(x) < 0.9; A = x < 0
    mu = np.exp(-V); mu /= mu.sum(); aA = (mu[A] * np.exp(-d[A])).sum() / mu[A].sum(); aB = (mu[~A] * np.exp(-d[~A])).sum() / mu[~A].sum()
    lnG = np.log(mu[A].sum() / aB + mu[~A].sum() / aA)
    nu = np.exp(V) * mid; nu /= nu.sum()
    r_od = lnG - np.log((nu * np.exp(d)).sum())
    r_tst = lnG - ((V + d)[mid].max() - V[mid].max())
    i = np.argmax((V + d)[mid]); xm = x[mid][i]
    wb = np.sqrt(max(-np.gradient(np.gradient(V + d, fc.h), fc.h)[mid][i], 1e-9))
    return r_tst, r_od, wb


def lam(gamma, w):
    return np.sqrt(1 + (gamma / w) ** 2 / 4) - gamma / w / 2


if __name__ == "__main__":
    B = 8.0; x = setup(B); V = B * (x**2 - 1) ** 2; wb0 = np.sqrt(4 * B)
    gammas = np.array([1.0, 2, 4, 8, 16, 32, 64, 128])               # gamma/omega_b from 0.18 to 23
    kram = lambda g, w: np.sqrt(8 * B) / (2 * np.pi) * lam(g, w) * np.exp(-B) * 2
    k0 = {g: slow_rate(V, g, kram(g, wb0)) for g in gammas}
    print(f"B = {B} kT, omega_b = {wb0:.2f}; reference rate / Kramers-Langer estimate:", {float(g): round(k0[g] / kram(g, wb0), 3) for g in gammas})
    rng = np.random.default_rng(0); rows = []; detail = []
    for c in range(24):
        w = rng.uniform(0.25, 0.6); amp = rng.uniform(-2.5, 2.5); c0 = rng.uniform(-0.15, 0.15)
        d = amp * np.exp(-(x - c0) ** 2 / (2 * w**2)) + rng.uniform(-0.6, 0.6) * np.exp(-(x - 1) ** 2 / (2 * 0.3**2))
        r_tst, r_od, wbh = limits(V, d)
        ex = np.array([np.log(slow_rate(V + d, g, k0[g] * np.exp(r_od)) / k0[g]) for g in gammas])      # the overdamped value is the best available guess
        kl = np.array([r_tst + np.log(lam(g, wbh) / lam(g, wb0)) for g in gammas])
        lo, hi = min(r_tst, r_od), max(r_tst, r_od)
        sel = gammas / wb0 >= 0.5                                     # spatial-diffusion / turnover regime; below that the rate is energy-diffusion limited
        mono = np.all(np.diff(ex[sel]) * np.sign(r_od - r_tst) >= -0.01)
        inside = np.all((ex[sel] >= lo - 0.03) & (ex[sel] <= hi + 0.03))
        rows.append((w, amp, r_tst, r_od, ex.min(), ex.max(), np.abs(ex - kl)[sel].max(), mono, inside, np.abs(ex[-1] - r_od), ex[0] - r_od))
        detail.append(np.r_[c, r_tst, r_od, ex, kl])
        print(f"case {c:2d}: width {w:.2f} amp {amp:+.2f} | R_TST {r_tst:+.3f}  R_OD {r_od:+.3f} | exact over gamma: [{ex.min():+.3f}, {ex.max():+.3f}] | max |exact - Kramers-Langer| {np.abs(ex-kl).max():.3f} | monotone {mono} | inside {inside}", flush=True)
    R = np.array(rows, dtype=float)
    print(f"\n24 smooth perturbations x {len(gammas)} frictions (gamma/omega_b = {gammas[0]/wb0:.2f} ... {gammas[-1]/wb0:.0f}):")
    print(f"   exact value inside [min(R_TST, R_OD), max(...)] (tolerance 0.03): {int(R[:,8].sum())}/24;   monotone in gamma: {int(R[:,7].sum())}/24")
    print(f"   high-friction end vs R_OD: max |diff| = {R[:,9].max():.3f};   lowest friction (gamma/omega_b = {gammas[0]/wb0:.2f}) minus R_OD: range [{R[:,10].min():+.3f}, {R[:,10].max():+.3f}]")
    print(f"   Kramers-Langer interpolation error: median {np.median(R[:,6]):.3f}, max {R[:,6].max():.3f};   bracket width |R_OD - R_TST|: median {np.median(abs(R[:,3]-R[:,2])):.3f}, max {abs(R[:,3]-R[:,2]).max():.3f}")
    np.savetxt("data/friction_bracket_1d.csv", np.array(detail), delimiter=",", comments="",
               header="case,R_TST,R_OD," + ",".join(f"exact_g{g:g}" for g in gammas) + "," + ",".join(f"KL_g{g:g}" for g in gammas))
