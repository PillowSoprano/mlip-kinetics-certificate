"""
Transition-count bound (Proposition 4): what the force error guarantees about a rare-event frequency.

Both dynamics start from the model equilibrium mu_hat.  The relative entropy of the two path measures on [0,T] is
beta T eps_hat^2 / 4 (Proposition 1); by data processing it bounds the relative entropy of the probability that an
event occurs in [0,T].  For tau_mol << T << 1/k that probability is nu T, and T cancels:

    h(nu_hat / nu) <= beta eps_hat^2 / (4 nu),        h(x) = x ln x - x + 1,
    to leading order  |d ln nu| <= eps_hat sqrt(beta / (2 nu)).

For Langevin dynamics with masses m_i and friction gamma, beta eps_hat^2 / 4 -> sum_i <|dF_i|^2> / (4 m_i gamma kT).

A  first-order supremum of |d ln nu| over errors of force error eps against the leading-order bound, double well
   V = B (x^2-1)^2, beta B = 1..16; the ratio tends to 1/sqrt(3).
B  finite perturbations of DW12 and DW8 at increasing force error: exact nu_hat / nu against the bound.
C  Li3OCl: the hop-rate ratios that the held-out force errors of the students admit at 1000 K.
    python transition_count_bound.py
"""
import csv
import numpy as np
from scipy.optimize import brentq
import doublewell_check as d

BETA = 1.0


def h(x):
    return x * np.log(x) - x + 1.0


def first_order_sup(B, n=400001, xmax=2.0):
    """sup |d ln nu| / eps and the bound sqrt(beta / 2 nu) / eps on a fine grid; cores A = {x <= -1}, B = {x >= 1}."""
    x = np.linspace(-xmax, xmax, n); dx = x[1] - x[0]
    w = np.exp(-BETA * B * (x**2 - 1) ** 2); mu = w / (w.sum() * dx)
    inner = (x > -1) & (x < 1)
    qp = np.where(inner, 1.0 / mu, 0.0); qp /= qp.sum() * dx          # q' propto 1/mu between the cores
    nu_R = np.sum(qp**2 * mu) * dx / BETA                             # A->B transitions per unit time
    nu = 2 * nu_R                                                     # counted in both directions
    lev = qp**2 * mu; lev /= lev.sum() * dx
    dens = lev - mu                                                   # d ln nu = -beta int dV (lev - mu)
    G = np.where(x < 0, np.cumsum(dens) * dx, -(np.cumsum(dens[::-1]) * dx)[::-1])
    sup = BETA * np.sqrt(np.sum(G**2 / mu) * dx)                      # attained by dV* with mu dV*' propto G
    return nu, sup, np.sqrt(BETA / (2 * nu))


def partA():
    rows = []
    for B in range(1, 17):
        nu, sup, bound = first_order_sup(B)
        rows.append((B, nu, sup, bound, sup / bound))
        print(f"  beta B = {B:2d}   nu = {nu:.3e}   sup = {sup:.4e}   bound = {bound:.4e}   ratio = {sup / bound:.4f}")
    print(f"  1/sqrt(3) = {1 / np.sqrt(3):.4f}")
    return rows


def partB(B, eps_list=(0.01, 0.03, 0.1, 0.3, 1.0), n=100):
    x, hh = d.grid()
    V = B * (x**2 - 1) ** 2
    w, _, pi = d.solve(V, hh)
    k = -w[1]
    nu = k / 2                               # symmetric reference: mu_hat(A) k_AB + mu_hat(B) k_BA = k / 2
    rows = []
    for eps_t in eps_list:
        for _ in range(n):
            dVp = d.random_perturbation(x)
            e0, _ = d.force_rmse(dVp, pi, hh)
            dVp *= eps_t / e0
            w2, _, pi2 = d.solve(V + dVp, hh)
            k2 = -w2[1]
            pA = pi2[x < 0].sum()
            nu2 = 2 * pA * (1 - pA) * k2     # stationary transition frequency of the model
            eps_hat, _ = d.force_rmse(dVp, pi2, hh)
            lhs, rhs = h(nu2 / nu), BETA * eps_hat**2 / (4 * nu)
            lin = abs(np.log(nu2 / nu)) / (eps_hat * np.sqrt(BETA / (2 * nu)))
            rows.append((B, eps_t, eps_hat, np.log(nu2 / nu), lhs, rhs, lhs / rhs, lin))
    a = np.array(rows)
    for eps_t in eps_list:
        s = a[a[:, 1] == eps_t]
        print(f"  DW{B}  eps = {eps_t:5.2f}   |ln nu_hat/nu| up to {np.abs(s[:, 3]).max():7.3f}   "
              f"max h/bound = {s[:, 6].max():.3f}   max |d ln nu| / linear bound = {s[:, 7].max():.3f}")
    return rows


def partC():
    """Li3OCl, 1000 K, BAOAB friction 10 / ps.  Teacher: all hops over the whole run (single-vacancy and extra-defect)."""
    amu = 1.0364269e-4                       # eV ps^2 / A^2
    kT = 8.617333e-5 * 1000.0
    gamma = 10.0
    counts = {"Li": (80, 6.94), "O": (27, 15.999), "Cl": (27, 35.45)}
    teacher = next(r for r in csv.DictReader(open("data/li3ocl_defect_states.csv")) if r["name"] == "teacher_1000K_censored")
    hops = int(teacher["hops_single"]) + int(teacher["hops_defect"])
    ns = float(teacher["ns_single"]) + float(teacher["ns_defect"])
    nu = hops / ns / 1000.0                  # per ps
    rmse = {"S1": 6.1, "S2": 6.5, "S3": 7.5, "S4": 9.3, "S5": 13.7, "S6": 19.9}   # meV/A per component, held out
    m_max = max(m for _, m in counts.values())
    rows = []
    print(f"  teacher hop frequency {nu * 1e3:.1f} / ns ({hops} hops in {ns:.2f} ns)")
    for s, r in rmse.items():
        s2 = (r * 1e-3) ** 2
        I_mass = sum(n * 3 * s2 / (4 * m * amu * gamma * kT) for n, m in counts.values())   # force error shared equally
        I_min = sum(n for n, _ in counts.values()) * 3 * s2 / (4 * m_max * amu * gamma * kT)   # every atom as heavy as Cl
        out = []
        for I in (I_min, I_mass):
            c = I / nu
            upper = brentq(lambda y: h(y) - c, 1.0, 1e9)
            lower = brentq(lambda y: h(y) - c, 1e-300, 1.0) if c < 1 else 0.0
            out += [I, lower, upper]
        rows.append((s, r, *out))
        print(f"  {s}  {r:5.1f} meV/A   all atoms as Cl: nu_hat/nu in [{out[1]:.2f}, {out[2]:.1f}]   "
              f"actual masses: [{out[4]:.2f}, {out[5]:.1f}]")
    return nu, rows


def main():
    print("A  first-order supremum against the bound")
    A = partA()
    with open("data/transition_count_bound_dw.csv", "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["betaB", "nu", "sup_dlnnu_per_eps", "bound_per_eps", "ratio"])
        w.writerows([[B, f"{a:.6e}", f"{b:.6e}", f"{c:.6e}", f"{r:.5f}"] for B, a, b, c, r in A])
    print("B  finite perturbations")
    Brows = partB(12.0, eps_list=(0.01, 0.03, 0.1, 0.3)) + partB(8.0)
    b = np.array([r[6] for r in Brows])
    print(f"  {len(b)} perturbations: h(nu_hat/nu) / bound at most {b.max():.3f}; inside the bound in {np.mean(b <= 1):.0%}")
    with open("data/transition_count_bound_finite.csv", "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["B", "eps_target", "eps_hat", "ln_nu_ratio", "h", "bound", "h_over_bound", "lin_ratio"])
        w.writerows([[f"{v:.6g}" for v in r] for r in Brows])
    print("C  Li3OCl students at 1000 K")
    nu, C = partC()
    with open("data/transition_count_bound_li3ocl.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["model", "force_rmse_meVA", "I_allCl_per_ps", "lower_allCl", "upper_allCl", "I_masses_per_ps", "lower_masses", "upper_masses"])
        w.writerows([[r[0], r[1]] + [f"{v:.4g}" for v in r[2:]] for r in C])


if __name__ == "__main__":
    main()
