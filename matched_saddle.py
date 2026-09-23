"""
Does the leverage density measure anything that local saddle data does not?

Family of perturbations that are IDENTICALLY ZERO in a neighbourhood of the saddle and of both wells:
compactly supported bumps at +-c inside the barrier region. Every member leaves V, V' and V'' at x = 0 and
x = +-1 exactly unchanged, so harmonic transition-state theory and Kramers theory at any friction predict
the same rate for all of them and for the reference. All members are scaled to a common force RMSE.
Compared: the exact slow rate, the harmonic/Kramers prediction (identically zero), and the bracket.
    python matched_saddle.py
"""
import numpy as np
import doublewell_check as d
from bracket_check import bracket

B, EPS, R = 8.0, 0.30, 0.08
x, h = d.grid(); V = B * (x**2 - 1) ** 2
w, phi, pi = d.solve(V, h); k = abs(w[1]); g2 = np.gradient(phi[:, 1], h) ** 2; A = x < 0

def bump(c, r=R):
    """C^infinity, supported exactly on [c-r, c+r] and its mirror image."""
    f = np.zeros_like(x)
    for s in (+1, -1):
        t = (x - s * c) / r; m = np.abs(t) < 1
        f[m] += np.exp(-1.0 / (1.0 - t[m] ** 2))
    return f

omega_b = np.sqrt(4 * B); omega_0 = np.sqrt(8 * B)      # V'' = 4B(3x^2-1)
print(f"reference: barrier {B:.1f} kT, omega_b {omega_b:.2f}, omega_0 {omega_0:.2f}, exact rate {k:.3e}")
print("\n c    supp      | local data unchanged?            | ln(k_hat/k)            | harmonic  | bracket")
print("                 | dV(0)   dV''(0)  dV(1)  dV''(1)  | exact                  | TST/Kramers|")
rows = []
for c in (0.11, 0.16, 0.22, 0.30, 0.40, 0.52, 0.66, 0.82):
    for sgn in (+1, -1):
        f = sgn * bump(c); dV = f * (EPS / d.force_rmse(f, pi, h)[0])
        i0 = np.argmin(abs(x)); i1 = np.argmin(abs(x - 1.0))
        dd = np.gradient(np.gradient(dV, h), h)
        ex = np.log(abs(d.solve(V + dV, h)[0][1]) / k)
        lo, up, first, var = bracket(dV, pi, g2, A)
        nu = g2 * pi; nu /= nu.sum(); mass = nu[(x > c - R) & (x < c + R)].sum() + nu[(x < -c + R) & (x > -c - R)].sum()
        print(f" {sgn*c:+.2f} [{c-R:.2f},{c+R:.2f}] | {dV[i0]:7.0e} {dd[i0]:8.0e} {dV[i1]:7.0e} {dd[i1]:8.0e} | {ex:+.3f}  (x {np.exp(ex):5.2f})  | {0.0:+.3f}    | [{lo:+.3f},{up:+.3f}]")
        rows.append((sgn * c, EPS, ex, lo, up, first, var, mass, dV.max() if sgn > 0 else dV.min()))
np.savetxt("data/matched_saddle.csv", rows, delimiter=",", comments="",
           header="c_signed,force_rmse,exact_ln_ratio,lower,upper,first_order,var_nu,nu_mass_on_support,peak_dV_kT")
R_ = np.array(rows); ins = ((R_[:, 2] >= R_[:, 3] - 1e-4) & (R_[:, 2] <= R_[:, 4] + 1e-4)).sum()
print(f"\nexact rate ratio spans {np.exp(R_[:,2]).min():.2f}--{np.exp(R_[:,2]).max():.2f} at a single force RMSE, while every harmonic prediction is 1.00")
print(f"inside the bracket: {ins}/{len(R_)};  the effect tracks the leverage mass on the support (Spearman {np.corrcoef(np.argsort(np.argsort(abs(R_[:,2]))), np.argsort(np.argsort(R_[:,7])))[0,1]:+.2f})")
