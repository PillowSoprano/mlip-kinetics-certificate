"""
Two-dimensional matched-saddle test.

Ring potential, wells at (+-1,0), saddles at (0,+-1) with barriers 5 (lower channel) and 7 kT (upper).
Perturbations are compactly supported bumps on the ring, placed strictly between a well and a saddle and kept at least
four grid spacings away from both, so that V, its gradient and its Hessian at all four critical points are unchanged:
harmonic transition-state theory summed over the two channels returns the same rate for every member. All members share
one force RMSE. Two series: the bump in the lower (fast) channel and the same bump in the upper (slow) channel.
Unlike one dimension the bracket is tight on neither side here.
    python matched_saddle_2d.py
"""
import numpy as np
import twochannel_blindspot as t2
from bracket_check import bracket

EPS, RR, RPHI = 0.60, 0.32, 0.30
X, Y, h = t2.X, t2.Y, t2.h
Rr = np.sqrt(X**2 + Y**2); PH = np.arctan2(Y, X)
V = t2.true_potential()
lam, phi, pi, _ = t2.slow_mode(V)
g2 = np.gradient(phi, h, axis=0) ** 2 + np.gradient(phi, h, axis=1) ** 2
basinA = X < 0
crit = [(1.0, 0.0), (-1.0, 0.0), (0.0, 1.0), (0.0, -1.0)]
nu = g2 * pi; nu = nu / nu.sum()

def bump(phi0):
    d = np.angle(np.exp(1j * (PH - phi0)))
    t = np.sqrt(((Rr - 1.0) / RR) ** 2 + (d / RPHI) ** 2)
    f = np.zeros_like(X); m = t < 1; f[m] = np.exp(-1.0 / (1.0 - t[m] ** 2))
    return f

def clearance(f):
    """distance from the support of f to the nearest well or saddle, in grid spacings"""
    s = f > 0
    if not s.any(): return np.inf
    return min(np.sqrt((X[s] - cx) ** 2 + (Y[s] - cy) ** 2).min() for cx, cy in crit) / h

def local_check(dV):
    gx, gy = np.gradient(dV, h, axis=0), np.gradient(dV, h, axis=1)
    hxx, hyy, hxy = np.gradient(gx, h, axis=0), np.gradient(gy, h, axis=1), np.gradient(gx, h, axis=1)
    out = 0.0
    for cx, cy in crit:
        m = (np.abs(X - cx) <= 3 * h) & (np.abs(Y - cy) <= 3 * h)
        out = max(out, *[np.abs(a[m]).max() for a in (dV, gx, gy, hxx, hyy, hxy)])
    return out

def scale(f):
    pex = np.sqrt(pi[:-1, :] * pi[1:, :]); pey = np.sqrt(pi[:, :-1] * pi[:, 1:])
    fx = -np.diff(f, axis=0) / h; fy = -np.diff(f, axis=1) / h
    return EPS / np.sqrt(((pex * fx**2).sum() + (pey * fy**2).sum()) / (pex.sum() + pey.sum()))

print(f"reference: barriers {t2.B_DN:.0f} (lower) / {t2.B_UP:.0f} kT (upper), lambda_1 = {lam:.4e}, grid spacing {h:.3f}")
print("\n channel  ang. dist.  clear  peak  | local |dV|,|grad|,|Hess| | nu mass | ln(k_hat/k)           | harmonic | bracket             width")
rows = []
for chan, base in (("lower", -np.pi / 2), ("upper", +np.pi / 2)):
    for dphi in (0.48, 0.62, 0.80, 1.00):                       # angular distance from the saddle
        phi0 = base + np.sign(base) * dphi * -1 if base < 0 else base - dphi
        phi0 = base + dphi if base < 0 else base - dphi          # move from the saddle towards the well at phi=0
        for sgn in (+1, -1):
            f = sgn * bump(phi0); dV = f * scale(np.abs(f))
            cl = clearance(np.abs(f))
            if cl < 6: print(f" {chan:>6s} {dphi:8.2f}  SKIPPED (support only {cl:.1f} grid spacings from a critical point)"); continue
            ex = np.log(t2.slow_mode(V + dV)[0] / lam)
            lo, up, first, var = bracket(dV.ravel(), pi.ravel(), g2.ravel(), basinA.ravel())
            mass = nu[np.abs(dV) > 0].sum()
            print(f" {chan:>6s} {dphi:8.2f} {cl:6.1f} {np.abs(dV).max():5.2f} | {local_check(dV):22.1e} | {mass:7.3f} | {ex:+.3f} (x {np.exp(ex):5.2f}) | {0.0:+.3f}   | [{lo:+.3f},{up:+.3f}] {up-lo:.3f}")
            rows.append((0 if chan == "lower" else 1, dphi, sgn, EPS, ex, lo, up, first, var, mass, np.abs(dV).max()))
R = np.array(rows)
ins = ((R[:, 4] >= R[:, 5] - 1e-3) & (R[:, 4] <= R[:, 6] + 1e-3)).sum()
np.savetxt("data/matched_saddle_2d.csv", R, delimiter=",", comments="",
           header="channel_upper,dphi_from_saddle,sign,force_rmse,exact_ln_ratio,lower,upper,first_order,var_nu,nu_mass,peak_dV_kT")
lo_ = R[R[:, 0] == 0]; up_ = R[R[:, 0] == 1]
print(f"\nexact rate ratio spans {np.exp(R[:,4]).min():.2f}--{np.exp(R[:,4]).max():.2f} at one force RMSE; every harmonic prediction is 1.00")
print(f"largest effect: lower channel {np.abs(lo_[:,4]).max():.3f}, upper channel {np.abs(up_[:,4]).max():.3f} in |ln(k_hat/k)|")
print(f"inside the bracket: {ins}/{len(R)};  bracket width median {np.median(R[:,6]-R[:,5]):.3f}, max {np.max(R[:,6]-R[:,5]):.3f}")
