"""
Nonlinear two-sided bracket for the slow rate (numerical test of the Dirichlet-Thomson bracket, Theorem 1).

With nu  = |grad phi|^2 mu / norm   (leverage density of the REFERENCE dynamics)
     a_X = < exp(-beta dV) >_{mu restricted to basin X},   G = mu_A / a_B + mu_B / a_A     (basin factor)

    G / < exp(+beta dV) >_nu   <=   k_hat / k   <=   G * < exp(-beta dV) >_nu            (metastable limit)

lower = Thomson principle with the unperturbed reactive flux, upper = Dirichlet principle with the unperturbed committor.
Both reduce to the first-order formula when beta^2 Var_nu(dV) << 1; in 1D the lower bound should be exact.
The same bracket evaluated with MODEL quantities (nu_hat, -dV) brackets k / k_hat without any reference dynamics.
"""
import numpy as np
import doublewell_check as d
import twochannel_blindspot as t


def bracket(dV, pi, g2, basinA):
    """returns (ln lower, ln upper, first-order ln ratio, beta^2 Var_nu dV)."""
    nu = g2 * pi; nu = nu / nu.sum()
    piA, piB = pi[basinA].sum(), pi[~basinA].sum()
    aA = (pi[basinA] * np.exp(-dV[basinA])).sum() / piA
    aB = (pi[~basinA] * np.exp(-dV[~basinA])).sum() / piB
    G = piA / aB + piB / aA
    lo = np.log(G) - np.log((nu * np.exp(dV)).sum())
    up = np.log(G) + np.log((nu * np.exp(-dV)).sum())
    m = (nu * dV).sum()
    first = -(m - (pi * dV).sum() - 0.0)                       # crude first order for reference only
    return lo, up, first, (nu * (dV - m) ** 2).sum()


if __name__ == "__main__":
    # ------------------------------------------------ 1D, B = 12, iso-RMSE ensemble (eps = 0.1) + named cases
    B = 12.0
    x, h = d.grid(); V = B * (x**2 - 1) ** 2
    w, phi, pi = d.solve(V, h); k = abs(w[1]); g2 = np.gradient(phi[:, 1], h) ** 2; A = x < 0
    d.rng = np.random.default_rng(0)
    rows = []
    for i in range(400):
        dV = d.random_perturbation(x); dV *= 0.1 / d.force_rmse(dV, pi, h)[0]
        ex = np.log(abs(d.solve(V + dV, h)[0][1]) / k)
        lo, up, _, var = bracket(dV, pi, g2, A)
        rows.append((ex, lo, up, var))
    R = np.array(rows)
    tol = 0.02
    print("1D ensemble (400 iso-RMSE perturbations, B=12, eps=0.1):")
    print(f"   exact inside [lower-{tol}, upper+{tol}] : {np.mean((R[:,0] >= R[:,1]-tol) & (R[:,0] <= R[:,2]+tol))*100:.1f}%")
    print(f"   |exact - lower|  median {np.median(abs(R[:,0]-R[:,1])):.4f}   max {abs(R[:,0]-R[:,1]).max():.4f}    (1D: Thomson bound should be exact)")
    print(f"   bracket width    median {np.median(R[:,2]-R[:,1]):.4f}   max {(R[:,2]-R[:,1]).max():.3f}")
    big = R[:, 3] > 0.05
    print(f"   corr(width, beta^2 Var_nu dV) = {np.corrcoef(R[:,2]-R[:,1], R[:,3])[0,1]:.3f};  width/Var median (where Var>0.05): {np.median((R[big,2]-R[big,1])/R[big,3]):.2f}")
    np.savetxt("data/bracket_1d_ensemble.csv", R, delimiter=",", header="exact,lower,upper,var_nu", comments="")
    print("   named cases:  exact | lower | upper")
    for name, dV in (("saddle bump", np.exp(-x**2/(2*.15**2))), ("odd wiggle", x/.15*np.exp(-x**2/(2*.15**2))), ("one well", np.exp(-(x-1)**2/(2*.15**2)))):
        dV = dV * 0.1 / d.force_rmse(dV, pi, h)[0]
        ex = np.log(abs(d.solve(V + dV, h)[0][1]) / k); lo, up, _, var = bracket(dV, pi, g2, A)
        print(f"     {name:12s} {ex:+.3f} | {lo:+.3f} | {up:+.3f}     beta^2 Var_nu = {var:.2f}")

    # ------------------------------------------------ 2D two-channel: reference-side and model-side brackets
    print("\n2D two-channel, blocking scan.  ln(k_hat/k): exact, reference-side bracket, and MODEL-side bracket (uses only phi_hat, mu_hat, dV)")
    Vt = t.true_potential(); lam, ph, pi2, _ = t.slow_mode(Vt); lam = abs(lam)
    gx, gy = np.gradient(ph, t.h); g2t = gx**2 + gy**2; A2 = t.X < 0
    out = []
    for block in (1, 3, 6, 9, 12, 18):
        t.BLOCK = float(block); t.rng = np.random.default_rng(1); dV = t.initial_error()
        lh, phh, pih, _ = t.slow_mode(Vt + dV); lh = abs(lh)
        ex = np.log(lh / lam)
        lo, up, _, _ = bracket(dV, pi2, g2t, A2)
        gxh, gyh = np.gradient(phh, t.h)
        lo_m, up_m, _, _ = bracket(-dV, pih, gxh**2 + gyh**2, A2)       # brackets ln(k / k_hat)
        out.append((block, ex, lo, up, -up_m, -lo_m))
        print(f"   block={block:2d}kT  exact={ex:+.3f}   ref-side [{lo:+.3f}, {up:+.3f}]   model-side [{-up_m:+.3f}, {-lo_m:+.3f}]")
    np.savetxt("data/bracket_2d_blockscan.csv", out, delimiter=",", header="block_kT,exact,ref_lower,ref_upper,model_lower,model_upper", comments="")
