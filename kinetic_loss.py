"""
Does a bracket-based training loss give better kinetics at the same label budget?

True PES: 2D two-channel ring potential.  Student: V_theta(x) = sum_k theta_k RBF_k(x)  (deliberately too small to be exact).
Label budget N is fixed.  Three students:
   FM          force matching on N equilibrium samples                                  (what everyone does)
   FM+TS       force matching on N-M equilibrium samples + M samples from the leverage density nu   (the chemist's heuristic)
   FM+TS+kin   same data as FM+TS, plus the kinetic loss on the ENERGIES of the M leverage samples:
                   L_kin = [ ln<e^{+dV}>_nu + ln<e^{-dV}>_nu ]  +  ( <dV>_nu - <dV>_basins )^2
               (bracket width + squared bracket centre; both invariant under a constant shift of dV)
Metric: exact |ln(k_hat/k)| from the generator of the student potential, and the equilibrium force RMSE on held-out data.
"""
import numpy as np, torch
import twochannel_blindspot as t
import tube_bounds as tb

torch.set_default_dtype(torch.float64)
N_TOT, M_TS, NSEED, LAM = 300, 40, 8, 30.0
cx = np.arange(-1.8, 1.81, 0.3); CX, CY = np.meshgrid(cx, cx, indexing="ij"); CEN = np.column_stack([CX.ravel(), CY.ravel()]); WID = 0.33


def V_true(p):
    x, y = p[:, 0], p[:, 1]; r2 = x**2 + y**2; r = np.sqrt(r2)
    s2 = y**2 / (r2 + 0.01); s = y / np.sqrt(r2 + 0.01)
    return t.KR * (r - 1) ** 2 + s2 * (0.5 * (t.B_UP + t.B_DN) + 0.5 * (t.B_UP - t.B_DN) * s)


def F_true(p, e=1e-5):
    g = np.zeros_like(p)
    for d in (0, 1):
        dp = np.zeros_like(p); dp[:, d] = e
        g[:, d] = (V_true(p + dp) - V_true(p - dp)) / (2 * e)
    return -g


def feats(p):
    d = p[:, None, :] - CEN[None, :, :]; phi = np.exp(-(d**2).sum(2) / (2 * WID**2))
    return phi, -d / WID**2 * phi[:, :, None]                       # values (n,K), gradients (n,K,2)


def sample(density, n, rng):
    k = rng.choice(t.N * t.N, size=n, p=density.ravel() / density.sum())
    return np.column_stack([t.X.ravel()[k], t.Y.ravel()[k]]) + rng.uniform(-t.h / 2, t.h / 2, (n, 2))


def fit(P_eq, P_ts, kinetic):
    P = np.vstack([P_eq, P_ts]) if len(P_ts) else P_eq
    _, G = feats(P); Fref = torch.tensor(F_true(P)); G = torch.tensor(G)
    theta = torch.zeros(len(CEN), requires_grad=True)
    if kinetic:
        phi_ts = torch.tensor(feats(P_ts)[0]); V_ts = torch.tensor(V_true(P_ts))
        basin = P_eq[:60]; phi_b = torch.tensor(feats(basin)[0]); V_b = torch.tensor(V_true(basin))
    opt = torch.optim.LBFGS([theta], max_iter=400, line_search_fn="strong_wolfe", tolerance_grad=1e-9)

    def closure():
        opt.zero_grad()
        Fm = -torch.einsum("nkd,k->nd", G, theta)
        loss = ((Fm - Fref) ** 2).mean() + 1e-6 * (theta**2).sum()
        if kinetic:
            d_ts = phi_ts @ theta - V_ts; d_b = phi_b @ theta - V_b
            n = np.log(len(d_ts))
            width = (torch.logsumexp(d_ts, 0) - n) + (torch.logsumexp(-d_ts, 0) - n)
            centre = d_ts.mean() - d_b.mean()
            loss = loss + LAM * (width + centre**2)
        loss.backward(); return loss
    opt.step(closure)
    return theta.detach().numpy()


if __name__ == "__main__":
    V = t.true_potential(); lam, _, mu, _ = t.slow_mode(V); lam = abs(lam)
    inA = ((V < 2.0) & (t.X < 0)).ravel(); inB = ((V < 2.0) & (t.X > 0)).ravel()
    c = tb.network(V); _, q = tb.capacity(c, inA, inB)
    nu_e = c * (q[tb.EI] - q[tb.EJ]) ** 2
    nu = (np.bincount(tb.EI, nu_e / 2, t.N**2) + np.bincount(tb.EJ, nu_e / 2, t.N**2)).reshape(t.N, t.N)
    grid_pts = np.column_stack([t.X.ravel(), t.Y.ravel()]); phi_grid = feats(grid_pts)[0]
    res = {k: [] for k in ("FM", "FM+TS", "FM+TS+kin")}
    for seed in range(NSEED):
        rng = np.random.default_rng(seed)
        P_all = sample(mu, N_TOT, rng); P_ts = sample(nu, M_TS, rng); P_test = sample(mu, 2000, rng)
        runs = {"FM": (P_all, P_ts[:0], False), "FM+TS": (P_all[:N_TOT - M_TS], P_ts, False), "FM+TS+kin": (P_all[:N_TOT - M_TS], P_ts, True)}
        for name, (pe, pt, kin) in runs.items():
            th = fit(pe, pt, kin)
            Vm = (phi_grid @ th).reshape(t.N, t.N); Vm = np.where(V > 25, V, Vm)            # outside the sampled region use the true wall
            lam_m = abs(t.slow_mode(Vm - Vm.min())[0])
            Fm = -np.einsum("nkd,k->nd", feats(P_test)[1], th)
            rmse = np.sqrt(((Fm - F_true(P_test)) ** 2).mean())
            d_ts = (phi_grid @ th - V.ravel()); w = nu.ravel() / nu.sum()
            width = np.log((w * np.exp(d_ts - (w * d_ts).sum())).sum()) + np.log((w * np.exp(-(d_ts - (w * d_ts).sum()))).sum())
            res[name].append((abs(np.log(lam_m / lam)), rmse, width))
        print(f"seed {seed}: " + "  ".join(f"{k}: |ln k ratio|={v[-1][0]:.3f} rmse={v[-1][1]:.3f}" for k, v in res.items()), flush=True)
    print(f"\nlabel budget {N_TOT} (of which {M_TS} on the leverage density), {len(CEN)} RBFs, {NSEED} seeds")
    print("student      | median |ln(k_hat/k)| | worst | median force RMSE (equilibrium test) | median bracket width on the true nu")
    rows = []
    for k, v in res.items():
        a = np.array(v); rows.append(np.r_[np.median(a[:, 0]), a[:, 0].max(), np.median(a[:, 1]), np.median(a[:, 2])])
        print(f"{k:12s} |      {rows[-1][0]:.3f}          | {rows[-1][1]:.3f} |            {rows[-1][2]:.3f}                     |   {rows[-1][3]:.3f}")
    np.savetxt("data/kinetic_loss_2d.csv", rows, delimiter=",", header="median_abs_ln_ratio,worst_abs_ln_ratio,median_force_rmse,median_bracket_width", comments="")
    np.savez("data/kinetic_loss_2d_all.npz", **{k.replace("+", "_"): np.array(v) for k, v in res.items()})
