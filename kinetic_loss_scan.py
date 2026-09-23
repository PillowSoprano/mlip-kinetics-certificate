"""
Kinetic loss at fixed label budget, for students of decreasing capacity (paper Table "lkin").

Same problem as kinetic_loss.py (2D two-channel ring, RBF student, 300 labels of which 40 on nu, 8 seeds), with
  - the basis size scanned: k x k Gaussian centres on [-1.8, 1.8]^2, width 1.1 x centre spacing (k = 13 is kinetic_loss.py);
  - the kinetic loss in the form written in the paper, L_kin = w + c^2 with the exact bracket ends
        L = ln G - ln<e^{+dV}>_nu,   U = ln G + ln<e^{-dV}>_nu,   c = (L+U)/2,   w = U - L,
        G = mu(A)/a_B + mu(B)/a_A,   a_X = <e^{-dV}>_{basin X}         (energies in units of kT)
    and, for comparison, the first-order centre (<dV>_nu - <dV>_basins)^2 of kinetic_loss.py;
  - single-threaded torch, so that every number is reproducible bit for bit.
    OMP_NUM_THREADS=1 python kinetic_loss_scan.py
"""
import numpy as np, torch
import kinetic_loss as K
import twochannel_blindspot as t
import tube_bounds as tb

torch.set_num_threads(1)
SIZES = (13, 9, 7, 6, 5)
PROTOCOLS = ("FM", "FM+TS", "FM+TS+kin1", "FM+TS+kin")          # kin1: first-order centre; kin: exact centre (paper)


def set_basis(k):
    cx = np.linspace(-1.8, 1.8, k); CX, CY = np.meshgrid(cx, cx, indexing="ij")
    K.CEN = np.column_stack([CX.ravel(), CY.ravel()]); K.WID = 1.1 * (cx[1] - cx[0])


def lmeanexp(x):
    return torch.logsumexp(x, 0) - np.log(len(x))


def fit(P_eq, P_ts, kinetic):
    P = np.vstack([P_eq, P_ts]) if len(P_ts) else P_eq
    _, G = K.feats(P); Fref = torch.tensor(K.F_true(P)); G = torch.tensor(G)
    theta = torch.zeros(len(K.CEN), requires_grad=True)
    if kinetic:
        phi_ts = torch.tensor(K.feats(P_ts)[0]); V_ts = torch.tensor(K.V_true(P_ts))
        basin = P_eq[:60]; phi_b = torch.tensor(K.feats(basin)[0]); V_b = torch.tensor(K.V_true(basin))
        isA = torch.tensor(basin[:, 0] < 0); mA = isA.double().mean()
    opt = torch.optim.LBFGS([theta], max_iter=400, line_search_fn="strong_wolfe", tolerance_grad=1e-9)

    def closure():
        opt.zero_grad()
        Fm = -torch.einsum("nkd,k->nd", G, theta)
        loss = ((Fm - Fref) ** 2).mean() + 1e-6 * (theta**2).sum()
        if kinetic:
            d_ts = phi_ts @ theta - V_ts; d_b = phi_b @ theta - V_b
            lp, lm = lmeanexp(d_ts), lmeanexp(-d_ts)
            width = lp + lm
            if kinetic == "first-order":
                centre = d_ts.mean() - d_b.mean()
            else:
                la_A, la_B = lmeanexp(-d_b[isA]), lmeanexp(-d_b[~isA])
                lnG = torch.logsumexp(torch.stack([torch.log(mA) - la_B, torch.log(1 - mA) - la_A]), 0)
                centre = lnG + 0.5 * (lm - lp)
            loss = loss + K.LAM * (width + centre**2)
        loss.backward(); return loss
    opt.step(closure)
    return theta.detach().numpy()


if __name__ == "__main__":
    V = t.true_potential(); lam, _, mu, _ = t.slow_mode(V); lam = abs(lam)
    inA = ((V < 2.0) & (t.X < 0)).ravel(); inB = ((V < 2.0) & (t.X > 0)).ravel()
    c = tb.network(V); _, q = tb.capacity(c, inA, inB)
    nu_e = c * (q[tb.EI] - q[tb.EJ]) ** 2
    nu = (np.bincount(tb.EI, nu_e / 2, t.N**2) + np.bincount(tb.EJ, nu_e / 2, t.N**2)).reshape(t.N, t.N)
    grid = np.column_stack([t.X.ravel(), t.Y.ravel()])
    rows, per_seed = [], {}
    for k in SIZES:
        set_basis(k); phi_grid = K.feats(grid)[0]; res = {p: [] for p in PROTOCOLS}
        for seed in range(K.NSEED):
            rng = np.random.default_rng(seed)
            P_all = K.sample(mu, K.N_TOT, rng); P_ts = K.sample(nu, K.M_TS, rng); P_test = K.sample(mu, 2000, rng)
            runs = {"FM": (P_all, P_ts[:0], False), "FM+TS": (P_all[:K.N_TOT - K.M_TS], P_ts, False),
                    "FM+TS+kin1": (P_all[:K.N_TOT - K.M_TS], P_ts, "first-order"), "FM+TS+kin": (P_all[:K.N_TOT - K.M_TS], P_ts, "exact")}
            for name, (pe, pt, kin) in runs.items():
                th = fit(pe, pt, kin)
                Vm = (phi_grid @ th).reshape(t.N, t.N); Vm = np.where(V > 25, V, Vm)
                err = abs(np.log(abs(t.slow_mode(Vm - Vm.min())[0]) / lam))
                rmse = np.sqrt(((-np.einsum("nkd,k->nd", K.feats(P_test)[1], th) - K.F_true(P_test)) ** 2).mean())
                res[name].append((err, rmse))
            print(f"n_rbf {k*k:3d} seed {seed}: " + "  ".join(f"{p} {res[p][-1][0]:.3f}" for p in PROTOCOLS), flush=True)
        for p in PROTOCOLS:
            a = np.array(res[p]); rows.append((k * k, PROTOCOLS.index(p), np.median(a[:, 0]), a[:, 0].max(), np.median(a[:, 1])))
            per_seed[f"n{k*k}_{p.replace('+', '_')}"] = a
    print("\nn_rbf | " + " | ".join(f"{p:>22s}" for p in PROTOCOLS) + "     (median |ln k ratio| / worst / median force RMSE)")
    for k in SIZES:
        r = [x for x in rows if x[0] == k * k]
        print(f"{k*k:5d} | " + " | ".join(f"{x[2]:.3f} / {x[3]:.3f} / {x[4]:.2f}".rjust(22) for x in r))
    np.savetxt("data/kinetic_loss_scan_exact.csv", rows, delimiter=",", comments="",
               header="n_rbf,protocol(0=FM 1=FM+TS 2=FM+TS+kin_first_order 3=FM+TS+kin_exact),median_abs_ln_ratio,worst_abs_ln_ratio,median_force_rmse")
    np.savez("data/kinetic_loss_scan_exact_all.npz", **per_seed)
