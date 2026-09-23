"""
Acquisition by the expected change of the slow rate caused by ONE query, in place of pointwise scores.

A query at q repairs the model:  V_hat -> V_hat - g_q * dV   (g_q = Gaussian repair footprint).
    int1  : first-order,  |d lambda| = | sum  g_q dV (|grad phi_hat|^2 - beta |lam_hat| phi_hat^2) mu_hat |
    intRQ : nonlinear,    | ln( RQ_q / |lam_hat| ) |, RQ_q = Rayleigh quotient of the model's own phi_hat under the
            repaired measure mu_hat * exp(+beta g_q dV)   (keeps the exp(beta*Delta) amplification that first order loses)
Candidate pools:  free = model energy < E_CUT (16 kT);   md = model energy < MD_CUT (8 kT, what plain MLIP-MD can reach)
As before dV is used as an oracle-calibrated uncertainty (best case for every strategy alike).
"""
import numpy as np
import twochannel_blindspot as t

MD_CUT = 8.0
t.ROUNDS = 14


def integrated_scores(V_true, dV):
    lam_hat, phi, pi, _ = t.slow_mode(V_true + dV)
    gx, gy = np.gradient(phi, t.h)
    g2 = gx**2 + gy**2
    E = V_true + dV; E = E - E.min()
    out = {}
    # ---- first order: separable Gaussian convolution of the signed sensitivity density
    D = dV * (g2 - abs(lam_hat) * phi**2) * pi
    K = np.exp(-(t.xs[:, None] - t.xs[None, :]) ** 2 / (2 * t.RHO**2))
    int1 = np.abs(K @ D @ K.T)
    # ---- Rayleigh quotient under the repaired measure, candidates on a stride-2 grid
    cand = np.argwhere((E < t.E_CUT))
    cand = cand[(cand[:, 0] % 2 == 0) & (cand[:, 1] % 2 == 0)]
    rq = np.zeros_like(E)
    pif, g2f, phif, dVf = pi.ravel(), g2.ravel(), phi.ravel(), dV.ravel()
    for c0 in range(0, len(cand), 400):
        c = cand[c0:c0 + 400]
        G = (K[c[:, 0]][:, :, None] * K[c[:, 1]][:, None, :]).reshape(len(c), -1)      # g_q on the grid
        W = pif * np.exp(np.clip(G * dVf, -50, 50)); W /= W.sum(1, keepdims=True)
        m1 = W @ phif
        var = W @ phif**2 - m1**2
        rq[c[:, 0], c[:, 1]] = np.abs(np.log((W @ g2f) / var / abs(lam_hat)))
    for pool, cut in (("free", t.E_CUT), ("md", MD_CUT)):
        out[f"int1_{pool}"] = np.where(E < cut, int1, 0)
        out[f"intRQ_{pool}"] = np.where(E < cut, rq, 0)
    return lam_hat, out


def run(V, dV0, lam, strategy):
    dV = dV0.copy(); hist = []; picks = []
    for r in range(t.ROUNDS + 1):
        if strategy.startswith("int"):
            lam_hat, sc = integrated_scores(V, dV)
        else:
            lam_hat, sc, _, _ = t.scores(V, dV)
        hist.append(abs(np.log(lam_hat / lam)))
        if r == t.ROUNDS:
            break
        i, j = np.unravel_index(np.argmax(sc[strategy]), (t.N, t.N))
        picks.append((t.xs[i], t.xs[j]))
        dV = dV * (1 - t.gauss(t.xs[i], t.xs[j], t.RHO))
    return np.array(hist), np.array(picks)


if __name__ == "__main__":
    V = t.true_potential(); lam = t.slow_mode(V)[0]
    strategies = ["unc_md", "unc_free", "lev_md", "lev_free", "int1_md", "int1_free", "intRQ_md", "intRQ_free"]

    print("(1) seed scan, block = 6 kT: median |ln rate error| over 8 seeds")
    allc = {s: [] for s in strategies}
    for seed in range(8):
        t.BLOCK = 6.0; t.rng = np.random.default_rng(100 + seed); dV0 = t.initial_error()
        for s in strategies:
            allc[s].append(run(V, dV0, lam, s)[0])
    med = {s: np.median(allc[s], axis=0) for s in strategies}
    print("   strategy      q0    q1    q2    q3    q6    q10   q14   | worst seed at q14")
    for s in strategies:
        m = med[s]
        print(f"   {s:11s} " + " ".join(f"{m[k]:5.2f}" for k in (0, 1, 2, 3, 6, 10, 14)) + f"   | {np.max(np.array(allc[s])[:, 14]):.2f}")
    np.savetxt("data/twochannel_al_median_curves.csv",
               np.column_stack([np.arange(t.ROUNDS + 1)] + [med[s] for s in strategies]),
               delimiter=",", header="round," + ",".join(strategies), comments="")
    np.savez("data/twochannel_al_allseeds.npz", **{s: np.array(v) for s, v in allc.items()})

    print("\n(2) blocking scan: first query that lands in the blocked lower channel (0 = never), and error after 6 queries")
    rows = []
    for block in (3, 6, 9, 12, 15, 18):
        t.BLOCK = float(block); t.rng = np.random.default_rng(1); dV0 = t.initial_error()
        line, row = [], [block]
        for s in strategies:
            c, p = run(V, dV0, lam, s)
            hit = [r + 1 for r, (px, py) in enumerate(p) if abs(px) < 0.6 and py < -0.5]
            row += [hit[0] if hit else 0, c[6]]
            line.append(f"{s}={hit[0] if hit else 0}/{c[6]:.2f}")
            if block == 6:
                np.savetxt(f"data/twochannel_picks_{s}.csv", p, delimiter=",", header="x,y", comments="")
        rows.append(row); print(f"   block={block:2d}kT  " + "  ".join(line))
    np.savetxt("data/twochannel_blockscan_all.csv", rows, delimiter=",", comments="",
               header="block_kT," + ",".join(f"first_{s},err6_{s}" for s in strategies))
