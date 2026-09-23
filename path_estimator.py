"""
A dimension-free estimator of leverage averages.

Claim.  For overdamped reversible dynamics, with q the committor and the expectation taken over REACTIVE trajectory segments (A -> B),

        < g >_nu  =  E_reactive [  int g(X_t) o dq(X_t)  ]         (Stratonovich integral of g against committor increments)

because the net crossing density of the level set {q = s} by reactive paths is the reactive flux  ~ mu |grad q|, and
nu(dx) = |grad q|^2 mu dx = (mu |grad q| dsigma_s) ds  by the co-area formula.  Consequences: <e^{+-beta dV}>_nu needs only
  (i) reactive paths of the reference dynamics, (ii) committor VALUES along them, (iii) the energy error dV on those frames
-- no gradient of q in 3N dimensions, no grid.

Test: 2D two-channel potential, brute-force Euler-Maruyama, compare with the grid (resistor-network) value.
"""
import numpy as np
from numba import njit
import twochannel_blindspot as t
import tube_bounds as tb

N, h, x0 = t.N, t.h, t.xs[0]


@njit(cache=True)
def interp(F, x, y):
    fx = (x - x0) / h; fy = (y - x0) / h
    i = int(np.floor(fx)); j = int(np.floor(fy))
    if i < 0: i = 0
    if j < 0: j = 0
    if i > N - 2: i = N - 2
    if j > N - 2: j = N - 2
    a = fx - i; b = fy - j
    return (1 - a) * (1 - b) * F[i, j] + a * (1 - b) * F[i + 1, j] + (1 - a) * b * F[i, j + 1] + a * b * F[i + 1, j + 1]


@njit(cache=True)
def simulate(V, Q, D1, D2, core, nsteps, dt, seed):
    """returns per-reactive-path sums of [dq, e^{+dV1} dq, e^{-dV1} dq, e^{+dV2} dq, e^{-dV2} dq] and the path direction."""
    np.random.seed(seed)
    x, y = -1.0, 0.0; s = np.sqrt(2 * dt); e = 1e-4
    out = np.zeros((200000, 6)); n = 0
    acc = np.zeros(5); last = 0                       # last = core last visited (1 = A, 2 = B)
    for k in range(nsteps):
        gx = (interp(V, x + e, y) - interp(V, x - e, y)) / (2 * e); gy = (interp(V, x, y + e) - interp(V, x, y - e)) / (2 * e)
        xn = x - gx * dt + s * np.random.randn(); yn = y - gy * dt + s * np.random.randn()
        c = interp(core, xn, yn)
        now = 1 if c < -0.5 else (2 if c > 0.5 else 0)
        dq = interp(Q, xn, yn) - interp(Q, x, y); xm = 0.5 * (x + xn); ym = 0.5 * (y + yn)
        d1 = interp(D1, xm, ym); d2 = interp(D2, xm, ym)
        acc[0] += dq; acc[1] += np.exp(d1) * dq; acc[2] += np.exp(-d1) * dq; acc[3] += np.exp(d2) * dq; acc[4] += np.exp(-d2) * dq
        if now != 0:
            if last != 0 and now != last and n < out.shape[0]:            # a reactive segment just ended
                out[n, :5] = acc; out[n, 5] = now; n += 1
            acc[:] = 0.0; last = now                                     # excursion bookkeeping restarts inside a core
        x, y = xn, yn
    return out[:n]


if __name__ == "__main__":
    V = t.true_potential()
    inA = ((V < 2.0) & (t.X < 0)); inB = ((V < 2.0) & (t.X > 0))
    c0 = tb.network(V); C, q = tb.capacity(c0, inA.ravel(), inB.ravel()); Q = q.reshape(N, N)
    core = np.where(inA, -1.0, np.where(inB, 1.0, 0.0))
    nu = c0 * (q[tb.EI] - q[tb.EJ]) ** 2; nu /= nu.sum()
    fields = []
    for block in (3.0, 6.0):
        t.BLOCK = block; t.rng = np.random.default_rng(1); fields.append(t.initial_error())
    grid = []
    for d in fields:
        de = 0.5 * (d.ravel()[tb.EI] + d.ravel()[tb.EJ]); grid += [(nu * np.exp(de)).sum(), (nu * np.exp(-de)).sum()]
    paths = np.concatenate([simulate(np.minimum(V, 40.0), Q, fields[0], fields[1], core, 120_000_000, 5e-4, seed) for seed in (1, 2, 3)])
    sgn = np.where(paths[:, 5] == 2, 1.0, -1.0)[:, None]                 # B -> A paths run the committor backwards
    P = paths[:, :5] * sgn
    print(f"{len(P)} reactive paths;  mean of int dq over a reactive path = {P[:,0].mean():.4f} (should be ~1: cores are where q is 0 / 1)")
    names = ["<e^{+dV}>_nu, block 3", "<e^{-dV}>_nu, block 3", "<e^{+dV}>_nu, block 6", "<e^{-dV}>_nu, block 6"]
    rows = []
    for k, nm in enumerate(names):
        v = P[:, k + 1] / P[:, 0].mean(); est = v.mean(); se = v.std() / np.sqrt(len(v))
        print(f"   {nm:24s} grid {grid[k]:9.4f}   path estimator {est:9.4f} +- {se:.4f}   (ln: {np.log(grid[k]):+.3f} vs {np.log(est):+.3f})")
        rows.append((grid[k], est, se))
    for m in (50, 200, 1000):
        sub = P[:m]; print(f"   with only {m:4d} reactive paths: ln<e^(+dV)>_nu (block 6) = {np.log(sub[:,3].mean()/sub[:,0].mean()):+.3f}   [grid {np.log(grid[2]):+.3f}]")
    np.savetxt("data/path_estimator_2d.csv", rows, delimiter=",", header="grid_value,path_estimate,standard_error", comments="")
