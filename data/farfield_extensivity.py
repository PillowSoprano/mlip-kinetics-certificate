"""
Extensivity of the bracket.  Model error = local part on the reaction coordinate + far-field part on degrees of freedom that do not couple to the hop:

    dV(x, y) = d_loc(x) + eta(y),    eta = stationary Gaussian process, variance s^2 (grows like the number of atoms), autocorrelation exp(-t/tau).

x: overdamped double well V = B (x^2-1)^2, beta = 1.  Because y does not couple to x, the TRUE rate ratio depends on d_loc only (1D: equals the lower bound):
    truth = -ln < e^{d_loc} >_nu .
Compared (expectation and scatter over repeated experiments with NP paths each):
    naive upper   ln<e^{-dV}>_nu - ln<e^{-dV}>_eq          theory: upper_loc                      (far field cancels)
    naive lower  -ln<e^{+dV}>_nu - ln<e^{-dV}>_eq          theory: truth - s^2                     (extensive loosening)
    paired upper  ln<e^{-(dV - dV_pre)}>_nu                theory: ln< e^{-d_loc + s^2 (1 - rho_t)} >_nu
    paired lower -ln<e^{+(dV - dV_pre)}>_nu                theory: -ln< e^{+d_loc + s^2 (1 - rho_t)} >_nu,   rho_t = exp(-(lag + time since leaving the basin) / tau)
dV_pre = error on a basin frame of the SAME path, a time `lag` before the path leaves the basin.
"""
import numpy as np, time
rng = np.random.default_rng(1); B, DT, STRIDE = 6.0, 1e-3, 5
V = lambda x: B * (x * x - 1) ** 2; F = lambda x: -4 * B * x * (x * x - 1)
dloc = lambda x: 1.5 * np.exp(-x * x / (2 * 0.3 ** 2))
xg = np.linspace(-0.8, 0.8, 4001); w = np.exp(V(xg)); qg = np.cumsum(w); qg = (qg - qg[0]) / (qg[-1] - qg[0]); nu = w / w.sum()
truth = -np.log((nu * np.exp(dloc(xg))).sum()); upper_loc = np.log((nu * np.exp(-dloc(xg))).sum())
qf = lambda x: np.interp(np.clip(x, -0.8, 0.8), xg, qg)

# ---- reactive paths of x (once)
NW, NS = 300, 400000; x = -np.ones(NW); X = np.empty((NS // STRIDE, NW), np.float32); t0 = time.time()
for k in range(NS):
    x += F(x) * DT + np.sqrt(2 * DT) * rng.standard_normal(NW)
    if k % STRIDE == 0: X[k // STRIDE] = x
    if k % 100000 == 0: print(f"  x dynamics {k}/{NS}  {time.time()-t0:.0f}s", flush=True)
paths = []                                         # (x values along reactive segment, x values of the preceding basin stretch)
LAGMAX = 4000 // STRIDE
for j in range(NW):
    xs = X[:, j]; st = np.where(xs < -0.8, 0, np.where(xs > 0.8, 1, -1)); idx = np.nonzero(st >= 0)[0]; ch = np.nonzero(st[idx][1:] != st[idx][:-1])[0]
    for c in ch:
        i0, i1 = idx[c], idx[c + 1]
        if i0 < LAGMAX: continue
        seg = xs[i0:i1 + 1].astype(float); sgn = 1.0 if st[i1] == 1 else -1.0; paths.append((sgn * seg, i1 - i0))
print(f"{len(paths)} reactive paths, median duration {np.median([p[1] for p in paths]) * STRIDE * DT:.3f}", flush=True)
dth = STRIDE * DT

def ou(n, tau, s, x0=None):
    """stationary OU sample of length n on the frame grid"""
    a = np.exp(-dth / tau); e = np.empty(n); e[0] = s * rng.standard_normal() if x0 is None else x0
    z = rng.standard_normal(n) * s * np.sqrt(1 - a * a)
    for k in range(1, n): e[k] = a * e[k - 1] + z[k]
    return e

def experiment(s, tau, lag, NP=200):
    sel = rng.choice(len(paths), NP, replace=False); nU = dU = nL = 0.0; pu = pl = 0.0; wsum = 0.0
    for p in sel:
        seg, n = paths[p]; q = qf(seg); dq = 0.5 * (np.r_[q[1:], q[-1]] - np.r_[q[0], q[:-1]])
        pre = s * rng.standard_normal(); rho = np.exp(-lag / tau); e0 = rho * pre + s * np.sqrt(1 - rho * rho) * rng.standard_normal()      # eta at basin frame, then at path start
        eta = ou(len(seg), tau, s, x0=e0); d = dloc(seg) + eta
        nU += (np.exp(-d) * dq).sum(); nL += (np.exp(d) * dq).sum(); pu += (np.exp(-(d - pre)) * dq).sum(); pl += (np.exp(d - pre) * dq).sum(); wsum += dq.sum()
    eq = s * rng.standard_normal(NP * 5); a = np.exp(-eq).mean()                                                                           # basin frames: d_loc ~ 0 there
    return np.log(nU / wsum) - np.log(a), -np.log(nL / wsum) - np.log(a), np.log(pu / wsum), -np.log(pl / wsum)

def theory_paired(s, tau, lag):
    nu_ = nl_ = ws = 0.0
    for seg, n in paths:
        q = qf(seg); dq = 0.5 * (np.r_[q[1:], q[-1]] - np.r_[q[0], q[:-1]]); g = s * s * (1 - np.exp(-(lag + np.arange(len(seg)) * dth) / tau))
        nu_ += (np.exp(-dloc(seg) + g) * dq).sum(); nl_ += (np.exp(dloc(seg) + g) * dq).sum(); ws += dq.sum()
    return np.log(nu_ / ws), -np.log(nl_ / ws)

print(f"truth {truth:+.3f}   local upper bound {upper_loc:+.3f}   (1D: truth = local lower bound)")
rows = []; print("  s    tau   lag |  naive upper      naive lower      | paired upper     paired lower     | theory: naive lower, paired upper/lower")
for s in (0.5, 1.0, 1.5, 2.0):
    for tau, lag in ((1.0, 0.1), (0.1, 0.1)):
        R = np.array([experiment(s, tau, lag) for _ in range(40)]); m, sd = R.mean(0), R.std(0); rho = np.exp(-lag / tau)
        thU, thL = theory_paired(s, tau, lag)
        print(f" {s:3.1f}  {tau:4.1f}  {lag:3.1f} | {m[0]:+.2f} +- {sd[0]:.2f}   {m[1]:+.2f} +- {sd[1]:.2f}   | {m[2]:+.2f} +- {sd[2]:.2f}   {m[3]:+.2f} +- {sd[3]:.2f}   | {truth - s*s:+.2f}, {thU:+.2f} / {thL:+.2f}", flush=True)
        rows.append((s, tau, lag, *m, *sd, truth, upper_loc, truth - s * s, thU, thL))
np.savetxt("farfield_extensivity.csv", rows, delimiter=",", comments="", header="s,tau,lag,naive_up,naive_lo,paired_up,paired_lo,sd_naive_up,sd_naive_lo,sd_paired_up,sd_paired_lo,truth,upper_loc,theory_naive_lo,theory_paired_up,theory_paired_lo")
