"""
How many labels does it take to learn KINETICS, as a function of where the labels are?

Model of learning: Gaussian-process regression of the potential from n noisy energy labels placed with density rho(x).
Quantity of interest: the first-order kinetic error  K = -beta * int dV(x) [nu(x) - mu_basin(x)] dx   (Prop. 'energy and force forms').
Its posterior variance is exact and closed form:  Var K = a^T Sigma_post a.

Local-averaging heuristic: posterior variance at x ~ s^2 / (n rho(x) ell)  =>  Var K ~ (s^2 ell / n) * int a(x)^2 / rho(x) dx,
minimised by  rho*(x) ~ |a(x)| = |nu - mu_basin|   (Cauchy-Schwarz), with minimum (int |a|)^2.
Equilibrium sampling rho = mu costs int a^2 / mu, which grows like e^{beta B}: an EXPONENTIAL label-efficiency gap.
"""
import numpy as np

x = np.linspace(-1.8, 1.8, 721); h = x[1] - x[0]
ELL, NOISE, PRIOR = 0.12, 0.05, 1.0


def densities(B):
    V = B * (x**2 - 1) ** 2; mu = np.exp(-V); mu /= mu.sum() * h
    nu = np.exp(V) * (np.abs(x) < 0.9); nu /= nu.sum() * h
    return V, mu, nu, nu - mu                                   # symmetric well: basin density = mu


def var_K(design_pdf, a, n, rng, reps=12):
    out = []
    cdf = np.cumsum(design_pdf); cdf /= cdf[-1]
    for _ in range(reps):
        xs = np.interp(rng.random(n), cdf, x)
        Kss = PRIOR * np.exp(-(xs[:, None] - xs[None, :]) ** 2 / (2 * ELL**2)) + NOISE**2 * np.eye(n)
        Kxs = PRIOR * np.exp(-(x[:, None] - xs[None, :]) ** 2 / (2 * ELL**2))
        prior = PRIOR * np.exp(-(x[:, None] - x[None, :]) ** 2 / (2 * ELL**2))
        post = prior - Kxs @ np.linalg.solve(Kss, Kxs.T)
        out.append((a * h) @ post @ (a * h))
    return np.mean(out)


if __name__ == "__main__":
    rng = np.random.default_rng(0); rows = []
    print("posterior sd of the first-order kinetic error ln(k_hat/k), n = 200 labels, by sampling design")
    print("  B/kT | equilibrium mu | uncertainty-like (uniform) | leverage nu | optimal |nu - mu| | 50/50 mu + nu || heuristic gap  int a^2/mu / (int|a|)^2")
    for B in (2, 4, 6, 8, 10):
        V, mu, nu, a = densities(B)
        designs = {"mu": mu, "uniform": np.ones_like(x), "nu": nu, "opt": np.abs(a), "mix": 0.5 * mu + 0.5 * nu}
        sd = {k: np.sqrt(var_K(p, a, 200, rng)) for k, p in designs.items()}
        gap = (a**2 / mu).sum() * h / ((np.abs(a)).sum() * h) ** 2
        rows.append([B, *sd.values(), gap])
        print(f"  {B:4d} |    {sd['mu']:.4f}      |        {sd['uniform']:.4f}              |   {sd['nu']:.4f}    |     {sd['opt']:.4f}     |    {sd['mix']:.4f}      ||  {gap:.3g}", flush=True)
    print("\nlabels needed to reach sd(ln k) = 0.1 at B = 8 kT:")
    V, mu, nu, a = densities(8)
    for name, p in (("equilibrium mu", mu), ("uniform", np.ones_like(x)), ("optimal |nu - mu|", np.abs(a)), ("50/50 mu + nu", 0.5 * mu + 0.5 * nu)):
        for n in (25, 50, 100, 200, 400, 800, 1600):
            s = np.sqrt(var_K(p, a, n, rng, reps=6))
            if s < 0.1:
                print(f"   {name:20s}: about {n}"); rows.append([-1, n, 0, 0, 0, 0, 0]); break
        else:
            print(f"   {name:20s}: more than 1600 (sd at 1600 = {s:.3f})"); rows.append([-1, 9999, s, 0, 0, 0, 0])
    np.savetxt("data/optimal_design_1d.csv", rows, delimiter=",", header="B,sd_mu,sd_uniform,sd_nu,sd_opt,sd_mix,heuristic_gap", comments="")
