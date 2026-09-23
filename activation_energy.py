"""
Activation energies from the leverage density (a Tolman-type theorem for diffusive dynamics).

Envelope theorem on the variational principles (the optimiser does not contribute to the beta-derivative):
    diffusion :  E_a^D  := -d ln D / d beta          =  <V>_nu - <V>_mu                         (exact, any dimension, D0 fixed)
    rate      :  E_a^k  := -d ln kappa / d beta      ~  <V>_nu - <V>_A - <V>_B + <V>_mu         (two-state regime, D0 fixed)
so the activation-energy error of a model is  dE_a = [<V_hat>_nu_hat - <V_hat>_mu_hat] - [<V>_nu - <V>_mu].
"""
import numpy as np
import diffusion_bounds as db
import twochannel_blindspot as t, tube_bounds as tb

def lev_means(V, beta):
    D, c, g, mu = db.solve_cell(beta * V)
    nu = c * g**2; nu /= nu.sum(); Ve = 0.5 * (V.ravel()[db.EI] + V.ravel()[db.EJ])
    return D, (nu * Ve).sum() - (mu * V.ravel()).sum()

print("DIFFUSION (2D periodic, no dominant saddle):  beta | D/D0 | E_a from finite differences of ln D | <V>_nu - <V>_mu")
V = 3.0 * np.cos(2*np.pi*db.X) * np.cos(2*np.pi*db.Y) + 1.5 * np.cos(4*np.pi*db.X) + 1.0 * np.sin(2*np.pi*(db.X + db.Y))
rows = []
for beta in (0.5, 1.0, 2.0, 3.0):
    e = 1e-3; Dp, _ = lev_means(V, beta + e); Dm, _ = lev_means(V, beta - e); D, ea = lev_means(V, beta)
    fd = -(np.log(Dp) - np.log(Dm)) / (2 * e); rows.append((beta, D, fd, ea)); print(f"   {beta:4.1f} | {D:.4f} | {fd:+.5f} | {ea:+.5f}")
rng = np.random.default_rng(0)
print("\nmodel error in the activation energy (beta = 2):  exact dE_a (finite differences) | leverage formula | naive 'barrier-height error' max dV on saddle region")
for k in range(5):
    dV = db.random_field(rng, 1.0)
    e = 1e-3; f = lambda b, U: np.log(lev_means(U, b)[0])
    ea_hat = -(f(2 + e, V + dV) - f(2 - e, V + dV)) / (2 * e); ea = -(f(2 + e, V) - f(2 - e, V)) / (2 * e)
    lev = lev_means(V + dV, 2.0)[1] - lev_means(V, 2.0)[1]
    print(f"   case {k}: {ea_hat - ea:+.5f} | {lev:+.5f}")
    rows.append((2.0, np.nan, ea_hat - ea, lev))
np.savetxt("data/activation_energy_diffusion.csv", rows, delimiter=",", header="beta,D_over_D0,Ea_finite_difference,Ea_leverage_formula", comments="")

print("\nRATE (2D two-channel):  beta | E_a from finite differences of ln kappa | <V>_nu - <V>_A - <V>_B + <V>_mu")
Vr = t.true_potential()
def rate_side(beta):
    lam, _, pi, _ = t.slow_mode(beta * Vr); inA = ((Vr < 2.0 / 1) & (t.X < 0)).ravel(); inB = ((Vr < 2.0) & (t.X > 0)).ravel()
    c = tb.network(beta * Vr); _, q = tb.capacity(c, inA, inB); nu = c * (q[tb.EI] - q[tb.EJ]) ** 2; nu /= nu.sum()
    Ve = 0.5 * (Vr.ravel()[tb.EI] + Vr.ravel()[tb.EJ]); p = pi.ravel(); A = (t.X < 0).ravel()
    return abs(lam), (nu * Ve).sum() - (p[A] * Vr.ravel()[A]).sum() / p[A].sum() - (p[~A] * Vr.ravel()[~A]).sum() / p[~A].sum() + (p * Vr.ravel()).sum()
for beta in (0.8, 1.0, 1.5):
    e = 2e-3; fd = -(np.log(rate_side(beta + e)[0]) - np.log(rate_side(beta - e)[0])) / (2 * e)
    print(f"   {beta:4.1f} | {fd:+.4f} | {rate_side(beta)[1]:+.4f}")
