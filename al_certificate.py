"""
Certificate-driven active learning in the two-channel model, under an IMPERFECT uncertainty model.

The agent never sees the true error dV. It holds an ensemble of K plausible error fields:
  * it knows WHERE errors may sit (the true error regions + 3 phantom regions with no error at all),
  * amplitudes are miscalibrated (log-normal, sigma = 0.5) and their SIGN is unknown (each member draws N(0,1) x nominal),
  * the smooth background is re-drawn for every member.
A query at q reveals the truth there: the true dV and every member are multiplied by (1 - g_q).

Strategies (free candidate pool: model energy < 16 kT, stride-4 lattice):
  unc       ensemble force variance                      E_k |dF_k|^2
  lev       the original proposal                        grad(phi_hat)^T Sigma_F grad(phi_hat) = E_k (dF_k . grad phi_hat)^2
  cert      expected certified kinetic error AFTER the query, E_k U_k(q), minimised
            U = max(|lower|, |upper|) of the model-side bracket for ln(k/k_hat):
            lower = Thomson with 16 re-weighted flux tubes (parallel law), upper = Dirichlet with the model committor, both times basin factor G
  cert_orc  same score evaluated with the true dV (upper limit of what the score can do)
"""
import numpy as np
import twochannel_blindspot as t
import tube_bounds as tb

K, ROUNDS, SEEDS, NT = 8, 12, 6, 16
REGIONS = [(0, -1, .3), (0, 1, .3), (1.45, 0, .25), (-1.45, 0, .25), (0, 0, .25)]          # true error regions
PHANTOM = [(0.55, 0.8, .25), (-0.6, -0.8, .25), (-1.0, 0.3, .25)]                          # uncertain but actually fine


def make_ensemble(rng, block):
    nominal = [block, 1.0, 6.0, 6.0, 6.0]
    members = []
    for _ in range(K):
        f = np.zeros_like(t.X)
        for (cx, cy, w), a in zip(REGIONS, nominal):
            f += a * rng.lognormal(0, .5) * rng.normal() * t.gauss(cx, cy, w)
        for cx, cy, w in PHANTOM:
            f += 4.0 * rng.normal() * t.gauss(cx, cy, w)
        bg = sum(rng.normal() * np.cos(rng.uniform(1, 4) * t.X + rng.uniform(0, 6)) * np.cos(rng.uniform(1, 4) * t.Y + rng.uniform(0, 6)) for _ in range(6))
        members.append(f + 0.15 * bg)
    return members


def model_geometry(Vhat):
    E = Vhat - Vhat.min()
    inA = ((E < 2.0) & (t.X < 0)).ravel(); inB = ((E < 2.0) & (t.X > 0)).ravel()
    c = tb.network(Vhat); _, q = tb.capacity(c, inA, inB)
    nu = c * (q[tb.EI] - q[tb.EJ]) ** 2; nu /= nu.sum()
    Jx = np.zeros((t.N, t.N)); ne = len(tb.ex_i)
    Jx[:-1, :] = (c[:ne] * (q[tb.ex_j] - q[tb.ex_i])).reshape(t.N - 1, t.N)
    psi = np.cumsum(Jx, axis=1); psi /= psi[:, -1].max()
    tube = np.clip((psi.ravel()[tb.EI] * NT).astype(int), 0, NT - 1)
    act = nu > 1e-13 * nu.max()
    T = np.zeros((act.sum(), NT)); T[np.arange(act.sum()), tube[act]] = nu[act]
    pi = np.exp(-E).ravel(); pi /= pi.sum()
    basin = pi > 1e-7 * pi.max()
    return dict(act=act, nu=nu[act], T=T, w=T.sum(0), pi=pi, basin=basin, A=(t.X.ravel() < 0), E=E)


def certified_error(geo, dV_fields, cand, sym=False):
    """U for every (field, candidate): returns array (n_fields, n_cand). cand = None -> no query (n_cand = 1)."""
    Kx = np.exp(-(t.xs[:, None] - t.xs[None, :]) ** 2 / (2 * t.RHO**2))
    out = []
    eI, eJ = tb.EI[geo["act"]], tb.EJ[geo["act"]]
    bn = np.where(geo["basin"])[0]; pA = geo["pi"][bn] * geo["A"][bn]; pB = geo["pi"][bn] * (~geo["A"][bn])
    muA, muB = pA.sum(), pB.sum()
    for dV in dV_fields:
        f = dV.ravel(); res = []; res_neg = []
        chunks = [None] if cand is None else [cand[i:i + 250] for i in range(0, len(cand), 250)]
        for ch in chunks:
            if ch is None:
                rem_n = np.ones((1, t.N * t.N))
            else:
                rem_n = 1 - (Kx[ch[:, 0]][:, :, None] * Kx[ch[:, 1]][:, None, :]).reshape(len(ch), -1)
            De = 0.5 * (f[eI] * rem_n[:, eI] + f[eJ] * rem_n[:, eJ])
            Ep = np.exp(np.clip(De, -60, 60))
            up = np.log(Ep @ geo["nu"])
            m = (1 / Ep) @ geo["T"]
            lo = np.log((geo["w"] ** 2 / np.maximum(m, 1e-300)).sum(1))
            Eb = np.exp(np.clip(f[bn] * rem_n[:, bn], -60, 60))
            aA, aB = (Eb @ pA) / muA, (Eb @ pB) / muB
            lnG = np.log(muA / aB + muB / aA) - np.log(muA + muB)
            res.append(np.maximum(np.abs(lo + lnG), np.abs(up + lnG)))
            if sym:                                            # the same member with the opposite sign: swap exp(x) <-> exp(-x)
                up2 = np.log((1 / Ep) @ geo["nu"]); lo2 = np.log((geo["w"] ** 2 / np.maximum(Ep @ geo["T"], 1e-300)).sum(1))
                aA2, aB2 = ((1 / Eb) @ pA) / muA, ((1 / Eb) @ pB) / muB
                lnG2 = np.log(muA / aB2 + muB / aA2) - np.log(muA + muB)
                res_neg.append(np.maximum(np.abs(lo2 + lnG2), np.abs(up2 + lnG2)))
        out.append(np.concatenate(res))
        if sym:
            out.append(np.concatenate(res_neg))
    return np.array(out)


def run(V, dV_true, members, lam, strategy):
    dV = dV_true.copy(); members = [m.copy() for m in members]
    err, cert = [], []
    for r in range(ROUNDS + 1):
        Vhat = V + dV
        lam_hat, phi, pi, _ = t.slow_mode(Vhat)
        err.append(abs(np.log(abs(lam_hat) / lam)))
        geo = model_geometry(Vhat)
        cert.append(certified_error(geo, [dV], None)[0, 0])
        if r == ROUNDS:
            break
        cand = np.argwhere(geo["E"] < t.E_CUT); cand = cand[(cand[:, 0] % 4 == 0) & (cand[:, 1] % 4 == 0)]
        if strategy in ("unc", "lev"):
            gx, gy = np.gradient(phi, t.h); s = np.zeros_like(dV)
            for m in members:
                fx, fy = np.gradient(-m, t.h)
                s += (fx**2 + fy**2) if strategy == "unc" else (fx * gx + fy * gy) ** 2
            score = -s[cand[:, 0], cand[:, 1]]
        else:
            fields = [dV] if strategy == "cert_orc" else members
            U = certified_error(geo, fields, cand, sym=strategy in ("cert_sym", "cert_symmax"))
            score = U.max(0) if strategy in ("cert_max", "cert_symmax") else U.mean(0)
        i, j = cand[np.argmin(score)]
        rem = 1 - t.gauss(t.xs[i], t.xs[j], t.RHO)
        dV = dV * rem; members = [m * rem for m in members]
    return np.array(err), np.array(cert)


if __name__ == "__main__":
    V = t.true_potential(); lam = abs(t.slow_mode(V)[0])
    strategies = ["unc", "lev", "cert", "cert_orc"]
    for block in (6.0, 12.0):
        res = {s: [] for s in strategies}; certs = []
        for seed in range(SEEDS):
            t.BLOCK = block; t.rng = np.random.default_rng(200 + seed); dV0 = t.initial_error()
            members = make_ensemble(np.random.default_rng(300 + seed), block)
            for s in strategies:
                e, c = run(V, dV0, members, lam, s); res[s].append(e)
                if s == "cert_orc":
                    certs.append(np.column_stack([e, c]))
        print(f"\nblock = {block:.0f} kT   median |ln(k_hat/k)| over {SEEDS} seeds, imperfect ensemble uncertainty (K={K})")
        print("   strategy    " + " ".join(f"q{k:<4d}" for k in (0, 1, 2, 3, 4, 6, 8, 12)) + " | worst seed at q12")
        for s in strategies:
            m = np.median(res[s], axis=0)
            print(f"   {s:9s}  " + " ".join(f"{m[k]:5.2f}" for k in (0, 1, 2, 3, 4, 6, 8, 12)) + f" | {np.max(np.array(res[s])[:, -1]):.2f}")
        C = np.array(certs)
        print(f"   certificate validity along the cert_orc runs: actual <= certified in {np.mean(C[:,:,0] <= C[:,:,1] + 1e-9)*100:.0f}% of (seed, round) pairs;"
              f" median certified/actual = {np.median(C[:,:,1] / np.maximum(C[:,:,0], 1e-3)):.1f}")
        np.savetxt(f"data/al_cert_block{block:.0f}.csv", np.column_stack([np.arange(ROUNDS + 1)] + [np.median(res[s], axis=0) for s in strategies]),
                   delimiter=",", header="round," + ",".join(strategies), comments="")
        np.savez(f"data/al_cert_block{block:.0f}_all.npz", certs=C, **{s: np.array(v) for s, v in res.items()})
