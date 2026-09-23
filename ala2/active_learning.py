"""
One active-learning experiment on alanine dipeptide (teacher = Amber14, students = committees of 4 networks).

Round 0: every strategy starts from the same N0 equilibrium frames (no deliberate transition-state data).
Each round adds NADD teacher labels chosen from a fixed candidate pool (equilibrium frames + frames from the barrier window):
    random     uniform over the pool
    unc        largest committee disagreement in the energy                      (standard uncertainty sampling)
    lev        committee disagreement x leverage ratio  nu(phi,psi) / pool density  (bound-driven: the bracket width is ~ beta^2 Var_nu(dV),
               so a label is worth its expected reduction of the leverage-weighted error variance)
After every round the committee is retrained from scratch; member 0 is saved as the student whose dynamics is measured.

usage: python active_learning.py prepare            build the pool (teacher-labelled lazily) and the leverage ratio
       python active_learning.py run STRATEGY SEED   run all rounds for one strategy
"""
import sys, glob, json, time
import numpy as np, torch
import analyze as a
from student import Student, descriptor, teacher_context, KJNM_TO_MEVA
from run_md_common import dihedral

N0, NADD, ROUNDS, NCOMM, HIDDEN, STEPS = 1500, 250, 3, 4, 96, 4000
PHI = [4, 6, 8, 14]; PSI = [6, 8, 14, 16]
torch.set_default_dtype(torch.float64); torch.set_num_threads(2)


def train(X, E, F, seed, steps=STEPS):
    torch.manual_seed(seed); rng = np.random.default_rng(seed); m = Student(HIDDEN)
    D = descriptor(X); m.mu, m.sd = D.mean(0), D.std(0).clamp_min(1e-3); m.e0, m.es = E.mean(), E.std()
    opt = torch.optim.Adam(m.parameters(), lr=3e-3); sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps, eta_min=1e-5); fs = F.std(); m.train()
    for _ in range(steps):
        idx = torch.tensor(rng.choice(len(X), size=min(512, len(X)), replace=False)); e, f = m.energy_forces(X[idx])
        loss = ((f - F[idx]) ** 2).mean() / fs**2 + 0.1 * ((e - E[idx]) ** 2).mean() / m.es**2
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    return m.eval()


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "prepare":
        import openmm.unit as u
        rng = np.random.default_rng(0); L = np.load("labels.npz")
        Xts = np.concatenate([np.load(f)["ts_xyz"][::40] for f in sorted(glob.glob("traj_tsx_s*.npz"))]).astype(np.float64)
        Xts = Xts[rng.choice(len(Xts), size=min(15000, len(Xts)), replace=False)]
        ctx, _, _ = teacher_context(); Ets = np.empty(len(Xts)); Fts = np.empty_like(Xts)
        for k, x in enumerate(Xts):
            ctx.setPositions(x); st = ctx.getState(getEnergy=True, getForces=True)
            Ets[k] = st.getPotentialEnergy().value_in_unit(u.kilojoule_per_mole); Fts[k] = st.getForces(asNumpy=True).value_in_unit(u.kilojoule_per_mole / u.nanometer)
        perm = rng.permutation(len(L["E"])); i0, ipool, itest = perm[:N0], perm[N0:N0 + 15000], perm[-3000:]
        Xp = np.concatenate([L["X"][ipool], Xts]); Ep = np.concatenate([L["E"][ipool], Ets]); Fp = np.concatenate([L["F"][ipool], Fts])
        ang = np.array([[dihedral(x, PHI), dihedral(x, PSI)] for x in Xp])
        # leverage ratio on the CV grid: nu from the teacher's free-energy-surface network / density of the pool
        ref, _ = a.load("ref_g1"); C_lag, keep = a.msm(ref); C = a.sqra(C_lag, keep); q = a.committor(C, keep)
        i, j = np.nonzero(np.triu(C, 1)); nu_e = C[i, j] * (q[i] - q[j]) ** 2
        nu = np.zeros(a.NB**2); nu[keep] = np.bincount(i, nu_e / 2, len(keep)) + np.bincount(j, nu_e / 2, len(keep)); nu /= nu.sum()
        b = np.clip(((ang + np.pi) / (2 * np.pi) * a.NB).astype(int), 0, a.NB - 1); cell = b[:, 0] * a.NB + b[:, 1]
        dens = np.bincount(cell, minlength=a.NB**2) / len(cell); ratio = nu[cell] / np.maximum(dens[cell], 1e-12)
        np.savez_compressed("al_pool.npz", X0=L["X"][i0], E0=L["E"][i0], F0=L["F"][i0], Xp=Xp, Ep=Ep, Fp=Fp, ratio=ratio, ang=ang,
                            Xt=L["X"][itest], Et=L["E"][itest], Ft=L["F"][itest])
        print(f"pool: {len(Xp)} frames ({len(Xts)} from the barrier window); leverage mass carried by the pool cells: {nu[np.unique(cell)].sum():.3f}; "
              f"frames with ratio > 1: {(ratio > 1).sum()}")

    elif mode == "run":
        strat, seed = sys.argv[2], int(sys.argv[3]); P = np.load("al_pool.npz"); rng = np.random.default_rng(100 + seed)
        X, E, F = (torch.tensor(P[k]) for k in ("X0", "E0", "F0")); Xp, Ep, Fp = (torch.tensor(P[k]) for k in ("Xp", "Ep", "Fp"))
        Xt, Ft = torch.tensor(P["Xt"]), torch.tensor(P["Ft"]); avail = np.ones(len(Xp), bool); log = []
        for r in range(ROUNDS + 1):
            t0 = time.time(); comm = [train(X, E, F, 1000 * seed + 10 * r + c) for c in range(NCOMM)]
            with torch.no_grad():
                Ec = torch.stack([m.energy(Xp) for m in comm]).numpy() / a.KJ_PER_KT(600.0)
            _, f = comm[0].energy_forces(Xt); rmse = torch.sqrt(((f - Ft) ** 2).mean()).item() * KJNM_TO_MEVA
            var = Ec.var(0); w = P["ratio"] / P["ratio"].sum()
            dts = Ec[0] - Ep.numpy() / a.KJ_PER_KT(600.0); dts -= np.median(dts)
            width = np.log((w * np.exp(dts - (w * dts).sum())).sum()) + np.log((w * np.exp(-(dts - (w * dts).sum()))).sum())
            torch.save({"state": comm[0].state_dict(), "hidden": HIDDEN, "ntrain": len(X), "seed": seed, "rmse_f_meVA": rmse, "rmse_e_kJ": 0.0}, f"student_AL-{strat}-r{r}-s{seed}.pt")
            frac_ts = float(np.mean(np.abs(np.degrees(P["ang"][~avail, 0])) < 40)) if (~avail).any() else 0.0
            log.append((r, len(X), rmse, width, frac_ts)); print(f"[{strat} s{seed}] round {r}: n={len(X)}  force RMSE {rmse:.1f} meV/A  leverage-weighted bracket width (oracle) {width:.2f}  "
                                                                  f"selected so far in |phi|<40: {frac_ts:.2f}  [{time.time()-t0:.0f}s]", flush=True)
            if r == ROUNDS: break
            score = {"random": rng.random(len(Xp)), "unc": var, "lev": var * P["ratio"]}[strat]; score = np.where(avail, score, -np.inf)
            pick = np.argsort(score)[-NADD:]; avail[pick] = False
            X, E, F = torch.cat([X, Xp[pick]]), torch.cat([E, Ep[pick]]), torch.cat([F, Fp[pick]])
        np.savetxt(f"../data/ala2_AL_{strat}_s{seed}.csv", log, delimiter=",", header="round,n_labels,force_rmse_meVA,bracket_width_oracle,frac_selected_in_barrier_window", comments="")
