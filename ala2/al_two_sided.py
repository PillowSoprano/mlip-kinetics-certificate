"""
R2b: TWO-SIDED acquisition on alanine dipeptide.  Same start, pool, budget (250 teacher labels per round, 3 rounds) and training as active_learning.py.

Each round:  125 labels  <- reference side: committee disagreement x leverage ratio on the fixed pool            (strategy 'lev')
             125 labels  <- model side: frames of the STUDENT's own reactive paths (2 ns of student MD, any route between the cores, including phi ~ 180),
                            drawn with probability ~ committee disagreement; each costs one teacher single point, counted in the budget.
usage: python al_two_sided.py SEED
"""
import sys, time, numpy as np, torch
import openmm as mm, openmm.app as app, openmm.unit as u
import analyze as a
from student import teacher_context, KJNM_TO_MEVA
from run_md_common import dihedral_indices, dihedral
from active_learning import train, N0, NADD, ROUNDS, NCOMM, HIDDEN
torch.set_default_dtype(torch.float64); torch.set_num_threads(2)
NS_MD, STRIDE = 2.0, 50
ctx, pdb, tsys = teacher_context(); PHI, PSI, OM1, OM2 = dihedral_indices(pdb.topology)

def student_reactive_frames(model, seed):
    system = mm.System()
    for i in range(tsys.getNumParticles()): system.addParticle(tsys.getParticleMass(i))
    for i in range(tsys.getNumConstraints()): system.addConstraint(*tsys.getConstraintParameters(i))
    def compute(state):
        x = torch.tensor(state.getPositions(asNumpy=True).value_in_unit(u.nanometer), dtype=torch.float64)[None]; e, f = model.energy_forces(x)
        return e.item() * u.kilojoule_per_mole, f[0].detach().numpy() * u.kilojoule_per_mole / u.nanometer
    system.addForce(mm.PythonForce(compute)); integ = mm.LangevinMiddleIntegrator(600 * u.kelvin, 1 / u.picosecond, 2 * u.femtosecond); integ.setRandomNumberSeed(seed)
    sim = app.Simulation(pdb.topology, system, integ, mm.Platform.getPlatformByName("Reference")); sim.context.setPositions(pdb.positions); sim.minimizeEnergy(maxIterations=300)
    sim.context.setVelocitiesToTemperature(600 * u.kelvin, seed); sim.step(2500); n = int(NS_MD * 1e6 / 2 / STRIDE); phi = np.empty(n); xs = {}
    for k in range(n):
        sim.step(STRIDE); x = sim.context.getState(getPositions=True).getPositions(asNumpy=True).value_in_unit(u.nanometer); phi[k] = np.degrees(dihedral(x, PHI))
        if not np.isfinite(phi[k]): phi = phi[:k]; break
        if not ((-150 < phi[k] < -40) or (30 < phi[k] < 110)): xs[k] = x.astype(np.float64)
    st = np.where((phi > -150) & (phi < -40), 0, np.where((phi > 30) & (phi < 110), 1, -1)); last, li, out, ntr = -1, -1, [], 0
    for k, s in enumerate(st):
        if s >= 0:
            if last >= 0 and s != last: out += [xs[j] for j in range(li + 1, k) if j in xs]; ntr += 1
            last, li = s, k
    return np.array(out), ntr

if __name__ == "__main__":
    seed = int(sys.argv[1]); P = np.load("al_pool.npz"); rng = np.random.default_rng(100 + seed)
    X, E, F = (torch.tensor(P[k]) for k in ("X0", "E0", "F0")); Xp, Ep, Fp = (torch.tensor(P[k]) for k in ("Xp", "Ep", "Fp")); Xt, Ft = torch.tensor(P["Xt"]), torch.tensor(P["Ft"])
    avail = np.ones(len(Xp), bool); log = []
    for r in range(ROUNDS + 1):
        t0 = time.time(); comm = [train(X, E, F, 1000 * seed + 10 * r + c) for c in range(NCOMM)]
        _, f = comm[0].energy_forces(Xt); rmse = torch.sqrt(((f - Ft) ** 2).mean()).item() * KJNM_TO_MEVA
        torch.save({"state": comm[0].state_dict(), "hidden": HIDDEN, "ntrain": len(X), "seed": seed, "rmse_f_meVA": rmse, "rmse_e_kJ": 0.0}, f"student_AL-two-r{r}-s{seed}.pt")
        print(f"[two s{seed}] round {r}: n={len(X)}  force RMSE {rmse:.1f} meV/A  [{time.time()-t0:.0f}s]", flush=True)
        if r == ROUNDS: break
        # reference side
        with torch.no_grad(): var = torch.stack([m.energy(Xp) for m in comm]).numpy().var(0)
        score = np.where(avail, var * P["ratio"], -np.inf); pick = np.argsort(score)[-NADD // 2:]; avail[pick] = False
        # model side
        t1 = time.time(); Xs, ntr = student_reactive_frames(comm[0], 7000 + 10 * seed + r); nm = NADD - len(pick)
        if len(Xs) >= nm:
            Xs_t = torch.tensor(Xs)
            with torch.no_grad(): v = torch.stack([m.energy(Xs_t) for m in comm]).numpy().var(0)
            sel = rng.choice(len(Xs), size=nm, replace=False, p=v / v.sum()); Xn = Xs[sel]; En = np.empty(nm); Fn = np.empty_like(Xn)
            for k, x in enumerate(Xn):
                ctx.setPositions(x); s_ = ctx.getState(getEnergy=True, getForces=True)
                En[k] = s_.getPotentialEnergy().value_in_unit(u.kilojoule_per_mole); Fn[k] = s_.getForces(asNumpy=True).value_in_unit(u.kilojoule_per_mole / u.nanometer)
            with torch.no_grad(): dv = (comm[0].energy(torch.tensor(Xn)).numpy() - En) / a.KJ_PER_KT(600.0)
            om = np.degrees(np.abs([[dihedral(x, OM1), dihedral(x, OM2)] for x in Xn])).min(1)
            print(f"           model side: {ntr} student transitions in {NS_MD} ns, {len(Xs)} reactive frames; picked {nm}: student-teacher energy error median {np.median(dv):+.1f} kT, "
                  f"5%/95% {np.quantile(dv,.05):+.0f}/{np.quantile(dv,.95):+.0f} kT; picked frames with a peptide bond below 120 deg: {np.mean(om < 120):.2f}  [{time.time()-t1:.0f}s]", flush=True)
            ok = np.isfinite(En) & (np.abs(Fn).max((1, 2)) < 1e5)                          # teacher energies of wildly unphysical frames can overflow: drop those labels (still counted)
            X, E, F = torch.cat([X, torch.tensor(Xn[ok])]), torch.cat([E, torch.tensor(En[ok])]), torch.cat([F, torch.tensor(Fn[ok])]); nbad = int((~ok).sum())
        else:
            print(f"           model side: only {len(Xs)} reactive frames ({ntr} transitions): all {NADD} labels go to the reference side", flush=True)
            score = np.where(avail, var * P["ratio"], -np.inf); extra = np.argsort(score)[-nm:]; avail[extra] = False; pick = np.concatenate([pick, extra]); nbad = 0
        X, E, F = torch.cat([X, Xp[pick]]), torch.cat([E, Ep[pick]]), torch.cat([F, Fp[pick]]); log.append((r, len(X), rmse, ntr, len(Xs), nbad))
    np.savetxt(f"../data/ala2_AL_two_s{seed}.csv", log, delimiter=",", header="round,n_labels_after,force_rmse_meVA,student_transitions_in_2ns,reactive_frames,unusable_labels", comments="")
