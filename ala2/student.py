"""
Stage 2: real machine-learned students of the alanine-dipeptide teacher (Amber14, vacuum).

  python student.py label                     teacher energies / forces on all stored Cartesian frames  -> labels.npz
  python student.py train NAME HIDDEN NTRAIN SEED [TS_FRAC]   train one student -> student_NAME.pt  (+ force RMSE in meV/A)
  python student.py speed NAME                steps per second of OpenMM MD driven by the student (PythonForce)

Student: invariant descriptor = inverse distances of all 231 atom pairs -> MLP -> energy; forces by autograd.
"""
import sys, glob, time
import numpy as np
import torch

torch.set_default_dtype(torch.float64); torch.set_num_threads(2)
KJNM_TO_MEVA = 1.0364
IU = np.triu_indices(22, 1)


def descriptor(x):                       # x: (n, 22, 3) in nm
    d = x[:, IU[0]] - x[:, IU[1]]
    return 0.1 / torch.sqrt((d**2).sum(-1))          # 1 / r in 1/Angstrom-ish units


class Student(torch.nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.net = torch.nn.Sequential(torch.nn.Linear(231, hidden), torch.nn.SiLU(), torch.nn.Linear(hidden, hidden), torch.nn.SiLU(), torch.nn.Linear(hidden, 1))
        self.register_buffer("mu", torch.zeros(231)); self.register_buffer("sd", torch.ones(231))
        self.register_buffer("e0", torch.zeros(())); self.register_buffer("es", torch.ones(()))

    def energy(self, x):
        return self.net((descriptor(x) - self.mu) / self.sd).squeeze(-1) * self.es + self.e0

    def energy_forces(self, x):
        x = x.clone().requires_grad_(True); e = self.energy(x)
        f, = torch.autograd.grad(e.sum(), x, create_graph=self.training)
        return e, -f


def teacher_context():
    import openmm as mm, openmm.app as app
    pdb = app.PDBFile("ala2.pdb"); ff = app.ForceField("amber14-all.xml")
    system = ff.createSystem(pdb.topology, nonbondedMethod=app.NoCutoff, constraints=app.HBonds)
    return mm.Context(system, mm.VerletIntegrator(0.001), mm.Platform.getPlatformByName("Reference")), pdb, system


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "label":
        import openmm.unit as u
        ctx, _, _ = teacher_context()
        X = np.concatenate([np.load(f)["xyz"] for f in sorted(glob.glob("traj_ref_g10_s*.npz")) + sorted(glob.glob("traj_fine_s*.npz"))]).astype(np.float64)
        E = np.empty(len(X)); F = np.empty_like(X)
        for k, x in enumerate(X):
            ctx.setPositions(x); st = ctx.getState(getEnergy=True, getForces=True)
            E[k] = st.getPotentialEnergy().value_in_unit(u.kilojoule_per_mole); F[k] = st.getForces(asNumpy=True).value_in_unit(u.kilojoule_per_mole / u.nanometer)
        np.savez_compressed("labels.npz", X=X, E=E, F=F); print(len(X), "frames labelled; energy sd", E.std().round(1), "kJ/mol; force rms", np.sqrt((F**2).mean()).round(1), "kJ/mol/nm")

    elif mode == "train":
        name, hidden, ntrain, seed = sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
        nsteps = int(sys.argv[6]) if len(sys.argv) > 6 else 6000
        d = np.load("labels.npz"); rng = np.random.default_rng(seed); perm = rng.permutation(len(d["E"]))
        tr, te = perm[:ntrain], perm[-3000:]
        X, E, F = (torch.tensor(d[k]) for k in ("X", "E", "F"))
        torch.manual_seed(seed); m = Student(hidden)
        # constrained X-H pairs have ~zero spread: floor the scale, or the net sees 1e6-sized inputs
        D = descriptor(X[tr]); m.mu, m.sd = D.mean(0), D.std(0).clamp_min(1e-3)
        m.e0, m.es = E[tr].mean(), E[tr].std()
        opt = torch.optim.Adam(m.parameters(), lr=3e-3); sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, nsteps, eta_min=1e-5)
        fs = F[tr].std(); t0 = time.time(); m.train()
        for ep in range(nsteps):
            idx = torch.tensor(rng.choice(tr, size=min(512, ntrain), replace=False))
            e, f = m.energy_forces(X[idx])
            loss = ((f - F[idx]) ** 2).mean() / fs**2 + 0.1 * ((e - E[idx]) ** 2).mean() / m.es**2
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        m.eval(); e, f = m.energy_forces(X[te])
        rmse_f = torch.sqrt(((f - F[te]) ** 2).mean()).item() * KJNM_TO_MEVA
        rmse_e = torch.sqrt(((e - E[te] - (e - E[te]).mean()) ** 2).mean()).item()
        torch.save({"state": m.state_dict(), "hidden": hidden, "ntrain": ntrain, "seed": seed, "rmse_f_meVA": rmse_f, "rmse_e_kJ": rmse_e}, f"student_{name}.pt")
        print(f"[{name}] hidden={hidden} ntrain={ntrain} seed={seed}: force RMSE {rmse_f:.1f} meV/A, energy RMSE {rmse_e:.2f} kJ/mol ({rmse_e/4.99:.2f} kT at 600 K)  [{time.time()-t0:.0f}s]", flush=True)

    elif mode == "speed":
        import openmm as mm, openmm.app as app, openmm.unit as u
        ck = torch.load(f"student_{sys.argv[2]}.pt"); m = Student(ck["hidden"]); m.load_state_dict(ck["state"]); m.eval()
        _, pdb, tsys = teacher_context()
        system = mm.System()
        for i in range(tsys.getNumParticles()): system.addParticle(tsys.getParticleMass(i))
        for i in range(tsys.getNumConstraints()): system.addConstraint(*tsys.getConstraintParameters(i))

        def compute(state):
            x = torch.tensor(state.getPositions(asNumpy=True).value_in_unit(u.nanometer))[None]
            e, f = m.energy_forces(x)
            return e.item() * u.kilojoule_per_mole, f[0].detach().numpy() * u.kilojoule_per_mole / u.nanometer
        system.addForce(mm.PythonForce(compute))
        integ = mm.LangevinMiddleIntegrator(600 * u.kelvin, 1 / u.picosecond, 2 * u.femtosecond)
        sim = app.Simulation(pdb.topology, system, integ, mm.Platform.getPlatformByName("Reference"))
        sim.context.setPositions(pdb.positions); sim.minimizeEnergy(maxIterations=200); sim.step(200)
        t0 = time.time(); sim.step(3000); dt = time.time() - t0
        print(f"student MD: {3000/dt:.0f} steps/s  ->  {3000/dt*2e-6*86400:.1f} ns/day on one core; potential energy now {sim.context.getState(getEnergy=True).getPotentialEnergy()}")
