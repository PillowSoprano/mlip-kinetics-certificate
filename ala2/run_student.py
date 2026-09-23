"""One single-threaded alanine-dipeptide trajectory driven by a neural-network student (OpenMM PythonForce).
usage: python run_student.py STUDENT NS SEED        ->  traj_stu-STUDENT_sSEED.npz   (phi, psi every 0.1 ps; omega every 1 ps)"""
import sys, time, os
import numpy as np, torch
import openmm as mm, openmm.app as app, openmm.unit as u
from student import Student, teacher_context
from run_md_common import dihedral_indices, dihedral

stu, ns, seed = sys.argv[1], float(sys.argv[2]), int(sys.argv[3])
torch.set_num_threads(1)
ck = torch.load(f"student_{stu}.pt"); m = Student(ck["hidden"]); m.load_state_dict(ck["state"]); m.eval()
_, pdb, tsys = teacher_context()
system = mm.System()
for i in range(tsys.getNumParticles()): system.addParticle(tsys.getParticleMass(i))
for i in range(tsys.getNumConstraints()): system.addConstraint(*tsys.getConstraintParameters(i))


def compute(state):
    x = torch.tensor(state.getPositions(asNumpy=True).value_in_unit(u.nanometer))[None]
    e, f = m.energy_forces(x)
    return e.item() * u.kilojoule_per_mole, f[0].detach().numpy() * u.kilojoule_per_mole / u.nanometer


system.addForce(mm.PythonForce(compute))
integ = mm.LangevinMiddleIntegrator(600 * u.kelvin, 1 / u.picosecond, 2 * u.femtosecond); integ.setRandomNumberSeed(seed)
sim = app.Simulation(pdb.topology, system, integ, mm.Platform.getPlatformByName("Reference"))
sim.context.setPositions(pdb.positions); sim.minimizeEnergy(maxIterations=300); sim.context.setVelocitiesToTemperature(600 * u.kelvin, seed); sim.step(2500)
PHI, PSI, OM1, OM2 = dihedral_indices(pdb.topology)
STRIDE = int(os.environ.get('ALA_STRIDE', '50')); TSX = os.environ.get('ALA_TSXYZ') == '1'; ts_xyz, ts_idx = [], []
nfr = int(ns * 1e6 / 2.0 / STRIDE); out = np.empty((nfr, 2), np.float32); om = np.empty((nfr, 2), np.float32); emax = -1e9; t0 = time.time()
for k in range(nfr):
    sim.step(STRIDE)
    st = sim.context.getState(getPositions=True, getEnergy=(k % 100 == 0)); x = st.getPositions(asNumpy=True).value_in_unit(u.nanometer)
    out[k] = dihedral(x, PHI), dihedral(x, PSI); om[k] = dihedral(x, OM1), dihedral(x, OM2)
    if TSX and -1.05 < out[k, 0] < 0.79:
        ts_xyz.append(x.astype(np.float32)); ts_idx.append(k)
    if not np.isfinite(out[k]).all():
        print(f"[{stu}] trajectory blew up at frame {k}", flush=True); out, om = out[:k], om[:k]; break
    if (k + 1) % max(1, nfr // 10) == 0:
        print(f"[stu-{stu}_s{seed}] {100*(k+1)//nfr}%  {(k+1)*STRIDE*2e-6:.2f} ns  elapsed {time.time()-t0:.0f}s", flush=True)
np.savez_compressed(f"traj_stu-{stu}_s{seed}.npz" if (not TSX or STRIDE == 50) else f"traj_stutsx-{stu}_s{seed}.npz", phipsi=out, dt_ps=STRIDE * 2e-3, ts_xyz=np.array(ts_xyz), ts_idx=np.array(ts_idx), seed=seed, temp=600.0, gamma=1.0,
                    cis_fraction=(np.abs(om) < np.pi / 2).mean(0), omega=om[::max(1, 500 // STRIDE)], rmse_f_meVA=ck["rmse_f_meVA"])
