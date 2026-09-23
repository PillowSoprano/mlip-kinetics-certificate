"""
One single-threaded alanine-dipeptide trajectory (vacuum, Amber14, Langevin dynamics at TEMP_K), optionally with a perturbation
    dV(phi, psi) = sum_k A_k * exp(-(dphi_k^2 + dpsi_k^2) / (2 w_k^2))        [periodic differences, A in kJ/mol, angles in rad]
Output: one small .npz with phi, psi every 0.1 ps.  One trajectory per call.

usage: python run_md.py NAME NS SEED TEMP_K GAMMA_PER_PS ['[[A, phi0, psi0, w], ...]']
Also stores both peptide-bond omegas (cis/trans monitor) and Cartesian frames every 10 ps (for true atomic force RMSE).
"""
import sys, json, time
import numpy as np
import openmm as mm, openmm.app as app, openmm.unit as u

name, ns, seed = sys.argv[1], float(sys.argv[2]), int(sys.argv[3])
TEMP, GAMMA = float(sys.argv[4]), float(sys.argv[5])
bumps = json.loads(sys.argv[6]) if len(sys.argv) > 6 else []
import os
STRIDE, DT_FS = int(os.environ.get('ALA_STRIDE', '50')), 2.0      # ALA_STRIDE=5 -> 10 fs output, for path integrals over ballistic crossings

pdb = app.PDBFile("ala2.pdb")
ff = app.ForceField("amber14-all.xml")
system = ff.createSystem(pdb.topology, nonbondedMethod=app.NoCutoff, constraints=app.HBonds)
atoms = {(a.residue.name, a.name): a.index for a in pdb.topology.atoms()}
PHI = [atoms[("ACE", "C")], atoms[("ALA", "N")], atoms[("ALA", "CA")], atoms[("ALA", "C")]]
PSI = [atoms[("ALA", "N")], atoms[("ALA", "CA")], atoms[("ALA", "C")], atoms[("NME", "N")]]
OM1 = [atoms[("ACE", "CH3")], atoms[("ACE", "C")], atoms[("ALA", "N")], atoms[("ALA", "CA")]]
OM2 = [atoms[("ALA", "CA")], atoms[("ALA", "C")], atoms[("NME", "N")], atoms[("NME", "C")]]
if bumps:
    wrap = lambda d: f"({d} - 6.283185307179586*floor(({d} + 3.141592653589793)/6.283185307179586))"
    expr = f"A*exp(-({wrap('(dihedral(p1,p2,p3,p4)-phi0)')}^2 + {wrap('(dihedral(p2,p3,p4,p5)-psi0)')}^2)/(2*w^2))"
    f = mm.CustomCompoundBondForce(5, expr)
    for p in ("A", "phi0", "psi0", "w"):
        f.addPerBondParameter(p)
    for A, phi0, psi0, w in bumps:
        f.addBond(PHI + [PSI[3]], [A, phi0, psi0, w])
    system.addForce(f)

integ = mm.LangevinMiddleIntegrator(TEMP * u.kelvin, GAMMA / u.picosecond, DT_FS * u.femtosecond)
integ.setRandomNumberSeed(seed)
sim = app.Simulation(pdb.topology, system, integ, mm.Platform.getPlatformByName("Reference"))
sim.context.setPositions(pdb.positions); sim.minimizeEnergy(); sim.context.setVelocitiesToTemperature(TEMP * u.kelvin, seed)
sim.step(5000)


def dihedral(x, idx):
    p0, p1, p2, p3 = x[idx]
    b0, b1, b2 = p0 - p1, p2 - p1, p3 - p2
    b1n = b1 / np.linalg.norm(b1)
    v, w_ = b0 - b0.dot(b1n) * b1n, b2 - b2.dot(b1n) * b1n
    return np.arctan2(np.cross(b1n, v).dot(w_), v.dot(w_))


nfr = int(ns * 1e6 / DT_FS / STRIDE)
out = np.empty((nfr, 2), dtype=np.float32); om = np.empty((nfr, 2), dtype=np.float32); xyz = []
TS_XYZ = os.environ.get('ALA_TSXYZ') == '1'; ts_xyz, ts_idx = [], []      # every frame with -60 < phi < 45 deg, for energy errors on transition paths
t0 = time.time()
for k in range(nfr):
    sim.step(STRIDE)
    x = sim.context.getState(getPositions=True).getPositions(asNumpy=True).value_in_unit(u.nanometer)
    out[k] = dihedral(x, PHI), dihedral(x, PSI); om[k] = dihedral(x, OM1), dihedral(x, OM2)
    if k % (5000 // STRIDE) == 0:
        xyz.append(x.astype(np.float32))
    if TS_XYZ and -1.05 < out[k, 0] < 0.79:
        ts_xyz.append(x.astype(np.float32)); ts_idx.append(k)
    if (k + 1) % (nfr // 10) == 0:
        print(f"[{name}] {100*(k+1)//nfr}%  {(k+1)*STRIDE*DT_FS*1e-6:.0f} ns  elapsed {time.time()-t0:.0f}s", flush=True)
np.savez_compressed(f"traj_{name}.npz", phipsi=out, dt_ps=STRIDE * DT_FS * 1e-3, bumps=np.array(bumps, dtype=float), seed=seed, temp=TEMP, gamma=GAMMA,
                    cis_fraction=(np.abs(om) < np.pi / 2).mean(0), omega=om[::max(1, 500 // STRIDE)], xyz=np.array(xyz),
                    ts_xyz=np.array(ts_xyz), ts_idx=np.array(ts_idx))
