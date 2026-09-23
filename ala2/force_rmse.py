"""True atomic force RMSE of each perturbation, evaluated on reference frames (the quantity MLIP papers report).
usage: python force_rmse.py REF_SYSTEM_NAME        e.g.  ref_g10
RMSE = sqrt( mean over frames, atoms, xyz components of |dF|^2 ),  reported in meV/Angstrom."""
import sys, glob, json
import numpy as np
import openmm as mm, openmm.app as app, openmm.unit as u

KJNM_TO_MEVA = 1.0364                                    # 1 kJ/mol/nm = 1.0364 meV/Angstrom
pdb = app.PDBFile("ala2.pdb")
atoms = {(a.residue.name, a.name): a.index for a in pdb.topology.atoms()}
FIVE = [atoms[("ACE", "C")], atoms[("ALA", "N")], atoms[("ALA", "CA")], atoms[("ALA", "C")], atoms[("NME", "N")]]
xyz = np.concatenate([np.load(f)["xyz"] for f in sorted(glob.glob(f"traj_{sys.argv[1]}_s*.npz"))])
wrap = lambda d: f"({d} - 6.283185307179586*floor(({d} + 3.141592653589793)/6.283185307179586))"
expr = f"A*exp(-({wrap('(dihedral(p1,p2,p3,p4)-phi0)')}^2 + {wrap('(dihedral(p2,p3,p4,p5)-psi0)')}^2)/(2*w^2))"
print(f"{len(xyz)} reference frames")
for name, bumps in json.load(open("design.json")).items():
    system = mm.System()
    for _ in range(pdb.topology.getNumAtoms()):
        system.addParticle(1.0)
    f = mm.CustomCompoundBondForce(5, expr)
    for p in ("A", "phi0", "psi0", "w"):
        f.addPerBondParameter(p)
    for b in bumps:
        f.addBond(FIVE, b)
    system.addForce(f)
    ctx = mm.Context(system, mm.VerletIntegrator(0.001), mm.Platform.getPlatformByName("Reference"))
    sq = 0.0; mx = 0.0
    for x in xyz:
        ctx.setPositions(x)
        F = ctx.getState(getForces=True).getForces(asNumpy=True).value_in_unit(u.kilojoule_per_mole / u.nanometer)
        sq += (F**2).mean(); mx = max(mx, np.abs(F).max())
    print(f"   {name:8s} amplitude {bumps[0][0]:+7.2f} kJ/mol   atomic force RMSE = {np.sqrt(sq/len(xyz))*KJNM_TO_MEVA:7.3f} meV/A   (max component {mx*KJNM_TO_MEVA:7.1f} meV/A)")
