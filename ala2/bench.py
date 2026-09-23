import time, sys, numpy as np
import openmm as mm, openmm.app as app, openmm.unit as u
pdb = app.PDBFile("ala2.pdb")
for label, ffs, kw in (("vacuum", ["amber14-all.xml"], {}), ("implicit GBn2", ["amber14-all.xml", "implicit/gbn2.xml"], {})):
    ff = app.ForceField(*ffs)
    system = ff.createSystem(pdb.topology, nonbondedMethod=app.NoCutoff, constraints=app.HBonds, **kw)
    for plat, props in (("CPU", {"Threads": "1"}), ("CPU", {"Threads": "4"}), ("Reference", {})):
        integ = mm.LangevinMiddleIntegrator(300 * u.kelvin, 1 / u.picosecond, 2 * u.femtosecond)
        sim = app.Simulation(pdb.topology, system, integ, mm.Platform.getPlatformByName(plat), props)
        sim.context.setPositions(pdb.positions); sim.minimizeEnergy(); sim.step(2000)
        n = 100000 if plat == "CPU" else 20000
        t0 = time.time(); sim.step(n); dt = time.time() - t0
        print(f"{label:14s} {plat:9s} {props}  {n*2e-6/dt*86400:9.0f} ns/day", flush=True)
