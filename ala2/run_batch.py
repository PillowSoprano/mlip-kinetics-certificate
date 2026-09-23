"""Run all perturbed systems of design.json, N_PAR single-threaded trajectories at a time, one progress line per finished job."""
import json, subprocess, sys, time
N_PAR, NS, TEMP = 4, 100, 600
GAMMA = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0
SYSTEMS = sys.argv[2].split(',') if len(sys.argv) > 2 else None
SEEDS = tuple(range(1, 1 + (int(sys.argv[3]) if len(sys.argv) > 3 else 3)))
TAG = '' if GAMMA == 1.0 else f'_g{GAMMA:g}'
design = dict(ref=[], **json.load(open("design.json")))
jobs = [(f"{n}{TAG}_s{s}", 10 * s + len(n), json.dumps(b, separators=(",", ":"))) for n, b in design.items() if (SYSTEMS is None or n in SYSTEMS) for s in SEEDS]
running, t0 = [], time.time()
while jobs or running:
    while jobs and len(running) < N_PAR:
        name, seed, bumps = jobs.pop(0)
        p = subprocess.Popen([sys.executable, "run_md.py", name, str(NS), str(seed), str(TEMP), str(GAMMA), bumps], stdout=open(f"log_{name}.txt", "w"), stderr=subprocess.STDOUT)
        running.append((name, p))
    for name, p in running[:]:
        if p.poll() is not None:
            running.remove((name, p))
            print(f"[{(time.time()-t0)/60:5.1f} min] finished {name} (exit {p.returncode}); {len(jobs)} queued, {len(running)} running", flush=True)
    time.sleep(5)
