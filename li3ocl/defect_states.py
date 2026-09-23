"""
Separate single-vacancy transport from transport with extra defects.   python defect_states.py FILE.npz [FILE2.npz ...]

State of a replica at a stored frame: every Li is assigned to its nearest site (framework-relative); n_double = number of sites holding two or more Li.
n_double = 0  <=>  80 Li on 80 distinct sites, one vacancy: the vacancy mechanism of the protocol.   n_double >= 1: a Frenkel-type defect (extra vacancy + shared/interstitial Li).
Frames exist only around hops and excursions; the state seen in a window is held until the next window of that replica (defects are long-lived on this scale).
A hop is attributed to the state at the START of its window (0.25-0.5 ps before it), so that the transit itself does not count as a defect.
"""
import sys, numpy as np
from analyze_hops import ideal

def analyse(path):
    d = np.load(path); X, st, rp, Z, L, S, ev = d["frames"], d["frame_step"], d["frame_replica"], d["numbers"], d["cell"], d["sites"], d["events"].reshape(-1, 5)
    pos0, _ = ideal(); li, fw = np.where(Z == 3)[0], np.where(Z != 3)[0]; ref = pos0[fw].mean(0); nrep = int(max(rp.max() if len(rp) else 0, ev[:, 1].max() if len(ev) else 0)) + 1
    if "hops" in d.files: nrep = max(nrep, len(d["hops"]))
    tot_steps = float(d["ns_per_replica"]) / 2e-6; dead = d["dead"] if "dead" in d.files else np.full(nrep, -1)
    gap = int(np.diff(np.unique(st)).min()) if len(st) > 1 else 10; T = np.zeros(2); H = np.zeros(2, int); hrep = np.zeros((nrep, 2), int); trep = np.zeros((nrep, 2)); ever = 0
    for b in range(nrep):
        end = tot_steps if dead[b] < 0 else dead[b]; idx = np.where(rp == b)[0]
        if len(idx) == 0: T[0] += end; trep[b, 0] = end; continue
        idx = idx[np.argsort(st[idx])]; t = st[idx]; wstart = np.r_[0, np.where(np.diff(t) > gap)[0] + 1]            # first frame of every contiguous window
        x = X[idx[wstart]].astype(np.float64); x = x - (x[:, fw].mean(1) - ref)[:, None]; dd = x[:, li][:, :, None, :] - S[None, None]; dd -= L * np.round(dd / L)
        a = np.linalg.norm(dd, axis=-1).argmin(-1); state = np.array([(np.bincount(r, minlength=len(S)) >= 2).sum() > 0 for r in a]).astype(int); tw = t[wstart]; ever += state.any()
        edges = np.r_[0, tw[1:], end]; edges = np.minimum(edges, end)                                                  # state of window k holds from its start (first window: from 0) to the next window
        for k in range(len(tw)): T[state[k]] += max(edges[k + 1] - edges[k], 0); trep[b, state[k]] += max(edges[k + 1] - edges[k], 0)
        for e in ev[ev[:, 1] == b]:
            k = np.searchsorted(tw, e[0], side="right") - 1; s_ = state[max(k, 0)]; H[s_] += 1; hrep[b, s_] += 1
    ns = T * 2e-6; rate = H / np.maximum(ns, 1e-12); ok = trep[:, 0] > 0.5 * tot_steps; h0 = hrep[ok, 0] / (trep[ok, 0] / tot_steps)
    return dict(ns=ns, hops=H, rate=rate, frac_defect_time=ns[1] / ns.sum(), replicas_with_defect=int(ever), nrep=nrep, vm_single=float(h0.var() / max(h0.mean(), 1e-9)) if ok.sum() > 3 else np.nan)

if __name__ == "__main__":
    print("file                          | single vacancy: ns, hops, rate/ns (var/mean) | extra defect: ns, hops, rate/ns | time with defect | replicas that ever had one")
    rows = []
    for f in sys.argv[1:]:
        r = analyse(f); name = f.split("/")[-1].replace(".npz", "").replace("hops_", "").replace("li3ocl_", "")
        print(f"{name:29s} | {r['ns'][0]:6.2f} {r['hops'][0]:5d} {r['rate'][0]:7.1f} +- {np.sqrt(max(r['hops'][0],1))/max(r['ns'][0],1e-9):4.1f} ({r['vm_single']:.1f}) | {r['ns'][1]:6.2f} {r['hops'][1]:5d} {r['rate'][1]:7.1f} | {r['frac_defect_time']:6.1%} | {r['replicas_with_defect']}/{r['nrep']}", flush=True)
        rows.append((name, *r["ns"], *r["hops"], *r["rate"], r["frac_defect_time"], r["replicas_with_defect"], r["nrep"], r["vm_single"]))
    with open("../data/li3ocl_defect_states.csv", "a") as fh:
        for r in rows: fh.write(",".join(str(x) for x in r) + "\n")
