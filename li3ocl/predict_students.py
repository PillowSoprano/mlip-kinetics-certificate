"""
Reference-side predictions for a Li3OCl student (protocol of Sec. VI C of the paper).

    python predict_students.py TEACHER.model STUDENT.model HOPS.npz FRAMES.npz EQ.npz T_K TAG

HOPS.npz   from analyze_hops.py (frame index into FRAMES, hop id, s, q, dq);   FRAMES.npz  the run it was made from;   EQ.npz  equilibrium teacher frames (key X) at T_K.
For R in (3,4,5,6,8 A, whole cell): dV_R = sum over atoms within R of the hop midpoint of (student - teacher) site energies, after removing the per-species mean of that
difference on the equilibrium frames (different E0 conventions would otherwise leak into a truncated sum).  Whole cell = total energies.
Printed and saved per R: bracket, series law (8 slices in q), first order, bootstrap sd over hops; and for total energies the PAIRED lower bound (frame 0.2 ps before the segment).
"""
import sys, numpy as np, torch
torch.set_num_threads(4)
try: torch.serialization.add_safe_globals([slice])
except Exception: pass
sys.path.insert(0, "../colab")
from batched_md import BatchedMD
from analyze_hops import ideal
KB = 8.617333e-5; RS = (3.0, 4.0, 5.0, 6.0, 8.0, np.inf); NSL = 8

def site_energies(model, numbers, L, X, chunk=16):
    out = []
    for i in range(0, len(X), chunk):
        eng = BatchedMD(model, numbers, L, X[i:i + chunk], 300.0); pos = eng.x.reshape(-1, 3)
        data = {"positions": pos, "node_attrs": eng.node_attrs, "edge_index": eng.edge_index, "unit_shifts": eng.unit_shifts, "shifts": eng.unit_shifts * eng.L, "cell": eng.cell,
                "batch": eng.batch, "ptr": eng.ptr, "head": torch.zeros(eng.B, dtype=torch.long)}
        o = eng.model(data, compute_force=False, training=False); ne = o["node_energy"].detach().reshape(eng.B, -1).double().numpy()
        assert np.allclose(ne.sum(1), o["energy"].detach().double().numpy(), atol=1e-3), "site energies do not sum to the total energy"
        out.append(ne)
        if (i // chunk) % 50 == 0: print(f"    site energies {min(i + chunk, len(X))}/{len(X)}", flush=True)
    return np.concatenate(out)

def mic(d, L): return d - L * np.round(d / L)

def estimates(dts, deq, dq, q, hop, beta, rng, nboot=200):
    def one(sel_h):
        m = np.isin(hop, sel_h) if sel_h is not None else np.ones(len(hop), bool)
        if sel_h is not None:                                   # bootstrap with multiplicity
            cnt = np.bincount(sel_h, minlength=hop.max() + 1)[hop]; w_ = cnt
        else: w_ = np.ones(len(hop))
        w = dq * w_; W = w.sum(); wa = np.abs(dq) * w_; x = beta * dts; a = np.exp(-beta * deq).mean()
        up = np.log((w * np.exp(-x)).sum() / W) - np.log(a); lo = -np.log((w * np.exp(x)).sum() / W) - np.log(a); fo = -(w * x).sum() / W + beta * deq.mean()
        o = np.argsort(q, kind="stable"); cw = np.cumsum(wa[o]) / wa.sum(); sl = np.empty(len(q), int); sl[o] = np.minimum((cw * NSL).astype(int), NSL - 1)
        pk = np.array([wa[sl == k].sum() for k in range(NSL)]); ak = np.array([(wa[sl == k] * np.exp(-x[sl == k])).sum() / max(wa[sl == k].sum(), 1e-300) for k in range(NSL)])
        ser = -np.log((pk / pk.sum() / ak).sum()) - np.log(a)   # series law as for alanine dipeptide: conductance <e^{-x}> inside a slice of q, slices in series
        return np.array([lo, up, ser, fo])
    c = one(None); H = np.unique(hop); bs = np.array([one(rng.choice(H, len(H))) for _ in range(nboot)]); return c, bs.std(0)

if __name__ == "__main__":
    tfile, sfile, hfile, ffile, efile, T, tag = sys.argv[1:8]; T = float(T); beta = 1 / (KB * T); rng = np.random.default_rng(0)
    teacher = torch.load(tfile, map_location="cpu", weights_only=False); student = torch.load(sfile, map_location="cpu", weights_only=False)
    H = np.load(hfile); F = np.load(ffile); Z, L, S = F["numbers"], F["cell"], F["sites"]; pos0, _ = ideal(); fw = np.where(Z != 3)[0]; li = np.where(Z == 3)[0]; ref = pos0[fw].mean(0)
    fr, hop, q, dq, s = H["frame"], H["hop"], H["q"], H["dq"], H["s"]; X = F["frames"][fr].astype(np.float64); X = X - (X[:, fw].mean(1) - ref)[:, None]
    Xeq = np.load(efile)["X"].astype(np.float64); Xeq = Xeq - (Xeq[:, fw].mean(1) - ref)[:, None]
    print(f"{tag}: {len(np.unique(hop))} hops, {len(X)} path frames, {len(Xeq)} equilibrium frames, T = {T:.0f} K", flush=True)
    # hop midpoints: the hopping ion is the Li whose s-coordinate was recorded -> recover it as the Li farthest from every site in the most central frame of each hop
    mid = np.empty((len(X), 3))
    for h in np.unique(hop):
        m = np.where(hop == h)[0]; c = m[np.argmin(np.abs(s[m] - 0.5))]; d = mic(X[c][li][:, None] - S[None], L); r = np.linalg.norm(d, axis=-1).min(1); ion = li[np.argmax(r)]
        mid[m] = X[c][ion]                                     # position of the hopping ion at the top of the hop ~ midpoint of the two sites
    # equilibrium frames: centre on the midpoint between the vacancy and one of its nearest Li sites
    mideq = np.empty((len(Xeq), 3))
    for k, x in enumerate(Xeq):
        d = mic(x[li][:, None] - S[None], L); occ = np.zeros(len(S), bool); occ[np.linalg.norm(d, axis=-1).argmin(1)] = True; vac = S[~occ][0] if (~occ).any() else S[0]
        nb = S[np.argsort(np.linalg.norm(mic(S - vac, L), axis=1))[1 + rng.integers(8)]]; mideq[k] = vac + 0.5 * mic(nb - vac, L)
    import os
    cache = f"../data/li3ocl_teacher_site_energies_{os.path.basename(hfile).replace('.npz','')}.npz"          # the teacher is evaluated once per (hops, eq) data set
    if os.path.exists(cache): c_ = np.load(cache); Tts, Teq = c_["ts"], c_["eq"]; assert len(Tts) == len(X) and len(Teq) == len(Xeq)
    else:
        print("  teacher site energies (cached afterwards)", flush=True); Tts, Teq = site_energies(teacher, Z, L, X), site_energies(teacher, Z, L, Xeq); np.savez_compressed(cache, ts=Tts, eq=Teq)
    print("  student site energies", flush=True); dts = site_energies(student, Z, L, X) - Tts; deq = site_energies(student, Z, L, Xeq) - Teq
    for z in np.unique(Z): off = deq[:, Z == z].mean(); dts[:, Z == z] -= off; deq[:, Z == z] -= off
    pre = H["pre_frame"] if "pre_frame" in H.files else None; PAIRED = None
    if pre is not None and (pre >= 0).sum() >= 10:
        okh = np.where(pre >= 0)[0]; Xp = F["frames"][pre[okh]].astype(np.float64); print("  site energies on the paired basin frames", flush=True)
        dpre = (site_energies(student, Z, L, Xp) - site_energies(teacher, Z, L, Xp)).sum(1); PAIRED = (okh, dpre)
    rows = []; print(f"   R (A) | atoms | sd(beta dV) on paths | lower    upper    | series law   | first order")
    for R in RS:
        mts = np.linalg.norm(mic(X - mid[:, None], L), axis=-1) < R; meq = np.linalg.norm(mic(Xeq - mideq[:, None], L), axis=-1) < R
        a, b = (dts * mts).sum(1), (deq * meq).sum(1); c, sd = estimates(a, b, dq, q, hop, beta, rng)
        print(f"  {R:5.1f}  | {mts.sum(1).mean():5.1f} | {beta * a.std():8.2f}             | {c[0]:+.2f}({sd[0]:.2f}) {c[1]:+.2f}({sd[1]:.2f}) | {c[2]:+.2f}({sd[2]:.2f}) | {c[3]:+.2f}({sd[3]:.2f})", flush=True)
        rows.append((R if np.isfinite(R) else -1, mts.sum(1).mean(), beta * a.std(), *c, *sd))
    if PAIRED is not None:
        okh, dpre = PAIRED; m = np.isin(hop, okh); dtot = dts.sum(1)[m] - dpre[np.searchsorted(okh, hop[m])]; w = dq[m]
        print(f"  whole cell, PAIRED lower bound ({len(okh)} hops with a basin frame 0.2 ps earlier): {-np.log((w * np.exp(beta * dtot)).sum() / w.sum()):+.2f}   (unpaired: {rows[-1][3]:+.2f})")
    np.savetxt(f"../data/li3ocl_pred_{tag}.csv", rows, delimiter=",", comments="", header="R_A,atoms,sd_beta_dV,lower,upper,series,first_order,sd_lower,sd_upper,sd_series,sd_first")
