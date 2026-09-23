"""
Hop rates split by whether the hop itself stays in the single-vacancy state.

The state analysis of defect_states.py attributes a hop to the state at the START of its window, so that the transit is
not mistaken for a defect.  That is the right convention for a long-lived defect, but it puts into the
single-vacancy count every hop whose transit passes through a fused lithium pair -- two ions within 1.4 A,
a configuration the equilibrium ensemble never reaches (its minimum lithium-lithium distance is 1.95 A) and
that density functional theory rejects.  Here a hop is called clean when no frame of its reactive segment fuses.
    python clean_channel.py FILE.npz [FILE2.npz ...]
"""
import sys, numpy as np
from analyze_hops import ideal, R_SWITCH, S_A, S_B

FUSE = 1.4

def analyse(path):
    d = np.load(path); X = d["frames"].astype(np.float64)
    st, rp, Z, L, S = d["frame_step"], d["frame_replica"], d["numbers"], d["cell"], d["sites"]
    pos0, num0 = ideal(); li, fw = np.where(Z == 3)[0], np.where(Z != 3)[0]; ref = pos0[fw].mean(0)
    nrep = int(rp.max()) + 1; tot = float(d["ns_per_replica"]) / 2e-6
    dead = d["dead"] if "dead" in d.files else np.full(nrep, -1)
    GAP = int(np.diff(np.sort(np.unique(st))).min())
    n_all = n_clean = 0; T_sv = 0.0
    for b in range(nrep):
        end = tot if dead[b] < 0 else dead[b]
        idx = np.where(rp == b)[0]
        if len(idx) < 2: T_sv += end; continue
        idx = idx[np.argsort(st[idx])]; t = st[idx]; x = X[idx]
        y = x[:, li] - (x[:, fw].mean(1) - ref)[:, None]
        dd = y[:, :, None, :] - S[None, None]; dd -= L * np.round(dd / L)
        r = np.linalg.norm(dd, axis=-1); a = r.argmin(-1); rmin = r.min(-1)
        p = y[:, :, None, :] - y[:, None, :, :]; p -= L * np.round(p / L)
        dmin = (np.linalg.norm(p, axis=-1) + np.eye(len(li))[None] * 9).min((1, 2))          # closest lithium pair, per frame
        ws = np.r_[0, np.where(np.diff(t) > GAP)[0] + 1]
        sv_state = np.array([(np.bincount(a[k], minlength=len(S)) >= 2).sum() == 0 for k in ws])
        ed = np.minimum(np.r_[0, t[ws[1:]], end], end)
        T_sv += sum(max(ed[k + 1] - ed[k], 0) for k in range(len(ws)) if sv_state[k])
        cur = a[0].copy()
        for k in range(1, len(idx)):
            gap = t[k] - t[k - 1] > GAP
            if gap: cur = np.where(rmin[k] < R_SWITCH, a[k], cur)
            for i in np.where((a[k] != cur) & (rmin[k] < R_SWITCH) & ~gap)[0]:
                Ri, Rj = S[cur[i]], S[a[k, i]]; e = Rj - Ri; e -= L * np.round(e / L); d2 = (e ** 2).sum()
                if abs(np.sqrt(d2) - L[0] / 3 / np.sqrt(2)) > 0.05: continue
                u = y[:, i] - Ri; u -= L * np.round(u / L); sc = u @ e / d2
                j0 = k
                while j0 > 0 and sc[j0] > S_A and t[j0] - t[j0 - 1] <= GAP: j0 -= 1
                j2 = j0
                while j2 < len(idx) - 1 and sc[j2] < S_B and t[j2 + 1] - t[j2] <= GAP: j2 += 1
                if not (sc[j0] <= S_A and sc[j2] >= S_B): continue
                w = np.searchsorted(t[ws], t[j0], side="right") - 1
                if not sv_state[max(w, 0)]: continue                                          # not the single-vacancy state to begin with
                n_all += 1; n_clean += dmin[j0:j2 + 1].min() >= FUSE
            cur[np.where((a[k] != cur) & (rmin[k] < R_SWITCH) & ~gap)[0]] = a[k, np.where((a[k] != cur) & (rmin[k] < R_SWITCH) & ~gap)[0]]
    ns = T_sv * 2e-6
    return n_all, n_clean, ns

print(f"{'run':22s} {'ns(1-vac)':>10s} {'hops':>6s} {'clean':>6s} {'fused %':>8s} {'rate all':>10s} {'rate clean':>11s}")
out = {}
for f in sys.argv[1:]:
    na, nc, ns = analyse(f); name = f.split("/")[-1].replace(".npz", "")
    out[name] = (na, nc, ns)
    print(f"{name:22s} {ns:10.3f} {na:6d} {nc:6d} {100*(1-nc/max(na,1)):8.1f} {na/max(ns,1e-9):10.1f} {nc/max(ns,1e-9):11.1f}", flush=True)
np.save("../data/li3ocl_clean_channel.npy", out, allow_pickle=True)
