"""
Hop analysis of batched Li3OCl runs (protocol: Sec. VI C of the paper).

    python analyze_hops.py RESULT.npz [TAG]

Reads the frames stored around hops, locates ions relative to the framework (idempotent: works for drift-corrected and raw frames), and produces
  - hop events with hysteresis (r_switch = 0.9 A), rate per ns and variance/mean of the counts per replica
  - for every hop the reactive segment in the progress coordinate s = (x - R_i).(R_j - R_i)/d^2, from the last exit of A = {s < 0.25} to the first entry of B = {s > 0.75}
  - F(s) from all hop segments' surroundings, the 1D committor q(s) ~ int exp(F), committor increments dq per frame (path representation of nu)
Output: ../data/li3ocl_hops_TAG.npz  (frame index into the input, hop id, s, q, dq)  +  printed summary.
"""
import sys, numpy as np

R_SWITCH, S_A, S_B, NBIN = 0.9, 0.25, 0.75, 40

def ideal(a=3.926):
    frac = [(0, 0, 0), (.5, .5, .5), (.5, .5, 0), (.5, 0, .5), (0, .5, .5)]; zz = [17, 8, 3, 3, 3]; pos, num = [], []
    for ix in range(3):
        for iy in range(3):
            for iz in range(3):
                for f, z in zip(frac, zz): pos.append((np.array(f) + [ix, iy, iz]) * a); num.append(z)
    pos, num = np.array(pos), np.array(num); first = np.where(num == 3)[0][0]
    return np.delete(pos, first, 0), np.delete(num, first)

def main(path, tag):
    d = np.load(path); X = d["frames"].astype(np.float64); st, rp, Z, L, S = d["frame_step"], d["frame_replica"], d["numbers"], d["cell"], d["sites"]
    pos0, num0 = ideal(); assert (num0 == Z).all(); li, fw = np.where(Z == 3)[0], np.where(Z != 3)[0]; ref = pos0[fw].mean(0)
    nrep = int(rp.max()) + 1; ns = float(d["ns_per_replica"]) if "ns_per_replica" in d.files else st.max() * 2e-6
    hops = np.zeros(nrep, int); seg = []; s_all = []; pre = []                                       # seg: (frame indices, s values) per hop
    for b in range(nrep):
        idx = np.where(rp == b)[0]; idx = idx[np.argsort(st[idx])]
        if len(idx) < 2: continue
        x = X[idx]; GAP = int(np.diff(np.sort(np.unique(st))).min()); y = x[:, li] - (x[:, fw].mean(1) - ref)[:, None]; t = st[idx]
        dd = y[:, :, None, :] - S[None, None]; dd -= L * np.round(dd / L); r = np.linalg.norm(dd, axis=-1); a = r.argmin(-1); rmin = r.min(-1)
        cur = a[0].copy()
        for k in range(1, len(idx)):
            gap = t[k] - t[k - 1] > GAP                                                   # frames are stored in windows; re-synchronise after a gap
            if gap: cur = np.where(rmin[k] < R_SWITCH, a[k], cur)
            sw = np.where((a[k] != cur) & (rmin[k] < R_SWITCH) & ~gap)[0]
            for i in sw:
                Ri, Rj = S[cur[i]], S[a[k, i]]; e = Rj - Ri; e -= L * np.round(e / L); d2 = (e ** 2).sum()
                if abs(np.sqrt(d2) - L[0] / 3 / np.sqrt(2)) > 0.05: print(f"  replica {b} step {t[k]}: hop of length {np.sqrt(d2):.2f} A is not nearest-neighbour"); continue
                u = y[:, i] - Ri; u -= L * np.round(u / L); s = u @ e / d2
                j1 = k
                while j1 > 0 and s[j1] > S_A and t[j1] - t[j1 - 1] <= GAP: j1 -= 1         # back to the last frame in A
                j0 = j1; j2 = j1
                while j2 < len(idx) - 1 and s[j2] < S_B and t[j2 + 1] - t[j2] <= GAP: j2 += 1
                npre = int(round(100 / GAP)); jp = j0 - npre                                 # basin frame 0.2 ps (100 steps) before the segment, same replica
                ok_pre = jp >= 0 and t[j0] - t[jp] == npre * GAP
                if s[j0] <= S_A and s[j2] >= S_B: seg.append((idx[j0:j2 + 1], s[j0:j2 + 1])); pre.append(idx[jp] if ok_pre else -1); s_all.append(s[max(0, j0 - 10):j2 + 11])
                hops[b] += 1
            cur[sw] = a[k, sw]
    n = hops.sum(); print(f"{tag}: {n} hops in {nrep} x {ns * 1e3:.1f} ps  ->  {n / (nrep * ns):.0f} +- {np.sqrt(max(n, 1)) / (nrep * ns):.0f} per ns;  variance/mean per replica {hops.var() / max(hops.mean(), 1e-9):.2f};  complete A->B segments {len(seg)}")
    if not seg: return
    DT = int(np.diff(np.sort(np.unique(st))).min()) * 2e-3; dur = np.array([len(s_) for _, s_ in seg]) * DT; print(f"  reactive segment duration: median {np.median(dur):.2f} ps, 90% below {np.quantile(dur, .9):.2f} ps;  frames per segment: median {int(np.median(dur) / DT)}")
    # F(s), step 2 of the protocol: histogram in [S_A, S_B] of ALL stored frames of hopping ions around hops (windows)
    sa = np.concatenate(s_all); edges = np.linspace(S_A, S_B, NBIN + 1); h, _ = np.histogram(sa, edges); F = -np.log(np.maximum(h, 0.5)); F -= F.min()
    qe = np.r_[0, np.cumsum(np.exp(F))]; qe /= qe[-1]; qf = lambda s: np.interp(np.clip(s, S_A, S_B), edges, qe)
    print(f"  barrier of F(s) relative to the core edge: {F.max() - min(F[0], F[-1]):.2f} kT (windowed sampling: a lower bound on the true barrier from the site minimum);  symmetry check F[0]-F[-1] = {F[0] - F[-1]:+.2f}")
    fi, hid, sv, qv, dq = [], [], [], [], []
    for m, (ii, s_) in enumerate(seg):
        q = qf(s_); g = 0.5 * (np.r_[q[1:], q[-1]] - np.r_[q[0], q[:-1]]); fi += ii.tolist(); hid += [m] * len(ii); sv += s_.tolist(); qv += q.tolist(); dq += g.tolist()
    dq = np.array(dq); sv = np.array(sv); w = np.abs(dq) / np.abs(dq).sum()
    print(f"  leverage density in s: mean {np.sum(w * sv):.3f}, sd {np.sqrt(np.sum(w * (sv - np.sum(w * sv)) ** 2)):.3f};  net dq per segment {dq.sum() / len(seg):.3f} (should be 1);  effective frames {1 / np.sum(w ** 2):.0f} of {len(w)}")
    np.savez_compressed(f"../data/li3ocl_hops_{tag}.npz", frame=np.array(fi), hop=np.array(hid), s=sv, q=np.array(qv), dq=dq, F=F, edges=edges, pre_frame=np.array(pre), hops_per_replica=hops, ns_per_replica=ns)

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "run")
