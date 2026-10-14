"""
Representative configurations for Figures 4 and 5, drawn as TikZ ball-and-stick pictures.

    python structures.py        (run from figs/; reads ../ala2 and ../colab, ../data)

Alanine dipeptide: a teacher frame and a student (A2) frame on reactive paths at phi ~ 0, the first with a
planar peptide bond, the second on the distorted-peptide pathway. Li3OCl: a teacher frame half way through a
single-vacancy hop, and a teacher frame with two lithium ions on one site. Every frame is a stored frame of the
production runs; nothing is relaxed or edited. The chosen frames are kept in data/struct_frames_ala.npz and
data/struct_frames_li.npz; when these exist the pictures are redrawn from them without the raw trajectories.
Output: data/struct_*.tex (TikZ commands) and data/struct_values.txt.
"""
import glob, os, sys, numpy as np
VIEW = [float(v) for v in os.environ.get("ALA_VIEW", "18,30").split(",")]

OUT = "data"
def dihedral(x, idx):
    p0, p1, p2, p3 = x[idx]; b0, b1, b2 = p0 - p1, p2 - p1, p3 - p2; b1n = b1 / np.linalg.norm(b1)
    v, w = b0 - b0.dot(b1n) * b1n, b2 - b2.dot(b1n) * b1n
    return np.degrees(np.arctan2(np.cross(b1n, v).dot(w), v.dot(w)))

def kabsch(P, Q):  # rotation R with P @ R ~ Q (both centred)
    U, _, Vt = np.linalg.svd(P.T @ Q); d = np.sign(np.linalg.det(U @ Vt)); D = np.diag([1, 1, d]); return U @ D @ Vt

def rot(axis, deg):
    t = np.radians(deg); c, s = np.cos(t), np.sin(t)
    return {"x": np.array([[1, 0, 0], [0, c, -s], [0, s, c]]), "y": np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])}[axis]

def tikz(atoms, bonds, scale, extra_back=(), extra_front=()):
    """atoms: list of (xyz, radius, fill, opacity); bonds: list of (i, j, width, colour). Painter's order by depth z.
    Flat atoms in the palette of Figure 1: a fill, a darker rim and a small highlight."""
    items = []
    for i, j, w, col in bonds:
        a, b = atoms[i][0], atoms[j][0]; items.append(((a[2] + b[2]) / 2 - 0.05, "b", (a, b, w, col)))
    for a in atoms: items.append((a[0][2], "a", a))
    items.sort(key=lambda t: t[0]); L = list(extra_back)
    for _, kind, it in items:
        if kind == "b":
            a, b, w, col = it; L.append(f"\\draw[line width={w}pt,{col},line cap=round] ({a[0]*scale:.3f},{a[1]*scale:.3f}) -- ({b[0]*scale:.3f},{b[1]*scale:.3f});")
        else:
            p, r, col, op = it; x, y, rr = p[0] * scale, p[1] * scale, r * scale
            rim = "mute" if col == "white" else f"{col}!70!black"
            L.append(f"\\filldraw[fill={col},draw={rim},line width=0.35pt,opacity={op}] ({x:.3f},{y:.3f}) circle ({rr:.3f});")
            L.append(f"\\fill[white,opacity={0.35*op:.2f}] ({x-0.38*rr:.3f},{y+0.38*rr:.3f}) circle ({0.30*rr:.3f});")
    return "\n".join(L + list(extra_front)) + "\n"

# ------------------------------------------------------------------ alanine dipeptide
EL = ["H", "C", "H", "H", "C", "O", "N", "H", "C", "H", "C", "H", "H", "H", "C", "O", "N", "H", "C", "H", "H", "H"]
RAD = {"H": 0.22, "C": 0.38, "N": 0.36, "O": 0.36}; COL = {"H": "white", "C": "axisgrey", "N": "skydeep", "O": "tomato"}
PHI, OM1 = [4, 6, 8, 14], [1, 4, 6, 8]

def ala_frames():
    T = []
    for f in sorted(glob.glob("../ala2/traj_tsx_s*.npz"))[:2]:
        d = np.load(f); phi = np.degrees(d["phipsi"][d["ts_idx"], 0]); k = np.where(np.abs(phi) < 3)[0]
        T += [d["ts_xyz"][i] * 10 for i in k[:: max(1, len(k) // 400)]]
    S = []
    for f in sorted(glob.glob("../ala2/traj_stutsx-A2_s*.npz")):
        d = np.load(f); phi = np.degrees(d["phipsi"][d["ts_idx"], 0]); k = np.where(np.abs(phi) < 3)[0]
        S += [d["ts_xyz"][i] * 10 for i in k]
    om_t = np.array([abs(dihedral(x, OM1)) for x in T]); om_s = np.array([abs(dihedral(x, OM1)) for x in S])
    t = T[int(np.argmin(np.abs(om_t - np.median(om_t))))]                       # typical teacher frame at phi ~ 0
    dist = om_s[om_s < 120]; s = S[int(np.argmin(np.abs(om_s - np.median(dist))))]  # typical distorted student frame
    return t, s, np.median(om_t), np.median(dist), (om_s < 120).mean()

def ala_draw(x, R, c0):
    y = (x - c0) @ R; atoms = [(y[i], RAD[e], COL[e], 1) for i, e in enumerate(EL)]; bonds = []
    for i in range(22):
        for j in range(i + 1, 22):
            cut = 1.25 if "H" in (EL[i], EL[j]) else 1.75
            if np.linalg.norm(x[i] - x[j]) < cut:
                hot = {i, j} in ({4, 6},)
                bonds.append((i, j, 3.4 if hot else 1.5, "amber" if hot else "mute"))
    return atoms, bonds

def alanine(vals):
    cache = f"{OUT}/struct_frames_ala.npz"                                         # the two chosen frames, so the view can be redrawn quickly
    if os.path.exists(cache): c = np.load(cache); t, s, om_t, om_s, frac = c["t"], c["s"], float(c["om_t"]), float(c["om_s"]), float(c["frac"])
    else: t, s, om_t, om_s, frac = ala_frames(); np.savez(cache, t=t, s=s, om_t=om_t, om_s=om_s, frac=frac)
    core = [6, 8, 10, 14]
    ct, cs = t[core].mean(0), s[core].mean(0); Rk = kabsch(s[core] - cs, t[core] - ct)
    s_al = (s - cs) @ Rk + ct                                                       # student frame on the teacher's central residue
    heavy = [i for i, e in enumerate(EL) if e != "H"]; c0 = t[heavy].mean(0)
    _, _, Vt = np.linalg.svd(t[heavy] - c0); R = Vt.T @ rot("x", VIEW[0]) @ rot("y", VIEW[1])
    if (((t[6] - c0) @ R)[0] > ((t[14] - c0) @ R)[0]): R = R @ np.diag([-1, 1, -1])  # N terminus on the left
    for name, x in (("teacher", t), ("student", s_al)):
        a, b = ala_draw(x, R, c0); open(f"{OUT}/struct_ala_{name}.tex", "w").write(tikz(a, b, 0.62))
    vals += [f"alanine teacher frame at phi~0: |omega1| = {abs(dihedral(t, OM1)):.0f} deg (median of teacher frames {om_t:.0f})",
             f"alanine student A2 frame at phi~0: |omega1| = {abs(dihedral(s, OM1)):.0f} deg (median of distorted frames {om_s:.0f}; {100*frac:.0f}% of A2 frames at phi~0 have |omega1|<120)"]

# ------------------------------------------------------------------ Li3OCl
A = 3.926
LCOL = {3: "white", 8: "slate", 17: "skypale"}; LRAD = {3: 0.42, 8: 0.58, 17: 0.80}

def li_load():
    d = np.load("../colab/li3ocl_teacher_1000K_censored.npz"); X, Z, L, S = d["frames"], d["numbers"], d["cell"], d["sites"]
    li, fw = np.where(Z == 3)[0], np.where(Z != 3)[0]
    ideal = np.load("../data/li3ocl_hops_T1000.npz")
    return X, Z, L, S, li, fw, ideal

def wrap(v, L): return v - L * np.round(v / L)

def li_scene(y, Z, L, S, center, axis, hot, empty, tag, note=()):
    """Slice through `center` normal to cubic `axis`: Li and O within 1.3 A of the plane, Cl of the plane a/2 behind."""
    ax = "xyz".index(axis); uv = [i for i in range(3) if i != ax]; atoms = []; idx = []
    for k in range(len(Z)):
        r = wrap(y[k] - center, L); dep = r[ax]
        inplane = abs(dep) < 1.3 and Z[k] in (3, 8); behind = Z[k] == 17 and abs(dep + A / 2) < 1.0
        if (inplane or behind) and abs(r[uv[0]]) < 1.3 * A and abs(r[uv[1]]) < 1.0 * A:
            col = LCOL[Z[k]]; op = 1 if inplane else 0.9
            if k in hot: col = hot[k]
            atoms.append((np.array([r[uv[0]], r[uv[1]], dep if inplane else dep - 2]), LRAD[Z[k]], col, op)); idx.append(k)
    bonds = []
    for a in range(len(idx)):
        for b in range(a + 1, len(idx)):
            if {Z[idx[a]], Z[idx[b]]} == {3, 8} and np.linalg.norm(wrap(y[idx[a]] - y[idx[b]], L)) < 2.35 and atoms[a][3] == 1 and atoms[b][3] == 1:
                bonds.append((a, b, 0.8, "mute!70"))
    sc = 0.34; back = []
    for e in empty:
        r = wrap(S[e] - center, L)
        if abs(r[ax]) < 1.3: back.append(f"\\draw[densely dashed,line width=0.7pt,slate] ({r[uv[0]]*sc:.3f},{r[uv[1]]*sc:.3f}) circle ({0.42*sc:.3f});")
    open(f"{OUT}/struct_li_{tag}.tex", "w").write(tikz(atoms, bonds, sc, back, note))

def li_select():
    """Pick the two teacher frames from the stored hop windows (needs the raw run); returns drift-corrected frames."""
    X, Z, L, S, li, fw, H = li_load(); X = X.astype(float)
    sys.path.insert(0, "../li3ocl"); from analyze_hops import ideal
    ref = ideal()[0][fw].mean(0); X = X - (X[:, fw].mean(1) - ref)[:, None]   # positions relative to the framework, as in analyze_hops
    def minLiLi(x):
        p = x[li]; dd = wrap(p[:, None] - p[None], L); r = np.linalg.norm(dd, axis=-1); r[np.diag_indices(len(li))] = 9; return r
    fr, sv = H["frame"], H["s"]; best = None
    for k in np.argsort(np.abs(sv - 0.5))[:200]:                                 # half way through a hop, no close Li pair in the cell
        if minLiLi(X[fr[k]]).min() > 2.0: best = int(fr[k]); break
    mins = np.array([minLiLi(X[k]).min() for k in range(0, len(X), 7)]); cand = np.arange(0, len(X), 7)[np.argsort(mins)]
    for kf in cand:                                                               # two Li 0.5-1 A apart
        if 0.5 < minLiLi(X[kf]).min() < 1.0: break
    return dict(x_hop=X[best], i_hop=best, x_fused=X[kf], i_fused=int(kf), Z=Z, L=L, S=S)

def lithium(vals):
    cache = f"{OUT}/struct_frames_li.npz"                                          # the two chosen frames, so the pictures can be redrawn
    if os.path.exists(cache): c = dict(np.load(cache))
    else: c = li_select(); np.savez(cache, **c)
    Z, L, S = c["Z"], c["L"], c["S"]; li = np.where(Z == 3)[0]
    def near(x):
        dd = wrap(x[li][:, None] - S[None], L); return np.linalg.norm(dd, axis=-1)
    def minLiLi(x):
        p = x[li]; dd = wrap(p[:, None] - p[None], L); r = np.linalg.norm(dd, axis=-1); r[np.diag_indices(len(li))] = 9; return r
    x = c["x_hop"]; r = near(x); hopper = int(np.argmax(r.min(1))); two = np.argsort(r[hopper])[:2]
    center = (S[two[0]] + wrap(S[two[1]] - S[two[0]], L) / 2); e = wrap(S[two[1]] - S[two[0]], L)
    axis = "xyz"[int(np.argmin(np.abs(e)))]
    li_scene(x, Z, L, S, center, axis, {li[hopper]: "amber"}, [int(t) for t in two], "hop")
    vals.append(f"Li3OCl hop frame {int(c['i_hop'])}: hopping ion {r[hopper].min():.2f} A from the nearest site; closest Li-Li {minLiLi(x).min():.2f} A")
    x = c["x_fused"]; m = minLiLi(x); i, j = np.unravel_index(np.argmin(m), m.shape)
    r = near(x); site = int(np.argmin(r[i] + r[j])); occ = (r < 1.2).any(0)
    empty = [e for e in range(len(S)) if not occ[e]]
    dists = [np.linalg.norm(wrap(S[e] - S[site], L)) for e in empty]
    e0 = empty[int(np.argmin(dists))]; dv = wrap(S[e0] - S[site], L); axis = "xyz"[int(np.argmin(np.abs(dv)))]   # slice containing the nearest empty site
    center = S[site] + dv / 2
    li_scene(x, Z, L, S, center, axis, {li[i]: "tomato", li[j]: "tomato"}, empty, "fused")
    vals.append(f"Li3OCl fused frame {int(c['i_fused'])}: Li-Li {m[i, j]:.2f} A on site {site}; empty sites at {', '.join(f'{d:.2f}' for d in sorted(dists))} A from it")

vals = []
if "li" not in sys.argv[1:]: alanine(vals)
if "ala" not in sys.argv[1:]: lithium(vals)
prev = open(f"{OUT}/struct_values.txt").read().splitlines() if os.path.exists(f"{OUT}/struct_values.txt") else []
keep = [l for l in prev if not any(l.startswith(v.split()[0]) for v in vals)]
open(f"{OUT}/struct_values.txt", "w").write("\n".join(keep + vals) + "\n"); print("\n".join(vals))
