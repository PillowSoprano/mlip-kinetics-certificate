"""
Batched Langevin MD for a MACE model: B independent replicas of the same crystal advanced with ONE model call per step.
Everything (neighbour list, model, integrator, hop bookkeeping) lives on the device; no ASE calculator in the loop.

Assumptions: fixed orthorhombic cell with every edge > 2 * (cutoff + skin), so the minimum-image convention gives the full neighbour list.
Units are ASE units (eV, Angstrom, amu; time in ASE units = fs * ase.units.fs).

    eng = BatchedMD(model, numbers, cell_lengths, positions[B, N, 3], temperature_K, dt_fs=2.0, friction_per_fs=0.01)
    eng.run(n_steps)                     # BAOAB Langevin
    e, f = eng.energy_forces()           # eV, eV/Angstrom   (for validation against MACECalculator)
"""
import numpy as np, torch
from ase import units
from ase.data import atomic_masses


class BatchedMD:
    def __init__(self, model, numbers, cell_lengths, positions, temperature_K, dt_fs=2.0, friction_per_fs=0.01, skin=0.6, rebuild_every=10,
                 device="cpu", dtype=torch.float32, seed=0, remove_com=True):
        self.dev, self.dt_ = torch.device(device), dtype
        self.model = model.to(self.dev).to(dtype).eval()
        for p in self.model.parameters(): p.requires_grad_(False)
        self.B, self.N = positions.shape[:2]
        self.L = torch.as_tensor(np.asarray(cell_lengths), dtype=dtype, device=self.dev)
        self.rc = float(model.r_max); self.skin, self.rebuild_every = skin, rebuild_every
        assert float(self.L.min()) > 2 * (self.rc + skin), "cell too small for the minimum-image neighbour list"
        zs = [int(z) for z in model.atomic_numbers]; idx = torch.tensor([zs.index(int(z)) for z in numbers])
        self.node_attrs = torch.nn.functional.one_hot(idx, len(zs)).to(dtype).repeat(self.B, 1).to(self.dev)
        self.batch = torch.arange(self.B, device=self.dev).repeat_interleave(self.N)
        self.ptr = torch.arange(0, (self.B + 1) * self.N, self.N, device=self.dev)
        self.cell = torch.diag(self.L).repeat(self.B, 1, 1).reshape(-1, 3)
        self.x = torch.as_tensor(positions, dtype=dtype, device=self.dev).clone()          # unwrapped positions
        m = torch.as_tensor(atomic_masses[np.asarray(numbers)], dtype=dtype, device=self.dev)
        self.m = m[None, :, None]
        self.kT = units.kB * temperature_K; self.dt = dt_fs * units.fs; gam = friction_per_fs / units.fs
        self.c1 = float(np.exp(-gam * self.dt)); self.c2 = float(np.sqrt(1 - self.c1**2))
        self.gen = torch.Generator(device=self.dev); self.gen.manual_seed(seed); self.remove_com = remove_com
        self.v = torch.sqrt(self.kT / self.m) * torch.randn(self.x.shape, generator=self.gen, device=self.dev, dtype=dtype)
        self._zero_com(); self.step_count = 0; self._build_neighbours(); self.e, self.f = self.energy_forces()

    # ---------------------------------------------------------------- neighbour list (minimum image, all replicas at once)
    def _build_neighbours(self):
        d = self.x[:, None, :, :] - self.x[:, :, None, :]                                # d[b, i, j] = x_j - x_i
        s = -torch.round(d / self.L)                                                     # unit shift so that x_j - x_i + s L is the minimum image
        r2 = ((d + s * self.L) ** 2).sum(-1)
        mask = (r2 < (self.rc + self.skin) ** 2) & ~torch.eye(self.N, dtype=torch.bool, device=self.dev)[None]
        b, i, j = mask.nonzero(as_tuple=True)
        self.edge_index = torch.stack([b * self.N + i, b * self.N + j])                   # sender i, receiver j
        self.unit_shifts = s[b, i, j]; self.x_ref = self.x.clone()

    def _zero_com(self):
        # Langevin noise does not conserve momentum: without this the whole crystal diffuses rigidly (2-4 A in 20 ps for 134 atoms at 1000 K)
        if self.remove_com: self.v = self.v - (self.m * self.v).sum(1, keepdim=True) / self.m.sum(1, keepdim=True)

    def _needs_rebuild(self):
        return bool((((self.x - self.x_ref) ** 2).sum(-1).max()) > (0.5 * self.skin) ** 2)

    # ---------------------------------------------------------------- one model call for all replicas
    def energy_forces(self):
        pos = self.x.reshape(-1, 3)
        data = {"positions": pos, "node_attrs": self.node_attrs, "edge_index": self.edge_index, "unit_shifts": self.unit_shifts,
                "shifts": self.unit_shifts * self.L, "cell": self.cell, "batch": self.batch, "ptr": self.ptr,
                "head": torch.zeros(self.B, dtype=torch.long, device=self.dev)}
        out = self.model(data, compute_force=True, training=False)
        return out["energy"].detach(), out["forces"].detach().reshape(self.B, self.N, 3)

    # ---------------------------------------------------------------- BAOAB Langevin
    def run(self, n_steps, callback=None, callback_every=10):
        h = self.dt
        for _ in range(n_steps):
            self.v = self.v + 0.5 * h * self.f / self.m
            self.x = self.x + 0.5 * h * self.v
            self.v = self.c1 * self.v + self.c2 * torch.sqrt(self.kT / self.m) * torch.randn(self.x.shape, generator=self.gen, device=self.dev, dtype=self.dt_)
            self._zero_com()
            self.x = self.x + 0.5 * h * self.v
            self.step_count += 1
            if self.step_count % self.rebuild_every == 0 and self._needs_rebuild():
                self._build_neighbours()
            self.e, self.f = self.energy_forces()
            self.v = self.v + 0.5 * h * self.f / self.m
            if callback is not None and self.step_count % callback_every == 0:
                callback(self)

    def temperature(self):
        return ((self.m * self.v**2).sum((1, 2)) / (3 * self.N * units.kB)).cpu().numpy()


class HopCounter:
    """Assign every mobile ion to its nearest lattice site (with hysteresis) and count site changes; keep unwrapped MSD and mid-hop frames."""
    def __init__(self, eng, mobile_idx, sites, r_switch=0.9, r_mid=1.2, keep_frames=True, window=25, framework_idx=None, framework_ref=None):
        # sites are fixed in space, so ions are located RELATIVE TO THE FRAMEWORK: subtract its rigid displacement since t = 0
        self.fw = None if framework_idx is None else torch.as_tensor(framework_idx, device=eng.dev)
        if self.fw is None: self.fw0 = None
        elif framework_ref is not None: self.fw0 = torch.as_tensor(np.asarray(framework_ref), dtype=eng.dt_, device=eng.dev)[None, None]   # mean framework position of the IDEAL crystal
        else: self.fw0 = eng.x[:, self.fw].mean(1, keepdim=True).clone()
        self.idx = torch.as_tensor(mobile_idx, device=eng.dev); self.sites = torch.as_tensor(sites, dtype=eng.dt_, device=eng.dev)
        self.r_switch, self.r_mid, self.keep = r_switch, r_mid, keep_frames
        self.x0 = self._mobile(eng).clone(); self.assign, _ = self._nearest(eng)
        self.hops = torch.zeros(eng.B, dtype=torch.long, device=eng.dev); self.msd = []; self.frames = []; self.events = []
        self.window, self.buf = window, []; self.saved_to = np.full(eng.B, -1, dtype=np.int64); self.rec_until = np.full(eng.B, -1, dtype=np.int64)

    def _mobile(self, eng):
        x = eng.x[:, self.idx]
        return x if self.fw is None else x - (eng.x[:, self.fw].mean(1, keepdim=True) - self.fw0)

    def _nearest(self, eng):
        d = self._mobile(eng)[:, :, None, :] - self.sites[None, None]; d = d - eng.L * torch.round(d / eng.L); r = d.norm(dim=-1); rmin, a = r.min(-1)
        return a, rmin

    def __call__(self, eng):
        a, r = self._nearest(eng); sw = (a != self.assign) & (r < self.r_switch)
        if sw.any():
            b, i = sw.nonzero(as_tuple=True)
            for bb, ii in zip(b.tolist(), i.tolist()):
                self.events.append((eng.step_count, bb, ii, int(self.assign[bb, ii]), int(a[bb, ii])))
        self.hops += sw.sum(1); self.assign = torch.where(sw, a, self.assign)
        self.msd.append(((self._mobile(eng) - self.x0) ** 2).sum(-1).mean(1).cpu().numpy())
        if self.keep:
            # complete hop paths: a rolling buffer of the last `window` callbacks; when a replica has an ion between sites (or a hop), dump its
            # buffer and keep recording it for another `window` callbacks.  Frames are (step, replica, positions relative to the framework).
            xs = (eng.x if self.fw is None else eng.x - (eng.x[:, self.fw].mean(1, keepdim=True) - self.fw0)).cpu().numpy().astype(np.float32)
            self.buf.append((eng.step_count, xs)); self.buf = self.buf[-self.window:]
            trig = ((r > self.r_mid).any(1) | sw.any(1)).cpu().numpy()
            for bb in np.nonzero(trig)[0]:
                for st, xb in self.buf:
                    if st > self.saved_to[bb]: self.frames.append((st, int(bb), xb[bb])); self.saved_to[bb] = st
                self.rec_until[bb] = len(self.msd) + self.window
            for bb in np.nonzero((self.rec_until >= len(self.msd)) & ~trig)[0]:
                if eng.step_count > self.saved_to[bb]: self.frames.append((eng.step_count, int(bb), xs[bb])); self.saved_to[bb] = eng.step_count
