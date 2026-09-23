"""Smooth two-dimensional matched-distribution experiment.

Physical domain: [-1,1] x periodic [0,1], V=6(1-x^2)^2+2(1-cos(4*pi*y)).
Solve in s=q(x), eta=CDF_mu_y(y); reference leverage is uniform in both.
The tensor is the coordinate transform of physical mobility diag(1,r).
Finite-volume conductances use midpoint quadrature. beta=1.
"""
from pathlib import Path
import csv
import json
import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.sparse import coo_matrix, diags
from scipy.sparse.linalg import spsolve

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'research_extensions' / 'results'


def bump(s):
    result = np.zeros_like(s)
    for centre in (0.25, 0.75):
        z = (s - centre) / 0.225
        mask = abs(z) < 1
        result[mask] += np.exp(1 - 1 / (1 - z[mask]**2))
    return result


def field(s, eta, winding):
    return bump(s) * np.cos(2 * np.pi * (eta + winding * s))


def problem(ns, ny, mobility, winding):
    # High-resolution one-dimensional quadrature only constructs the coordinates.
    x = np.linspace(-1, 1, 40001)
    u = 6 * (1 - x*x)**2
    integral = cumulative_trapezoid(np.exp(u), x, initial=0)
    resistance = integral[-1]
    sx = integral / resistance
    y = np.linspace(0, 1, 40001)
    w = 2 * (1 - np.cos(4 * np.pi * y))
    integral_y = cumulative_trapezoid(np.exp(-w), y, initial=0)
    partition_y = integral_y[-1]
    eta_y = integral_y / partition_y
    s = np.linspace(0, 1, ns)
    eta = np.arange(ny) / ny
    ds, de = s[1], 1 / ny
    se, en = (s[:-1] + s[1:]) / 2, (eta + de / 2) % 1
    x_of_s = np.interp(s, sx, x)
    y_of_eta = np.interp(en, eta_y, y)
    u_s = 6 * (1 - x_of_s**2)**2
    w_eta = 2 * (1 - np.cos(4 * np.pi * y_of_eta))
    # Divide all conductances by the fixed axial coefficient Z_y/I_x.
    ax = np.ones((ns - 1, ny)) * de / ds
    transverse = mobility * (resistance / partition_y)**2 * np.exp(
        -2 * u_s[:, None] - 2 * w_eta[None, :]) * ds / de
    nodes = np.arange(ns * ny).reshape(ns, ny)
    tail = np.r_[nodes[:-1].ravel(), nodes.ravel()]
    head = np.r_[nodes[1:].ravel(), np.roll(nodes, -1, axis=1).ravel()]
    count = len(tail)
    incidence = coo_matrix((np.r_[-np.ones(count), np.ones(count)],
                           (np.r_[np.arange(count), np.arange(count)],
                            np.r_[tail, head])), shape=(count, ns * ny)).tocsr()
    c = np.r_[ax.ravel(), transverse.ravel()]
    f = np.r_[field(se[:, None], eta[None, :], winding).ravel(),
              field(s[:, None], en[None, :], winding).ravel()]
    interior = nodes[1:-1].ravel()
    q = np.repeat(s, ny)
    dq = incidence @ q
    cap = float(np.sum(c * dq*dq))
    nu = c * dq*dq / cap
    lap = incidence.T @ diags(c) @ incidence
    rhs = incidence.T @ (c * f * dq)
    qdot = np.zeros_like(q)
    qdot[interior] = spsolve(lap[interior][:, interior], rhs[interior])
    mean = float(nu @ f)
    variance = float(nu @ (f - mean)**2)
    relax = float(np.sum(c * (incidence @ qdot)**2) / cap)
    chi = relax / variance
    curvature = variance - 2 * relax
    current = c * dq / cap
    current_dot = c * ((incidence @ qdot) - (f - mean) * dq) / cap
    divergence = incidence.T @ current_dot
    assert np.max(abs(divergence[interior])) < 1e-8
    assert abs(divergence[nodes[0]].sum()) < 1e-8

    def evaluate(t):
        ct = c * np.exp(-t * f)
        lt = incidence.T @ diags(ct) @ incidence
        qt = np.zeros_like(q)
        qt[nodes[-1]] = 1
        qt[interior] = spsolve(lt[interior][:, interior],
                               -lt[interior][:, nodes[-1]] @ np.ones(ny))
        dqt = incidence @ qt
        capt = float(np.sum(ct * dqt*dqt))
        jt = ct * dqt / capt
        exact = float(np.log(capt / cap))
        lower = float(-np.log(nu @ np.exp(t * f)))
        upper = float(np.log(nu @ np.exp(-t * f)))
        # Flow-coordinate series/parallel refinement, with exact grid trial
        # fields for this separable reference problem.
        axial_f = f[:ax.size].reshape(ax.shape)
        series_upper = float(-np.log(np.mean(1 / np.mean(np.exp(-t * axial_f), axis=1))))
        parallel_lower = float(np.log(np.mean(1 / np.mean(np.exp(t * axial_f), axis=0))))
        trial_q = q + t * qdot
        trial_j = current + t * current_dot
        response_upper = float(np.log(np.sum(ct * (incidence @ trial_q)**2) / cap))
        response_lower = float(-np.log(cap * np.sum(trial_j**2 / ct)))
        tight_lower = max(lower, response_lower)
        tight_upper = min(upper, response_upper)
        gap_d = float(np.sum(ct * (incidence @ (q - qt))**2) / capt)
        gap_t = float(capt * np.sum((current - jt)**2 / ct))
        identity_error = max(abs(upper - exact - np.log1p(gap_d)),
                             abs(exact - lower - np.log1p(gap_t)))
        assert identity_error < 1e-7, identity_error
        assert lower - 1e-8 <= exact <= upper + 1e-8
        assert tight_lower - 1e-8 <= exact <= tight_upper + 1e-8
        assert parallel_lower - 1e-8 <= exact <= series_upper + 1e-8
        return dict(amplitude=t, exact_log_capacity=exact, lower=lower, upper=upper,
                    quadratic=-t * mean + t*t * curvature / 2,
                    response_lower=response_lower, response_upper=response_upper,
                    tight_lower=tight_lower, tight_upper=tight_upper,
                    frozen_width=upper-lower, response_width=response_upper-response_lower,
                    series_upper=series_upper, parallel_lower=parallel_lower,
                    series_parallel_width=series_upper-parallel_lower,
                    gap_identity_error=identity_error)

    return dict(ns=ns, ny=ny, mobility=mobility, winding=winding,
                mean=mean, variance=variance, chi=chi, curvature=curvature), evaluate


def write_csv(path, rows):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    OUT.mkdir(exist_ok=True)
    scan, amplitudes = [], []
    for r in np.logspace(-5, 1, 13):
        for winding in (0, 2):
            row, evaluate = problem(161, 64, float(r), winding)
            small = evaluate(0.02)
            row['curvature_fd'] = 2 * small['exact_log_capacity'] / 0.02**2
            row['fd_error'] = abs(row['curvature_fd'] - row['curvature'])
            scan.append(row)
            print(f"r={r:.4g} winding={winding} chi={row['chi']:.6f} curvature={row['curvature']:+.6f}", flush=True)
            if np.isclose(r, 0.001):
                for t in np.linspace(0, 2, 17):
                    amplitudes.append({**row, **evaluate(float(t))})
    write_csv(OUT / 'response_geometry_mobility.csv', scan)
    write_csv(OUT / 'response_geometry_amplitude.csv', amplitudes)
    refinements = []
    for ns, ny in ((81, 32), (161, 64), (321, 128)):
        for winding in (0, 2):
            row, evaluate = problem(ns, ny, 0.001, winding)
            refinements.append({**row, **evaluate(1.)})
    write_csv(OUT / 'response_geometry_refinement.csv', refinements)
    # Explicit perturbation maps, exported as numbers for the R figure workflow.
    s = np.linspace(0, 1, 201)
    eta = np.linspace(0, 1, 129)
    maps = [dict(s=float(si), eta=float(ei), winding=m,
                 error=float(field(np.array([si]), np.array([ei]), m)[0]))
            for m in (0, 2) for si in s for ei in eta]
    write_csv(OUT / 'response_geometry_fields.csv', maps)
    summary = dict(cases=len(scan), max_fd_error=max(v['fd_error'] for v in scan),
                   max_matched_variance_difference=max(abs(scan[i]['variance']-scan[i+1]['variance'])
                                                       for i in range(0,len(scan),2)),
                   refinements=refinements)
    (OUT / 'response_geometry_continuum_summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
