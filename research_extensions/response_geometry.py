"""Check capacity response geometry on finite positive conductance networks.

Run with a Python containing numpy.
"""
from pathlib import Path
import json
import numpy as np


def solve(n, edges, conductance):
    incidence = np.zeros((len(edges), n))
    for e, (i, j) in enumerate(edges):
        incidence[e, i], incidence[e, j] = -1.0, 1.0
    lap = incidence.T @ (conductance[:, None] * incidence)
    q = np.zeros(n)
    q[-1] = 1.0
    interior = np.arange(1, n - 1)
    if len(interior):
        q[interior] = np.linalg.solve(lap[np.ix_(interior, interior)],
                                     -lap[interior, -1])
    dq = incidence @ q
    capacity = np.sum(conductance * dq**2)
    current = conductance * dq / capacity
    return capacity, q, current, incidence, lap, interior


def check(name, n, edges, c, error):
    cap, q, current, inc, lap, interior = solve(n, edges, c)
    dq = inc @ q
    nu = c * dq**2 / cap
    mean = nu @ error
    variance = nu @ (error - mean)**2
    qdot = np.zeros(n)
    if len(interior):
        rhs = inc.T @ (c * error * dq)
        qdot[interior] = np.linalg.solve(lap[np.ix_(interior, interior)], rhs[interior])
    relaxation = np.sum(c * (inc @ qdot)**2) / cap
    curvature = variance - 2 * relaxation  # beta = 1

    def logcap(theta):
        return np.log(solve(n, edges, c * np.exp(-theta * error))[0])

    h = 0.002
    def fd(step):
        return (logcap(step) - 2 * logcap(0) + logcap(-step)) / step**2
    fd_curvature = (4 * fd(h / 2) - fd(h)) / 3
    hatted_c = c * np.exp(-0.7 * error)
    hcap, hq, hcurrent, *_ = solve(n, edges, hatted_c)
    trial_d = np.sum(hatted_c * dq**2)
    trial_t = np.sum(current**2 / hatted_c)
    d_gap = np.sum(hatted_c * (inc @ (q - hq))**2)
    t_gap = np.sum((current - hcurrent)**2 / hatted_c)
    gap_error = max(abs(trial_d / hcap - 1 - d_gap / hcap),
                    abs(hcap * trial_t - 1 - hcap * t_gap))
    current_dot = c * ((inc @ qdot) - (error - mean) * dq) / cap
    response_widths = {}
    for amplitude in (0.025, 0.05, 0.1, 1., -1.2):
        ct = c * np.exp(-amplitude * error)
        target = solve(n, edges, ct)[0]
        upper = np.sum(ct * (inc @ (q + amplitude * qdot))**2) / cap
        lower = 1 / (cap * np.sum((current + amplitude * current_dot)**2 / ct))
        assert lower <= target / cap + 1e-11
        assert target / cap <= upper + 1e-11
        response_widths[str(amplitude)] = float(np.log(upper/lower))
    assert abs(curvature - fd_curvature) < 5e-6
    assert gap_error < 1e-10
    assert -1e-10 <= relaxation <= variance + 1e-10
    return dict(name=name, variance=float(variance),
                chi=float(relaxation / variance), curvature=float(curvature),
                fd_curvature=float(fd_curvature),
                curvature_error=float(abs(curvature - fd_curvature)),
                pythagorean_error=float(gap_error),
                response_widths=response_widths)


def main():
    results = [
        check('series', 3, [(0, 1), (1, 2)], np.ones(2), np.array([1., -1.])),
        check('parallel', 2, [(0, 1), (0, 1)], np.ones(2), np.array([1., -1.])),
    ]
    rng = np.random.default_rng(20260922)
    for seed in range(32):
        n = 10
        edges = [(i, i + 1) for i in range(n - 1)]
        edges += [(i, j) for i in range(n) for j in range(i + 2, n)
                  if rng.random() < 0.3]
        c = np.exp(rng.normal(0, 1, len(edges)))
        error = rng.normal(0, 1, len(edges))
        results.append(check(f'network_{seed}', n, edges, c, error))
    out = Path(__file__).resolve().parent / 'results' / 'response_geometry.json'
    out.write_text(json.dumps(results, indent=2) + '\n')
    print(json.dumps(dict(cases=len(results),
          max_curvature_error=max(r['curvature_error'] for r in results),
          max_pythagorean_error=max(r['pythagorean_error'] for r in results),
          series=results[0], parallel=results[1]), indent=2))


if __name__ == '__main__':
    main()
