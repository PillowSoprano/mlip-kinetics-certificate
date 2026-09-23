"""Sign-robust variants of the certificate-driven acquisition, many seeds, run in parallel.
  cert        mean over ensemble members                         (baseline, fails when the ensemble's sign is biased)
  cert_sym    mean over members AND their negatives              (sign-symmetrised belief)
  cert_max    worst member instead of the mean                   (minimax)
  cert_symmax worst case over members and their negatives
"""
import numpy as np
from multiprocessing import Pool
import al_certificate as a
import twochannel_blindspot as t

STRATS = ["unc", "cert", "cert_sym", "cert_max", "cert_symmax"]
NSEED = 16


def job(args):
    block, seed = args
    V = t.true_potential(); lam = abs(t.slow_mode(V)[0])
    t.BLOCK = block; t.rng = np.random.default_rng(200 + seed); dV0 = t.initial_error()
    members = a.make_ensemble(np.random.default_rng(300 + seed), block)
    sat = members[0][60, np.argmin(abs(t.xs + 1))]
    frac_neg = np.mean([m[60, np.argmin(abs(t.xs + 1))] < 0 for m in members])
    return block, seed, frac_neg, {s: a.run(V, dV0, members, lam, s)[0] for s in STRATS}


if __name__ == "__main__":
    jobs = [(b, s) for b in (6.0, 12.0) for s in range(NSEED)]
    with Pool(8) as p:
        out = p.map(job, jobs)
    for block in (6.0, 12.0):
        rs = [o for o in out if o[0] == block]
        print(f"\nblock = {block:.0f} kT, {NSEED} seeds, |ln(k_hat/k)|")
        print("   strategy      median q2  q4    q8    q12  | mean q12 | worst q12 | seeds with q12 error > 0.5")
        for s in STRATS:
            E = np.array([o[3][s] for o in rs]); m = np.median(E, axis=0)
            print(f"   {s:12s}  {m[2]:6.2f} {m[4]:5.2f} {m[8]:5.2f} {m[12]:5.2f} | {E[:,12].mean():7.2f}  | {E[:,12].max():8.2f}  | {int((E[:,12] > 0.5).sum())} / {NSEED}")
            np.savetxt(f"data/al_signrobust_block{block:.0f}_{s}.csv", E, delimiter=",")
        fn = np.array([o[2] for o in rs]); Ec = np.array([o[3]["cert"][12] for o in rs])
        print(f"   plain cert failures vs ensemble sign bias at the blocked saddle: failing seeds have frac(negative members) = {fn[Ec > 0.5].round(2)}; others median {np.median(fn[Ec <= 0.5]):.2f}")
