"""
How many transition-path frames does the bracket need?   (statistics of the exponential average)

If beta*dV on the leverage ensemble has spread sigma, the estimator  ln <e^{+-beta dV}>_N  of N independent frames has
    relative variance of the mean  = (e^{sigma^2} - 1) / N          (exact for Gaussian dV)
    bias of the log                = -(e^{sigma^2} - 1) / (2N)      (the log of a noisy mean is biased low)
so the number of frames for a target accuracy eps in ln k is   N ~ (e^{sigma^2} - 1) / eps^2 :  EXPONENTIAL in the error spread.
A second-cumulant (Gaussian) estimate  ln<e^{x}> ~ mean + var/2  needs only N ~ (sigma^2 + sigma^4/2)/eps^2, but is biased if dV is not Gaussian.
"""
import numpy as np
rng = np.random.default_rng(0)
print("sigma (kT) | N | sd of ln<e^x>: simulated | formula sqrt((e^s2-1)/N) | bias: simulated | formula | sd of the cumulant estimate | frames for +-0.1")
rows = []
for s in (0.5, 1.0, 1.5, 2.0, 2.5):
    for N in (200, 2000):
        x = rng.normal(0, s, size=(4000, N)); est = np.log(np.exp(x).mean(1)); cum = x.mean(1) + x.var(1) / 2
        f_sd = np.sqrt((np.exp(s**2) - 1) / N); f_b = -(np.exp(s**2) - 1) / (2 * N)
        rows.append((s, N, est.std(), f_sd, est.mean() - s**2 / 2, f_b, cum.std(), (np.exp(s**2) - 1) / 0.01))
        print(f"   {s:.1f}     | {N:4d} |      {est.std():.3f}           |        {f_sd:.3f}           |    {est.mean() - s**2/2:+.3f}     | {f_b:+.3f}  |        {cum.std():.3f}              | {(np.exp(s**2)-1)/0.01:9.0f}")
np.savetxt("data/exp_average_error.csv", rows, delimiter=",", header="sigma,N,sd_sim,sd_formula,bias_sim,bias_formula,sd_cumulant,N_for_0.1", comments="")
