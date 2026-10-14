"""Smooth deep-well concentration defeats fixed L2 force control for d>=3.

Reference V=0 on [-2,2]^d. Equal basin balls of radius 0.5 contain disjoint
compact perturbations. Their unnormalized capacity is exactly unchanged, so
the capacity-rate ratio equals the inverse basin-partition ratio.
"""
from pathlib import Path
import numpy as np
from scipy.integrate import quad
from scipy.special import gamma,expit
from core import write_csv

OUT=Path(__file__).resolve().parent/'results'


def bump(t):
    if t<=.5:return 1.,0.
    if t>=1:return 0.,0.
    z=2*t-1
    value=expit(1/z-1/(1-z))
    derivative=-2*value*(1-value)*(1/z**2+1/(1-z)**2)
    return value,derivative


def main():
    rows=[]
    for d in [3,4]:
        surface=2*np.pi**(d/2)/gamma(d/2)
        basin=surface*.5**d/d;volume=4.**d
        gradient=surface*quad(lambda t:bump(t)[1]**2*t**(d-1),.5,1,epsabs=1e-10)[0]
        for radius in [1e-1,3e-2,1e-2,3e-3,1e-3,1e-4,1e-5,1e-6,1e-7]:
            depth=(d+1)*np.log(1/radius)
            integral=quad(lambda t:np.expm1(depth*bump(t)[0])*t**(d-1),0,1,
                          points=[.5],epsabs=1e-8,epsrel=1e-10)[0]
            target_basin=basin+surface*radius**d*integral
            eps=np.sqrt(2*depth**2*radius**(d-2)*gradient/volume)
            rows.append(dict(dimension=d,radius=radius,depth_kT=depth,
                force_uniform_rms=eps,capacity_ratio=1.,basin_partition_ratio=target_basin/basin,
                capacity_rate_ratio=basin/target_basin,
                log_capacity_rate_ratio=float(np.log(basin/target_basin))))
    write_csv(OUT/'concentration_counterexample.csv',rows)
    for d in [3,4]:
        r=[r for r in rows if r['dimension']==d][-1]
        print(r,flush=True)


if __name__=='__main__':main()
