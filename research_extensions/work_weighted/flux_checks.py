"""Response-flux checks of Supporting Information Section S22: condition numbers of weighted force errors
on the double well (doublewell.csv) and the opened-route example (finite_amplitude_counterexample.csv).
    python flux_checks.py
"""
from pathlib import Path
import sys,json
import numpy as np
from scipy.special import logsumexp
from scipy.integrate import simpson
from core import Network,square_ring,write_csv

OUT=Path(__file__).resolve().parent/'results'


def doublewell():
    rows=[]; checks=[]
    for n in [301,601,1201]:
        x=np.linspace(-1.5,1.5,n); dx=x[1]-x[0]
        net=Network(x[:,None],np.arange(n-1),np.arange(1,n),np.full(n-1,dx))
        for barrier in [3.,5.,7.,9.,11.,13.,15.]:
            v=barrier*(x*x-1)**2
            s=net.spectral(v); rho=s['rho']; j=np.cumsum(rho)[:-1]
            wstar=dx*np.abs(j); length=wstar.sum(); wstar/=length
            pe=net.edge_weights(s['mass']); unif=np.full(n-1,1/(n-1))
            wm=.9*wstar+.1*unif
            coeff=lambda w: float(np.sqrt(np.sum(j*j*dx*dx/w)))
            rows.append(dict(n=n,barrier=barrier,rate=s['rate'],
                kappa_equilibrium=coeff(pe),kappa_uniform=coeff(unif),
                kappa_work=length,kappa_mixed=coeff(wm),
                finite_coefficient=np.sqrt(np.sum(dx*dx/wm))))
            if n==601:
                h=np.exp(-x*x/(2*.18**2))+.15*np.sin(3*x)
                dt=1e-3
                fd=(np.log(net.spectral(v+dt*h)['rate'])-
                    np.log(net.spectral(v-dt*h)['rate']))/(2*dt)
                pred=-rho@h
                checks.append(dict(barrier=barrier,finite_difference=fd,
                    analytic=pred,absolute_error=abs(fd-pred),
                    work_identity_error=abs(h@rho-(net.G@h)@j)))
    write_csv(OUT/'doublewell.csv',rows);write_csv(OUT/'derivative_checks.csv',checks)
    print('doublewell',rows[-7:],flush=True)



def counterexample():
    # Two parallel two-edge paths with fixed terminal energies. The initially
    # suppressed route is opened until both paths have equal conductance.
    xy=np.array([[-1.,0.],[1.,0.],[0.,1.],[0.,-1.]])
    net=Network(xy,np.array([0,2,0,3]),np.array([2,1,3,1]),np.ones(4))
    a=np.array([1,0,0,0],bool);b=np.array([0,1,0,0],bool)
    rows=[]
    for delta in [2.,4.,8.,12.,16.,20.]:
        v=np.array([0.,0.,2*delta,0.]);h=np.array([0.,0.,-2*delta,0.])
        p=1/(1+np.exp(delta));j=np.array([-p/4,p/4,-(1-p)/4,(1-p)/4])
        ref=net.capacity(v,a,b)
        assert np.max(abs(net.G.T@j-ref['rho']))<1e-10
        w=np.array([p/2,p/2,(1-p)/2,(1-p)/2])
        eps=net.epsilon(h,w);local=.5*eps
        exact=float(np.log(net.capacity(v+h,a,b)['rate']/ref['rate']))
        assert abs(exact-np.log(2*(1-p)))<1e-9
        rows.append(dict(delta=delta,reference_upper_fraction=p,epsilon_work=eps,
            kappa_optimal=.5,local_bound=local,exact_log_rate=exact,
            violation_factor=exact/local))
    write_csv(OUT/'finite_amplitude_counterexample.csv',rows)
    print('counterexample',rows[-1],flush=True)


if __name__=='__main__':
    OUT.mkdir(parents=True,exist_ok=True)
    doublewell(); counterexample()
