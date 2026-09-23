"""Apply the new comparison to the existing alanine free-energy network.

This checks the reference-response algorithm on a molecular-data-derived
overdamped surrogate, not a certificate for underdamped atomistic MD.
No new force-field labels or trajectories are generated.
"""
from pathlib import Path
import os
import sys
import csv
import json
import numpy as np
from scipy.sparse import coo_matrix, diags, csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import spsolve

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'ala2'))
import analyze as a


def main():
    os.chdir(ROOT/'ala2')
    trajectories, temp=a.load('ref_g1')
    lag, keep=a.msm(trajectories)
    network=a.sqra(lag,keep)
    n=len(keep)
    phi=np.degrees(a.centres()[0][keep])
    in_a=(phi>a.CORE_A[0]) & (phi<a.CORE_A[1])
    in_b=(phi>a.CORE_B[0]) & (phi<a.CORE_B[1])
    _, components=connected_components(csr_matrix(network))
    anchored=np.isin(components,np.unique(components[in_a|in_b]))
    free=np.flatnonzero(anchored & ~(in_a|in_b))
    ii,jj=np.nonzero(np.triu(network,1))
    c=network[ii,jj]
    edge=np.arange(len(c))
    incidence=coo_matrix((np.r_[-np.ones(len(c)),np.ones(len(c))],
                         (np.r_[edge,edge],np.r_[ii,jj])),shape=(len(c),n)).tocsr()

    def solve(conductance):
        lap=incidence.T@diags(conductance)@incidence
        q=np.zeros(n);q[in_b]=1
        q[free]=spsolve(lap[free][:,free],-lap[free]@q)
        capacity=float(np.sum(conductance*(incidence@q)**2))
        return capacity,q,lap

    cap,q,lap=solve(c)
    dq=incidence@q
    nu=c*dq*dq/cap
    j=c*dq/cap
    rows=[]
    designs=json.loads(Path('design.json').read_text())
    for name,bumps in designs.items():
        node_error=a.gauss_field(bumps,a.KJ_PER_KT(temp))[keep]
        f=(node_error[ii]+node_error[jj])/2
        mean=float(nu@f)
        qdot=np.zeros(n)
        rhs=incidence.T@(c*f*dq)
        qdot[free]=spsolve(lap[free][:,free],rhs[free])
        jdot=c*((incidence@qdot)-(f-mean)*dq)/cap
        variance=float(nu@(f-mean)**2)
        curvature=variance-2*float(np.sum(c*(incidence@qdot)**2))/cap
        for amplitude in (.1,.25,.5,1.):
            ct=c*np.exp(-amplitude*f)
            target,_,_=solve(ct)
            exact=float(np.log(target/cap))
            lo=float(-np.log(nu@np.exp(amplitude*f)))
            hi=float(np.log(nu@np.exp(-amplitude*f)))
            adapt_lo=float(-np.log(cap*np.sum((j+amplitude*jdot)**2/ct)))
            adapt_hi=float(np.log(np.sum(ct*(incidence@(q+amplitude*qdot))**2)/cap))
            assert max(lo,adapt_lo)-1e-9<=exact<=min(hi,adapt_hi)+1e-9
            rows.append(dict(design=name,amplitude=amplitude,nodes=n,edges=len(c),
                reference_capacity=cap,exact_log_capacity=exact,
                frozen_lower=lo,frozen_upper=hi,response_lower=adapt_lo,response_upper=adapt_hi,
                frozen_width=hi-lo,response_width=adapt_hi-adapt_lo,
                quadratic=-amplitude*mean+amplitude**2*curvature/2))
    out=ROOT/'research_extensions'/'results'/'response_geometry_alanine.csv'
    with out.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    print(f'{len(rows)} cases; all adapted/frozen intersections contain the network capacity.')
    for row in rows:
        if row['amplitude']==1:
            print(row['design'], 'frozen',row['frozen_width'],'adapted',row['response_width'],
                  'log_capacity',row['exact_log_capacity'])


if __name__=='__main__':
    main()
