"""Work-weighted response on reversible finite-volume networks.

Energy is in kT (beta=1). Edges have physical length ell. The incidence
matrix has +1 at the tail and -1 at the head: G f / ell is the force along
the oriented edge. Source measures and forces use the same discretization.
"""
from dataclasses import dataclass
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve, eigsh
from scipy.optimize import linprog


@dataclass
class Network:
    xy: np.ndarray
    tail: np.ndarray
    head: np.ndarray
    ell: np.ndarray

    def __post_init__(self):
        n, m = len(self.xy), len(self.tail)
        e = np.arange(m)
        self.G = sparse.coo_matrix((np.r_[np.ones(m), -np.ones(m)],
            (np.r_[e,e], np.r_[self.tail,self.head])), shape=(m,n)).tocsr()

    def landscape(self, v):
        mass = np.exp(-(v-np.min(v)))
        mass /= mass.sum()
        c = np.sqrt(mass[self.tail]*mass[self.head])/self.ell**2
        return mass, c

    def capacity(self, v, a, b):
        mass,c = self.landscape(v)
        lap = self.G.T @ sparse.diags(c) @ self.G
        q = np.zeros(len(v)); q[b] = 1
        free = np.flatnonzero(~(a|b))
        q[free] = spsolve(lap[free][:,free], -(lap@q)[free])
        cap = float(np.sum(c*(self.G@q)**2))
        nu_e = c*(self.G@q)**2/cap
        nu = self.nodes(nu_e)
        eta_a = mass*a/np.sum(mass[a])
        eta_b = mass*b/np.sum(mass[b])
        pa = mass[a].sum()/(mass[a].sum()+mass[b].sum())
        eta = (1-pa)*eta_a + pa*eta_b
        rho = nu-eta; rho[-1] -= rho.sum()
        rate = cap*(1/mass[a].sum()+1/mass[b].sum())
        return dict(mass=mass,c=c,q=q,cap=cap,rate=rate,nu=nu,nu_e=nu_e,
                    eta=eta,rho=rho,eta_a=eta_a,eta_b=eta_b,pa=pa)

    def spectral(self, v):
        mass,c = self.landscape(v)
        lap = self.G.T @ sparse.diags(c) @ self.G
        inv = sparse.diags(1/np.sqrt(mass))
        sym = (inv@lap@inv).tocsc()
        vals,vecs = eigsh(sym,k=3,sigma=-1e-7,which='LM',tol=1e-11,
                         v0=np.random.default_rng(204).normal(size=len(v)))
        ix=np.argsort(vals); k=float(vals[ix[1]])
        phi=vecs[:,ix[1]]/np.sqrt(mass)
        nu_e=c*(self.G@phi)**2
        nu_e/=nu_e.sum()
        eta=mass*phi**2; eta/=eta.sum()
        rho=self.nodes(nu_e)-eta; rho[-1]-=rho.sum()
        return dict(rate=k,mass=mass,rho=rho,nu=self.nodes(nu_e),eta=eta,phi=phi)

    def nodes(self, edge_weight):
        return (np.bincount(self.tail,edge_weight/2,minlength=len(self.xy))+
                np.bincount(self.head,edge_weight/2,minlength=len(self.xy)))

    def edge_weights(self, node_weight):
        p=(node_weight[self.tail]+node_weight[self.head])/2
        return p/p.sum()

    def transport(self,rho):
        """Graph Beckmann LP; the metric is the edge-length shortest path."""
        m=len(self.ell)
        ae=sparse.hstack([self.G.T,-self.G.T],format='csr')[:-1]
        sol=linprog(np.r_[self.ell,self.ell],A_eq=ae,b_eq=rho[:-1],
                    bounds=(0,None),method='highs')
        if not sol.success: raise RuntimeError(sol.message)
        j=sol.x[:m]-sol.x[m:]
        residual=float(np.max(np.abs(self.G.T@j-rho)))
        if residual>2e-7: raise AssertionError(residual)
        length=float(self.ell@np.abs(j))
        return dict(j=j,length=length,w=self.ell*np.abs(j)/length,
                    residual=residual,dual=float(sol.fun))

    def condition(self,rho,w):
        """Exact local condition number for gradient errors in the edge norm."""
        if np.min(w)<=0: raise ValueError('Use positive weights for the solve.')
        lap=self.G.T@sparse.diags(w/self.ell**2)@self.G
        u=np.zeros(len(rho)); u[:-1]=spsolve(lap[:-1,:-1],rho[:-1])
        return float(np.sqrt(max(0,rho@u)))

    def resistance_diameter(self,w):
        """Finite-amplitude coefficient: max pairwise resistance distance."""
        lap=(self.G.T@sparse.diags(w/self.ell**2)@self.G).toarray()
        inv=np.linalg.inv(lap[:-1,:-1])
        z=np.zeros_like(lap); z[:-1,:-1]=inv
        d=np.diag(z)
        return float(np.sqrt(np.max(d[:,None]+d[None,:]-2*z)))

    def epsilon(self,h,w):
        return float(np.sqrt(np.sum(w*(self.G@h/self.ell)**2)))


def square_ring(n=41,barrier=5.):
    x=np.linspace(-1.8,1.8,n); X,Y=np.meshgrid(x,x,indexing='ij')
    ids=np.arange(n*n).reshape(n,n)
    i=np.r_[ids[:-1].ravel(),ids[:,:-1].ravel()]
    j=np.r_[ids[1:].ravel(),ids[:,1:].ravel()]
    xy=np.c_[X.ravel(),Y.ravel()]; r=np.hypot(X,Y)
    net=Network(xy,i,j,np.full(len(i),x[1]-x[0]))
    s=Y/np.sqrt(r*r+.01)
    wall=30*(r-1)**2
    v=wall+s*s*(barrier+1+s)
    a=(X.ravel()<-.65)&(np.abs(Y.ravel())<.5)
    b=(X.ravel()>.65)&(np.abs(Y.ravel())<.5)
    return net,v.ravel(),a,b,wall.ravel()


def write_csv(path,rows):
    import csv
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',newline='') as f:
        wr=csv.DictWriter(f,fieldnames=list(rows[0]));wr.writeheader();wr.writerows(rows)
