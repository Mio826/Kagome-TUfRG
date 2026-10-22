"""Exact finite-M TU source eigenvectors -> spinful Hermitian 24x24 operators."""
from __future__ import annotations
import hashlib, io, json
from pathlib import Path
import numpy as np
import TUfRG_physics as ph
import gl_gateA_preflight as pre
CELLS=pre.CELLS
CELL_I=pre.CELL_I
Q=('M_a','M_b','M_c')
QRED={'M_a':(0.,.5),'M_b':(.5,0.),'M_c':(.5,.5)}
TAU2=np.array(((1,1),(0,1),(1,2)),dtype=np.int64)
C6=np.array(((0,-1),(1,1)),dtype=np.int64)


def load_geometry(path):
 d=json.loads(Path(path).read_text()); n=int(d['nff'])
 R=np.array(d['mi_to_R'],dtype=int); aa=np.array(d['mi_to_ofrom'],dtype=int);bb=np.array(d['mi_to_oto'],dtype=int)
 assert R.shape==(n,3) and aa.shape==(n,) and bb.shape==(n,)
 lookup={(tuple(R[i]),int(aa[i]),int(bb[i])):i for i in range(n)}
 assert len(lookup)==n
 rev=np.array([lookup[(tuple(-R[i]),int(bb[i]),int(aa[i]))] for i in range(n)],dtype=int)
 assert np.array_equal(rev[rev],np.arange(n))
 return ph.Geometry(R,aa,bb,rev,24,n,24**2)


def load_source_vector(z,snap,q,geometry):
 root=f'snapshots/{snap}/'; diag=json.loads(z.read(root+'diagnosis.json'))
 labels=diag['q'][q]['longitudinal']['sectors']['TR_odd']['source_labels']
 iq=geometry.high_symmetry_q()[q]
 basis=ph.longitudinal_basis(geometry,iq)
 idx=np.flatnonzero(basis.parity==-1)
 assert list(np.array(basis.labels)[idx])==labels, 'Archive and exact source basis labels differ!'
 modes=np.load(io.BytesIO(z.read(root+'leading_modes.npz')))
 gap=np.array(modes[q+'_longitudinal_TR_odd_gap'],dtype=float)
 assert gap.shape==(len(idx),)
 assert abs(np.linalg.norm(gap)-1)<1e-8
 f=(basis.transform[:,idx]@gap).reshape(4,geometry.nff)
 assert np.linalg.norm(f[1:3])<1e-13
 return f,{'source_dimension':len(idx),'gap_norm':float(np.linalg.norm(gap)),
           'source_labels_sha256':hashlib.sha256('\n'.join(labels).encode()).hexdigest(),
           'basis_coeff_norm':float(np.linalg.norm(f)),
           'onsite_spinful_indices':[int(i) for i in range(geometry.nff) if geometry.reverse[i]==i]}


def build_W(K,geometry,q,f):
 """K coordinates relative to 2x2 supercell; source c†_(n+R,a) c_(n,b)."""
 K=np.asarray(K,dtype=float); assert K.shape==(2,)
 qbits=np.array(QRED[q])
 W=np.zeros((24,24),complex)
 for n in CELLS:
  char=np.exp(2j*np.pi*np.dot(qbits,n))
  for i in range(geometry.nff):
   ra,rb=map(int,geometry.ff_R[i,:2]); a=int(geometry.from_orb[i]); b=int(geometry.to_orb[i]);
   row_unc=(n[0]+ra,n[1]+rb); rowcell=(row_unc[0]%2,row_unc[1]%2)
   row_wrap=((row_unc[0]-rowcell[0])//2,(row_unc[1]-rowcell[1])//2)
   boundary=np.exp(-2j*np.pi*np.dot(K,row_wrap))
   for s in (0,1):
    row=s*12+CELL_I[rowcell]*3+a
    col=s*12+CELL_I[n]*3+b
    W[row,col]+=char*boundary*f[(0 if s==0 else 3),i]
 return W


def site_C6_perm():
 """At K=0: exact 12-site orbital permutation for +60° C6 about hexagon origin."""
 p=np.empty(12,dtype=int)
 for i,n in enumerate(CELLS):
  for orb in range(3):
   doubled=C6@(2*np.array(n,dtype=np.int64)+TAU2[orb]); m=[]
   for o,t in enumerate(TAU2):
    if np.all((doubled-t)%2==0):m.append((o,(doubled-t)//2))
   assert len(m)==1
   o,v=m[0];cell=(int(v[0]%2),int(v[1]%2))
   p[i*3+orb]=CELL_I[cell]*3+o
 assert len(set(p.tolist()))==12
 P=np.zeros((24,24),complex)
 for s in (0,1):
  for i,j in enumerate(p):P[12*s+j,12*s+i]=1
 return P


def folded_point_group_C6(K):
    """Unitary supercell Bloch sewing matrix, with 2x2 boundary phases.

    A acts on primitive cell integers; supercell K'=A^{-T} K.
    Gauge uses H_ij(K)=sum_R h_ij(R)e^{+i K.R}.
    """
    A=C6
    K=np.asarray(K,float)
    kp=np.linalg.inv(A).T@K
    P=np.zeros((24,24),complex)
    for cell_idx,n in enumerate(CELLS):
        for a in range(3):
            transformed=A@(2*np.asarray(n)+TAU2[a])
            for b,tau in enumerate(TAU2):
                if np.all((transformed-tau)%2==0):
                    unwrapped=(transformed-tau)//2
                    modcell=(int(unwrapped[0]%2),int(unwrapped[1]%2))
                    wrap=(unwrapped-np.array(modcell,dtype=int))//2
                    ci=CELL_I[modcell]
                    for s in (0,1):
                        P[12*s+3*ci+b,12*s+3*cell_idx+a]=np.exp(-2j*np.pi*np.dot(kp,wrap))
                    break
            else:raise AssertionError('Non-bijective site map')
    assert np.linalg.norm(P@P.conj().T-np.eye(24))<1e-12
    return kp,P
