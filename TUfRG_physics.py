#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Physical TUfRG kernels for spin-conserving intrinsic Kane--Mele kagome models.

Stage-3 core numerical module (NOT a phase classifier and NOT a FRG runner).

Inputs:
  - archived O.dvg with raw P/C/D and integrated PP/PH loops,
  - native **FULL-PROJECTED** p,d,m matrices exported by the audited divERGe C
    projector (e.g. TUfRG_restore.c), never raw P/C/D substituted for them.

Conventions verified in Stage 2:
  Longitudinal/transverse PH: H = (T^H full_d T) @ (T^H L_PH T).
  Cooper at Gamma:          H = -(0.5*T^H full_p T) @ (T^H L_PP T).
  In both cases, L is the signed flow-integrated divERGe loop as serialized,
  not a necessarily positive/negative static bare susceptibility.

The full (potentially complex) spectrum of H is retained.  Physical Hermitian
PH-source eigenmodes must have real coefficients in the real-source basis.
Do not clip loops, use abs(eigenvalue), or use Cholesky/singular values to rank
instabilities.  A report of a leading real branch is NOT a thermodynamic proof
of ordering, nor an automatic classification of its order parameter.

This module is self-contained apart from NumPy, so tests do not depend on the
old, previously incorrect TUfRG_instability.py or on a Stage-2 test directory.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import numpy as np

_COMPLEX = np.dtype('<c16')
_INTEGER = np.dtype('<i8')


class PhysicsInputError(ValueError):
    """An archived array, source basis or physical kernel fails validation."""


def _ratio(a: np.ndarray, denominator: np.ndarray | None = None) -> float:
    return float(np.linalg.norm(a) / max(np.linalg.norm(a if denominator is None else denominator), 1e-30))


def _hermiticity(a: np.ndarray) -> float:
    return _ratio(a - a.conj().T, a)


def _cross(a: np.ndarray, rows: np.ndarray, cols: np.ndarray) -> float:
    return _ratio(a[np.ix_(rows, cols)], a)


@dataclass(frozen=True)
class Geometry:
    ff_R: np.ndarray          # (nff,3), integer lattice translations
    from_orb: np.ndarray      # nff
    to_orb: np.ndarray        # nff
    reverse: np.ndarray       # nff, involution
    nk: int
    nff: int
    nq: int

    @classmethod
    def from_O(cls, path: str | Path) -> 'Geometry':
        path = Path(path)
        h = np.fromfile(path, dtype=_INTEGER, count=256)
        if h.size != 256 or int(h[0]) != 84:
            raise PhysicsInputError('O.dvg v0.10 header missing or unrecognized')
        norb, nspin, nq, nff = (int(h[k]) for k in (1, 2, 3, 7))
        nk = int(round(nq ** 0.5))
        if norb != 3 or nspin != 2 or int(h[6]) != 0 or nk * nk != nq or nk % 2:
            raise PhysicsInputError('Expected 3-orbital explicit-spin (SU2=0), even square q grid')
        if int(h[5]) != nq or int(h[9]) != 0:
            raise PhysicsInputError('Symmetry-compressed O.dvg is unsupported')
        # Include the five complex128 blocks and FF metadata in the extent audit.
        filesize = path.stat().st_size
        for index in (84, 86, 88, 90, 92):
            off, num = int(h[index]), int(h[index + 1])
            if num != 16 * nff * nff * nq or off < 0 or off + 16 * num > filesize:
                raise PhysicsInputError(f'Incomplete/invalid O.dvg block {index}')
        for off_i, len_i, num in ((10,11,nff),(12,13,nff),(14,15,3*nff)):
            off = int(h[off_i])
            if int(h[len_i]) != num or off < 0 or off + 8 * num > filesize:
                raise PhysicsInputError(f'Invalid form-factor metadata {off_i}')
        a = np.fromfile(path, dtype=_INTEGER, count=nff, offset=int(h[10]))
        b = np.fromfile(path, dtype=_INTEGER, count=nff, offset=int(h[12]))
        R = np.fromfile(path, dtype=_INTEGER, count=3*nff, offset=int(h[14])).reshape(nff,3)
        if not ((a >= 0).all() and (a < norb).all() and (b >= 0).all() and (b < norb).all()):
            raise PhysicsInputError('Out-of-range FF orbitals')
        if (R[:,2] != 0).any():
            raise PhysicsInputError('Non-2D form factors')
        keys = [(tuple(map(int,R[i])),int(a[i]),int(b[i])) for i in range(nff)]
        lookup = {key:i for i,key in enumerate(keys)}
        if len(lookup) != nff:
            raise PhysicsInputError('Duplicated form factors')
        partners=[]
        for vec,src,dst in keys:
            rev=(tuple(-v for v in vec),dst,src)
            if rev not in lookup:
                raise PhysicsInputError(f'FF missing reverse partner: {(vec,src,dst)}')
            partners.append(lookup[rev])
        partners=np.asarray(partners,dtype=int)
        if not np.array_equal(partners[partners],np.arange(nff)):
            raise PhysicsInputError('FF reversal not involutive')
        return cls(R,a,b,partners,nk,nff,nq)

    def high_symmetry_q(self) -> dict[str,int]:
        n=self.nk;half=n//2
        return {'Gamma':0,'M_a':half,'M_b':half*n,'M_c':half*n+half}

    def is_ph_trim(self, iq:int) -> bool:
        """Whether q = -q modulo reciprocal lattice vectors on this mesh."""
        if not 0 <= iq < self.nq:
            return False
        ix, iy = divmod(int(iq), self.nk)
        return (2*ix) % self.nk == 0 and (2*iy) % self.nk == 0

    def reduced_q(self, iq:int) -> np.ndarray:
        if not 0<=iq<self.nq:raise PhysicsInputError(f'q outside mesh: {iq}')
        return np.asarray([iq//self.nk, iq%self.nk],dtype=float)/self.nk


@dataclass(frozen=True)
class SourceBasis:
    transform: np.ndarray
    parity: np.ndarray | None
    labels: tuple[str,...]
    sector: np.ndarray | None


def _orthonormal(T: np.ndarray, name:str) -> None:
    gram=T.conj().T@T
    if np.linalg.norm(gram - np.eye(T.shape[1])) > 1e-9:
        raise PhysicsInputError(f'{name} is not orthonormal')


def longitudinal_basis(geometry:Geometry, iq:int) -> SourceBasis:
    """Hermitian physical sources rho,Sz at Gamma/M: dim=2*nff.

    Ordered source labels: `0:on:i`, `0:R:i,j`, `0:I:i,j`, then Sz=1.
    Finite-M conjugation includes the inverse-bond Bloch phase.
    """
    if not geometry.is_ph_trim(iq):
        raise PhysicsInputError(
            'Hermitian PH sources require q=-q (mod G), i.e. Gamma/three M. '
            'Generic-q diagnosis requires a distinct q/-q paired complex-source '
            'formulation and is NOT certified by this module.')
    R,rev=geometry.ff_R,geometry.reverse
    q=geometry.reduced_q(iq)
    n=geometry.nff
    spin=np.array([[1,0,0,1],[1,0,0,-1]],complex)/np.sqrt(2)
    cols=[];tr=[];labels=[]
    for ic in range(2):
        for i,j in enumerate(rev):
            if j<i:continue
            phase=np.exp(-2j*np.pi*np.dot(q,R[i,:2]))
            if i==j:
                if abs(phase-1)>1e-9:
                    raise PhysicsInputError('PH self-reverse FF has nontrivial anchor phase')
                f=np.zeros(n,complex);f[i]=1
                cols.append(np.kron(spin[ic],f)); labels.append(f'{ic}:on:{i}')
                tr.append(1 if ic==0 else -1)
            else:
                for imaginary in (False,True):
                    f=np.zeros(n,complex)
                    f[i]=(1j if imaginary else 1)/np.sqrt(2)
                    f[j]=phase*f[i].conjugate()
                    cols.append(np.kron(spin[ic],f))
                    labels.append(f'{ic}:{"I" if imaginary else "R"}:{i},{j}')
                    tr.append((1 if ic==0 else -1)*(-1 if imaginary else 1))
    T=np.column_stack(cols);_orthonormal(T,'longitudinal PH sources')
    if T.shape[1] != 2*n:raise PhysicsInputError('Incorrect longitudinal physical dimension')
    return SourceBasis(T,np.asarray(tr,int),tuple(labels),None)


def transverse_basis(longitudinal:SourceBasis, nff:int) -> SourceBasis:
    """Hermitian Sx/Sy sources from the *same* reverse-bond basis.

    Intrinsic KM preserves U(1)_Sz; xy covariance and TR-sector splitting are
    checked at the kernel level rather than hardcoded into the basis.
    """
    if longitudinal.parity is None or len(longitudinal.labels)!=2*nff:
        raise PhysicsInputError('Require a longitudinal Gamma/M basis')
    oldsp=np.array([[1,0,0,1],[1,0,0,-1]],complex)/np.sqrt(2)
    newsp=np.array([[0,1,1,0],[0,-1j,1j,0]],complex)/np.sqrt(2)
    T=np.empty_like(longitudinal.transform)
    parity=np.empty(len(longitudinal.labels),int)
    newlabels=[]
    for i,label in enumerate(longitudinal.labels):
        ic=int(label[0])
        f=oldsp[ic].conj()@longitudinal.transform[:,i].reshape(4,nff)
        T[:,i]=np.kron(newsp[ic],f)
        parity[i]=int(longitudinal.parity[i])*(-1 if ic==0 else 1)
        newlabels.append(('Sx' if ic==0 else 'Sy')+label[1:])
    _orthonormal(T,'transverse PH sources')
    return SourceBasis(T,parity,tuple(newlabels),None)


def cooper_basis(geometry:Geometry) -> SourceBasis:
    """Pauli antisymmetric Gamma pair sources, dim=279 for nff=141.

    One spin-singlet basis for each symmetric bond and spin-triplet bases for
    each antisymmetric bond; pairing Sz0 includes singlet AND triplet Sz=0.
    """
    n=geometry.nff;rev=geometry.reverse
    sing=np.array([0,1,-1,0],complex)/np.sqrt(2)
    trip={'0':np.array([0,1,1,0],complex)/np.sqrt(2),
          '+1':np.array([1,0,0,0],complex),
          '-1':np.array([0,0,0,1],complex)}
    cols=[];sectors=[];labels=[]
    for i,j in enumerate(rev):
        if j<i:continue
        f=np.zeros(n,complex)
        if i==j:
            f[i]=1
            cols.append(np.kron(sing,f));sectors.append('Sz0');labels.append(f'singlet:on:{i}')
        else:
            f[i]=f[j]=1/np.sqrt(2)
            cols.append(np.kron(sing,f));sectors.append('Sz0');labels.append(f'singlet:bond:{i},{j}')
            g=np.zeros(n,complex);g[i]=1/np.sqrt(2);g[j]=-1/np.sqrt(2)
            for s in ('0','+1','-1'):
                cols.append(np.kron(trip[s],g))
                sectors.append('Sz0' if s=='0' else s)
                labels.append(f'triplet{s}:bond:{i},{j}')
    T=np.column_stack(cols);_orthonormal(T,'Pauli Cooper sources')
    return SourceBasis(T,None,tuple(labels),np.asarray(sectors))


def native_full_matrix(projected_prefix:str | Path, kind:str, iq:int, nff:int) -> np.ndarray:
    """Load a *source-projected* p/d/m matrix exported by audited native C.

    This is NOT a loader for raw P/C/D from O.dvg. The suffix is lowercase.
    """
    if kind not in ('p','d','m'):
        raise PhysicsInputError('Expected audited full p/d/m projector; never raw P/C/D')
    path=Path(f'{projected_prefix}_{kind}_q{iq}.bin')
    expected=16*(4*nff)**2
    if not path.is_file() or path.stat().st_size!=expected:
        raise PhysicsInputError(f'Missing/incorrect native FULL projected vertex: {path}')
    mat=np.fromfile(path,dtype=_COMPLEX)
    return mat.reshape(4*nff,4*nff)


def native_loop(O_path:str | Path, kind:str, iq:int, geom:Geometry) -> np.ndarray:
    """Load only one q-slice of saved integrated PP or PH loop.

    C-saved PH loop has first/fourth spin indices swapped relative to native
    PH source order.  PP loop matches native P-spin order.  Both are signed.
    """
    if kind not in ('PH','PP'):
        raise PhysicsInputError('Expected PH or PP loop')
    O_path=Path(O_path)
    h=np.fromfile(O_path,dtype=_INTEGER,count=256)
    if h.size!=256 or int(h[7])!=geom.nff or int(h[3])!=geom.nq:
        raise PhysicsInputError('O.dvg header/geometry mismatch')
    off_index = 92 if kind=='PH' else 90
    off=int(h[off_index]);count=int(h[off_index+1]);n=geom.nff
    if count != 16*n*n*geom.nq or off + count*16 > O_path.stat().st_size:
        raise PhysicsInputError('Corrupted integrated loop block')
    raw=np.memmap(O_path,dtype=_COMPLEX,mode='r',offset=off,
                  shape=(2,2,n,2,2,n,geom.nq))
    sample=np.asarray(raw[...,iq])
    if kind=='PH':sample=sample.transpose(3,1,2,0,4,5)
    out=np.ascontiguousarray(sample).reshape(4*n,4*n)
    del raw
    return out


@dataclass(frozen=True)
class SectorMatrices:
    K: np.ndarray
    L: np.ndarray
    spin_or_tr_sector: str
    source_labels: tuple[str,...]
    source_indices: np.ndarray
    equation_sign: int

    def gap_matrix(self)->np.ndarray:
        return self.equation_sign * (self.K @ self.L)


def physical_matrices(full:np.ndarray, loop:np.ndarray, basis:SourceBasis,
                      channel:str) -> tuple[np.ndarray,np.ndarray]:
    """Return normalized physical K and signed integrated L for PH or PP."""
    if channel not in ('PH','PP'):raise PhysicsInputError('Bad channel')
    T=basis.transform
    expected=T.shape[0]
    if full.shape!=(expected,expected) or loop.shape!=(expected,expected):
        raise PhysicsInputError('Full vertex or loop has incompatible dimensions')
    if not (np.isfinite(full).all() and np.isfinite(loop).all()):
        raise PhysicsInputError('Nonfinite native matrices')
    K=T.conj().T@full@T
    if channel=='PP':K *= 0.5
    L=T.conj().T@loop@T
    return K,L


def split_ph(K:np.ndarray,L:np.ndarray,basis:SourceBasis) -> dict[str,SectorMatrices]:
    if basis.parity is None:raise PhysicsInputError('PH source parity missing')
    sectors={}
    for name,p in [('TR_even',+1),('TR_odd',-1)]:
        idx=np.flatnonzero(basis.parity==p)
        sectors[name]=SectorMatrices(K[np.ix_(idx,idx)],L[np.ix_(idx,idx)],name,
                                     tuple(basis.labels[i] for i in idx),idx,+1)
    return sectors


def split_pp(K:np.ndarray,L:np.ndarray,basis:SourceBasis) -> dict[str,SectorMatrices]:
    if basis.sector is None:raise PhysicsInputError('PP Sz sector information missing')
    sectors={}
    for name in ('Sz0','+1','-1'):
        idx=np.flatnonzero(basis.sector==name)
        sectors[name]=SectorMatrices(K[np.ix_(idx,idx)],L[np.ix_(idx,idx)],name,
                                     tuple(basis.labels[i] for i in idx),idx,-1)
    return sectors


@dataclass
class Spectrum:
    eigenvalues: np.ndarray
    leading_real: float | None
    leading_vector: np.ndarray | None
    leading_response: np.ndarray | None
    second_real: float | None
    nonreal_count: int
    nonreal_max_real: float | None
    leading_residual: float | None
    physical_real_residual: float | None
    dominant_real_branch: bool
    # Orthonormal bases for the full leading RIGHT eigenspace and its L-image.
    # Their column-wise vectors are not individually canonical under degeneracy.
    leading_basis: np.ndarray | None = None
    response_basis: np.ndarray | None = None
    leading_multiplicity: int = 0
    subspace_eigenpair_relative_residual: float | None = None
    subspace_orthonormality: float | None = None
    response_subspace_rank: int = 0
    next_distinct_real: float | None = None

    def summary(self)->dict[str,Any]:
        return {
            'dimension':int(self.eigenvalues.size),
            'leading_real':self.leading_real,'second_real':self.second_real,
            'nonreal_count':self.nonreal_count,
            'nonreal_max_real_part':self.nonreal_max_real,
            'leading_eigenpair_relative_residual':self.leading_residual,
            'physical_real_source_relative_residual':self.physical_real_residual,
            'dominant_real_branch':self.dominant_real_branch,
            'leading_multiplicity':self.leading_multiplicity,
            'next_distinct_real':self.next_distinct_real,
            'subspace_eigenpair_relative_residual':self.subspace_eigenpair_relative_residual,
            'subspace_orthonormality':self.subspace_orthonormality,
            'response_subspace_rank':self.response_subspace_rank,
            'subspace_weights':'orthogonal projector diagonal / rank (basis invariant)',
        }


def _span_svd(vectors:np.ndarray, *, real:bool, expected_rank:int,
              rank_tolerance:float=1e-9)->np.ndarray:
    """Orthonormalize a right-eigenspace without assuming H is normal.

    SVD on the *span* avoids treating eigenvectors returned by eig() as an
    orthonormal basis.  For real PH sources, the real and imaginary parts of
    complex-gauge vectors span the underlying physical real eigenspace.
    """
    A=np.column_stack((vectors.real,vectors.imag)) if real else vectors
    U,sv,_=np.linalg.svd(A,full_matrices=False)
    if not len(sv) or sv[0]<=0:
        raise PhysicsInputError('Empty leading eigenspace')
    rank=int(np.count_nonzero(sv>sv[0]*rank_tolerance))
    if rank!=expected_rank:
        raise PhysicsInputError(f'Leading eigenspace rank {rank} != multiplicity '
                                f'{expected_rank}; nearly defective eigenproblem')
    Q=U[:,:rank].copy()
    return Q.real if real else Q


def projector_diagonal(basis:np.ndarray)->np.ndarray:
    """diag(P)/rank for the orthogonal projector P on span(basis)."""
    B=np.asarray(basis)
    if B.ndim!=2 or not 0<B.shape[1]<=B.shape[0]:
        raise PhysicsInputError('Expected nonempty orthonormal source subspace')
    gram=B.conj().T@B
    if np.linalg.norm(gram-np.eye(B.shape[1]))>1e-7:
        raise PhysicsInputError('Projector basis is not orthonormal')
    return np.sum(np.abs(B)**2,axis=1)/B.shape[1]


def signed_spectrum(sector:SectorMatrices, *, hermitian_source:bool,
                    spectral_tolerance:float=1e-7,
                    residual_tolerance:float=1e-8,
                    degeneracy_tolerance:float=1e-9) -> Spectrum:
    """Signed general KL spectrum, retaining the full leading right eigenspace.

    H need not be normal or Hermitian because L may be indefinite. Therefore
    naive eigenvector weights are gauge-dependent at degeneracy. We use the
    orthogonal projector onto the RIGHT eigenspace for invariant weights.
    """
    H=sector.gap_matrix()
    if not np.isfinite(H).all():
        raise PhysicsInputError('Nonfinite signed gap matrix')
    evals,vr=np.linalg.eig(H)
    scale=np.maximum(1.0,np.abs(evals.real))
    is_real=np.abs(evals.imag)<=spectral_tolerance*scale
    real_indices=np.flatnonzero(is_real)
    nonreal=evals[~is_real]
    nonreal_max=float(np.max(nonreal.real)) if len(nonreal) else None
    if not len(real_indices):
        return Spectrum(evals,None,None,None,None,len(nonreal),nonreal_max,
                        None,None,False)
    order=real_indices[np.argsort(evals.real[real_indices])[::-1]]
    lam=float(evals[order[0]].real)
    # Degeneracy and "approximately real" are DIFFERENT tolerances.
    # A 1e-7 ABSOLUTE degeneracy window would merge distinct ~1e-7 Cooper
    # eigenvalues in small fixtures. Base it on the actual spectral radius.
    spectral_radius=float(np.max(np.abs(evals)))
    tol=max(1e-12,degeneracy_tolerance*max(abs(lam),spectral_radius))
    selected=order[np.abs(evals.real[order]-lam)<=tol]
    m=len(selected)
    Q=_span_svd(vr[:,selected],real=hermitian_source,expected_rank=m)
    # The selected eigenvalues may be slightly split numerically.  The subspace
    # residual checks invariance, not the arbitrary orientation of its columns.
    hnorm=max(float(np.linalg.norm(H)),1e-30)
    subspace_res=float(np.linalg.norm(H@Q-lam*Q)/(hnorm*np.linalg.norm(Q)))
    ortho_res=float(np.linalg.norm(Q.conj().T@Q-np.eye(m)))
    vec=Q[:,0].copy()
    if not hermitian_source:
        j=int(np.argmax(np.abs(vec)))
        if abs(vec[j])>0:vec*=np.exp(-1j*np.angle(vec[j]))
    single_res=float(np.linalg.norm(H@vec-lam*vec)/hnorm)
    real_res=single_res if hermitian_source else None
    # The induced response space is L E. Its canonical object is its span, not
    # the collection of unnormalized individual images L v_i.
    responses=sector.L@Q
    try:
        RQ=_span_svd(responses,real=hermitian_source,expected_rank=m)
        rrank=m
    except PhysicsInputError:
        RQ=None
        rrank=0
    dominant=(subspace_res<residual_tolerance and
              single_res<residual_tolerance and
              ortho_res<residual_tolerance and rrank==m and
              (nonreal_max is None or nonreal_max<lam-tol))
    next_distinct=float(evals[order[m]].real) if len(order)>m else None
    # second_real preserves v0.1 meaning (including a degenerate second mode).
    second=float(evals[order[1]].real) if len(order)>1 else None
    return Spectrum(evals,lam,vec,sector.L@vec,second,
                    len(nonreal),nonreal_max,single_res,real_res,dominant,
                    Q,RQ,m,subspace_res,ortho_res,rrank,next_distinct)


def kernel_sanity(K:np.ndarray,L:np.ndarray,basis:SourceBasis,
                  transverse:bool=False)->dict[str,float]:
    """Numerical symmetry metrics, independent of leading-order labels."""
    checks={'K_hermiticity':_hermiticity(K),'L_hermiticity':_hermiticity(L)}
    if basis.parity is not None:
        even=np.flatnonzero(basis.parity==+1)
        odd=np.flatnonzero(basis.parity==-1)
        checks['K_TR_cross']=_cross(K,even,odd)
        checks['L_TR_cross']=_cross(L,even,odd)
    if transverse:
        n=K.shape[0]//2
        for name,A in [('K',K),('L',L)]:
            checks[name+'_xx_minus_yy']=_ratio(A[:n,:n]-A[n:,n:],A)
            checks[name+'_xy_plus_yx']=_ratio(A[:n,n:]+A[n:,:n],A)
    if basis.sector is not None:
        for name,A in [('K',K),('L',L)]:
            for s,t in (('Sz0','+1'),('Sz0','-1'),('+1','-1')):
                checks[f'{name}_cross_{s}_{t}']=_cross(A,
                     np.flatnonzero(basis.sector==s),np.flatnonzero(basis.sector==t))
    return checks


def weights_by_source_type(vector:np.ndarray,labels:tuple[str,...])->dict[str,float]:
    """Coefficient-norm decomposition, not an expectation value or real current."""
    if vector is None or len(vector)!=len(labels):
        raise PhysicsInputError('Source labels/coefficients mismatch')
    v2=abs(vector)**2
    tot=float(v2.sum())
    if tot<=0:raise PhysicsInputError('Zero source vector')
    summary:dict[str,float]={}
    for w,label in zip(v2/tot,labels):
        comp,typ,*_=label.split(':')
        key=(('rho' if comp=='0' else 'Sz')+'_'+typ) if comp in ('0','1') else f'{comp}_{typ}'
        summary[key]=summary.get(key,0.)+float(w)
    return summary


def compare_d_m(d:np.ndarray,m:np.ndarray,basis:SourceBasis)->float:
    T=basis.transform
    a=T.conj().T@d@T;b=T.conj().T@m@T
    return _ratio(a-b,a)


def no_soc_su2_regression(d:np.ndarray, loop:np.ndarray,nff:int)->dict[str,float]:
    """Matrix-level SU(2) spin xyz regression in the no-SOC limit."""
    spin=np.array([[1,0,0,1],[0,1,1,0],[0,-1j,1j,0],[1,0,0,-1]],complex)/np.sqrt(2)
    out={}
    for name,A in [('K',d),('L',loop)]:
        tensor=A.reshape(4,nff,4,nff)
        spinm=[np.einsum('a,ambn,b->mn',v.conj(),tensor,v) for v in spin]
        for index,axis in ((1,'x'),(2,'y')):
            out[f'{name}_spin_{axis}_vs_z']=_ratio(spinm[index]-spinm[3],spinm[3])
    return out


__all__=['PhysicsInputError','Geometry','SourceBasis','longitudinal_basis',
         'transverse_basis','cooper_basis','native_full_matrix','native_loop',
         'physical_matrices','split_ph','split_pp','SectorMatrices','Spectrum',
         'signed_spectrum','kernel_sanity','weights_by_source_type',
         'compare_d_m','no_soc_su2_regression']
