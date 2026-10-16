#!/usr/bin/env python3
# coding: utf-8
"""Independent finite-M real/imaginary bond audit for divERGe TUFRG.

PURPOSE
=======
Reclassify saved ph-rho and ph-sz right singular vectors, without rerunning flow.
For an anchored TU bond (R,af,at):
    B(n;R,af,at) = exp(2*pi*i*q.n) f(R,af,at)
where at sits in cell n and af in cell n+R, q is reduced momentum.
At M=-M (mod G), Hermiticity requires:
    f(-R,at,af) = exp(-2*pi*i*q.R) * conj(f(R,af,at))

Define anti-linear involution H_q f[R,af,at] =
    exp(-2*pi*i*q.R) conj(f[-R,at,af]).
The two physical Hermitian quadratures from an arbitrary complex TU mode are
    h0=(f+H_q f)/2; h90=(i*f+H_q(i*f))/2.
For arbitrary global mode phase, the pair (h0,h90) rotates as a real O(2)
frame. Sum of squared real/imaginary parts is therefore phase-invariant.

Outputs are NN real/imaginary BOND CHARACTER, not physical current response,
not susceptibility, and not proof of equilibrium loop currents. Assumes TU
mode entries are in the stated anchored density-bond convention; validate
C3 covariance and synthetic tests. A singular vector of KL is not necessarily
an eigenvector of the PRR linearized gap equation.

This script does not modify existing diagnosis or any O.dvg files.
"""
from __future__ import annotations
import argparse
import ast
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence
import numpy as np
import pandas as pd
import TUfRG_model as km

Q = {'M1': np.array([0.5,0.0]), 'M2': np.array([0.0,0.5]),
     'M3': np.array([0.5,0.5])}
C3_A = np.array([[-1,-1],[1,0]], dtype=int)
C3_PERM = [1,2,0]        # A -> B -> C -> A, grounded in TUfRG_model geometry
C3_CELL_SHIFT = np.array([[-1,0],[-1,-1],[-2,0]], dtype=int)
C3_NEXT = {'M1':'M2','M2':'M3','M3':'M1'}
SECTORS = ('ph-rho','ph-sz')


def write_csv(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True,exist_ok=True)
    if not rows:
        path.write_text('')
        return
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w',newline='') as io:
        wr=csv.DictWriter(io,fieldnames=keys);wr.writeheader();wr.writerows(rows)


def ff_geometry(R, a, b):
    """Prepare structural geometry and strict NN masks from the actual O maps."""
    R=np.asarray(R,dtype=int)
    if R.ndim!=2 or R.shape[1]!=3:
        raise ValueError(f'Expected mi_to_R (nff,3), got {R.shape}')
    a=np.asarray(a,dtype=int).reshape(-1)
    b=np.asarray(b,dtype=int).reshape(-1)
    n=len(a)
    if R.shape[0]!=n or len(b)!=n:raise ValueError('TU arrays have inconsistent sizes')
    keys={(tuple(R[i]),int(a[i]),int(b[i])):i for i in range(n)}
    if len(keys)!=n:raise ValueError('Duplicate TU form-factor keys')
    rev=np.array([keys.get((tuple(-R[i]),int(b[i]),int(a[i])),-1) for i in range(n)],int)
    if np.any(rev<0):
        raise ValueError(f'{np.sum(rev<0)} form factors lack reverse in full TU basis')
    if not np.array_equal(rev[rev],np.arange(n)):
        raise ValueError('Reverse map is not an involution')
    lattice,pos=km.kagome_lattice_and_positions()
    d=np.array([np.linalg.norm(km.tu_ff_displacement_vector(
        lattice,pos,int(R[i,0]),int(R[i,1]),int(a[i]),int(b[i]))[:2]) for i in range(n)])
    nn=np.abs(d-0.5)<1e-5
    if nn.sum()!=12:
        raise ValueError(f'Expected 12 directed NN bonds, found {nn.sum()} / {n}')
    if not np.array_equal(nn[rev],nn):raise ValueError('NN set not reversal-closed')
    # C3: source ofrom is in n+R, anchor oto in n.
    c3_index=np.full(n,-1,int)
    for i in range(n):
        Rp=C3_A@R[i,:2]+C3_CELL_SHIFT[int(a[i])]-C3_CELL_SHIFT[int(b[i])]
        key=((int(Rp[0]),int(Rp[1]),0),C3_PERM[int(a[i])],C3_PERM[int(b[i])])
        c3_index[i]=keys.get(key,-1)
    if np.any(c3_index<0) or len(set(c3_index))!=n:
        raise ValueError('C3 map fails to permute full TU basis (geometry mismatch)')
    if not np.array_equal(nn[c3_index],nn):
        raise ValueError('C3 mapping does not preserve NN mask')
    return rev,nn,c3_index


def hermitian_map(f: np.ndarray, R, rev, q):
    f=np.asarray(f,dtype=complex).reshape(-1)
    ph=np.exp(-2j*np.pi*(np.asarray(R)[:,:2]@np.asarray(q)))
    return ph*np.conj(f[rev])


def two_quadratures(f,R,rev,q):
    h0=.5*(f+hermitian_map(f,R,rev,q))
    h90=.5*(1j*f+hermitian_map(1j*f,R,rev,q))
    assert np.linalg.norm(h0-hermitian_map(h0,R,rev,q))<1e-7
    assert np.linalg.norm(h90-hermitian_map(h90,R,rev,q))<1e-7
    return h0,h90


def weights(f,R,rev,nn,q):
    f=np.asarray(f,dtype=complex).reshape(-1)
    nrm=float(np.vdot(f,f).real)
    if nrm<1e-20:raise ValueError('Mode norm is zero')
    h0,h90=two_quadratures(f,R,rev,q)
    den=sum(float(np.vdot(h,h).real) for h in (h0,h90))
    if not np.isclose(den,nrm,atol=1e-8,rtol=1e-8):
        raise AssertionError(f'Hermitian quadrature norm mismatch {den}/{nrm}')
    real=sum(float(np.sum(np.real(h[nn])**2)) for h in (h0,h90))/den
    imag=sum(float(np.sum(np.imag(h[nn])**2)) for h in (h0,h90))/den
    nnb=float(np.vdot(f[nn],f[nn]).real)/nrm
    if not np.isclose(real+imag,nnb,atol=1e-8):
        raise AssertionError('NN real+imag weights do not reconstruct total NN norm')
    return dict(nn_total_weight=nnb,nn_real_bond_weight=real,
                nn_imag_bond_weight=imag,
                nn_imag_fraction_of_nn=imag/max(nnb,1e-300),
                nn_real_fraction_of_nn=real/max(nnb,1e-300),
                nn_onsite_excluded_note='NN weights normalized to full form-factor norm')


def c3_rotate(f,R,ato,c3_index,qdst):
    """Rotate cell-anchored bond coefficients to the next M point."""
    dst=np.zeros_like(f)
    # at = anchor orbital; n'=A n + d_at
    anchor_shift=C3_CELL_SHIFT[np.asarray(ato,dtype=int)]
    phases=np.exp(-2j*np.pi*(anchor_shift@qdst))
    dst[c3_index]=np.asarray(f,dtype=complex)*phases
    return dst


def overlap(f,g):
    nf=np.linalg.norm(f);ng=np.linalg.norm(g)
    return float(abs(np.vdot(f,g))/(nf*ng)) if nf>0 and ng>0 else float('nan')


def geom_self_test(verbose=True):
    """Synthetic 141-FF lattice fixture; no divERGe/output or stored modes needed."""
    lattice,pos=km.kagome_lattice_and_positions()
    arr=[]
    for x in range(-2,3):
        for y in range(-2,3):
            for a in range(3):
                for b in range(3):
                    dr=km.tu_ff_displacement_vector(lattice,pos,x,y,a,b)
                    if np.linalg.norm(dr[:2])<2.01-1e-10:
                        arr.append(((x,y,0),a,b))
    R=np.array([z[0] for z in arr],int); aa=np.array([z[1] for z in arr],int);bb=np.array([z[2] for z in arr],int)
    if len(arr)!=141:raise AssertionError(f'Synthetic TU FF expected 141, found {len(arr)}')
    rev,nn,c3=ff_geometry(R,aa,bb)
    # C3 is consistent with model NN hopping, including SOC and spin.
    hops={}
    for mr,a,b,s1,s2,amp in km.spinful_hoppings(t=1,lambda_nn=.5,lambda_nnn=0,include_mu=False):
        if s1==s2:
            key=(tuple(mr),a,b,s1)
            hops[key]=hops.get(key,0j)+complex(*amp)
    for (r,a,b,s),t in hops.items():
        dst=C3_A@np.asarray(r[:2])+C3_CELL_SHIFT[b]-C3_CELL_SHIFT[a]
        key=((int(dst[0]),int(dst[1]),0),C3_PERM[a],C3_PERM[b],s)
        if not np.isclose(t,hops.get(key,np.nan)):
            raise AssertionError('Hamiltonian NN C3 test failed')
    # Real/imaginary bond should not swap upon q/rotation changes.
    halfphase=np.exp(-2j*np.pi*(R[:,:2]@Q['M1']))
    candidates=np.where(nn & (np.real(halfphase)<-.5))[0]
    if len(candidates)==0:raise AssertionError('Cannot find M1 bond with odd reciprocal phase')
    base=int(candidates[0])
    for kind in ['real','imag']:
        f=np.zeros(len(R),complex)
        f[base]=1. if kind=='real' else 1j
        phi=np.exp(-2j*np.pi*np.dot(Q['M1'],R[base,:2]))
        f[rev[base]]=phi*np.conj(f[base])
        f_initial=f.copy()
        for label in ('M1','M2','M3'):
            h=hermitian_map(f,R,rev,Q[label])
            if not np.allclose(h,f):raise AssertionError(f'Synthetic {kind} not Hermitian at {label}')
            w=weights(f,R,rev,nn,Q[label])
            if (w['nn_imag_fraction_of_nn']>1e-10 if kind=='real' else
                    w['nn_imag_fraction_of_nn']<1-1e-10):
                raise AssertionError(f'Synthetic {kind} classification failure at {label}: {w}')
            rng=np.random.default_rng(157)
            theta=float(rng.uniform(-np.pi,np.pi))
            wr=weights(np.exp(1j*theta)*f,R,rev,nn,Q[label])
            if abs(wr['nn_imag_fraction_of_nn']-w['nn_imag_fraction_of_nn'])>1e-10:
                raise AssertionError('Global mode phase changed imaginary bond weight')
            if label=='M1' and kind=='real':
                # The OLD reversal-only parity mislabels this physical real CBO.
                old_antisym=abs(f[base]-f[rev[base]])**2 / (abs(f[base]-f[rev[base]])**2+abs(f[base]+f[rev[base]])**2)
                if not np.isclose(old_antisym,1):raise AssertionError('Expected legacy false-positive absent')
            new=C3_NEXT[label]
            f=c3_rotate(f,R,bb,c3,Q[new])
        if not np.allclose(f, f_initial, atol=1e-10):
            raise AssertionError('C3 cubed must return the original bond source')
    if verbose:print('[PASS] 141-FF synthetic geometry, hopping C3, real/imag, Hermiticity, global-phase tests')


def load_O(path):
    import diverge.output as dvo
    return dvo.read(str(path))


def collect_case(case:Path,out:Path,nmodes:int,sector_set:tuple):
    snap=case/'snapshots'/'final'
    for rel in ('O.dvg','instability/instability_summary.csv','instability/instability_modes.npz'):
        if not (snap/rel).is_file():raise FileNotFoundError(snap/rel)
    metadata=json.loads((snap/'metadata.json').read_text())
    rp=metadata.get('model_summary',{}).get('metadata',{}).get('resolved_params',{})
    if rp:
        if not (abs(float(rp.get('t',-1))-1)<1e-9 and
                abs(float(rp.get('lambda_nn',-1))-0.5)<1e-9 and
                abs(float(rp.get('lambda_nnn',-1)))<1e-9):
            raise ValueError(f'Unexpected model settings in {case}: {rp}')
    obj=load_O(snap/'O.dvg')
    R=np.asarray(obj.mi_to_R,dtype=int)
    af=np.asarray(obj.mi_to_ofrom,dtype=int)
    at=np.asarray(obj.mi_to_oto,dtype=int)
    rev,nn,c3=ff_geometry(R,af,at)
    summary=pd.read_csv(snap/'instability/instability_summary.csv')
    summary=summary[(summary['family']=='ph') & summary['q_label'].isin(Q) &
                    summary['sector'].isin(sector_set) & (summary['mode']<nmodes)]
    if summary.empty:raise RuntimeError(f'{case.name}: no matching PH modes')
    rows=[];vectors={}
    with np.load(snap/'instability/instability_modes.npz') as modes:
        for rr in summary.itertuples(index=False):
            key=str(rr.mode_key)
            if key not in modes:raise KeyError(f'Missing vector {key}')
            f=np.asarray(modes[key],complex).reshape(-1)
            if len(f)!=len(rev):raise RuntimeError(f'Mode {key}: len {len(f)} expected {len(rev)}')
            label=str(rr.q_label);sec=str(rr.sector);mode=int(rr.mode)
            q_from_row=np.asarray(ast.literal_eval(str(rr.q_reduced))[:2],float)
            if not np.allclose(np.mod(q_from_row,1),Q[label],atol=1e-10):
                raise RuntimeError(f'Bad q mapping for {key}: {q_from_row} vs {Q[label]}')
            w=weights(f,R,rev,nn,Q[label])
            rows.append(dict(case=case.name,sector=sec,q=label,mode=mode,mode_key=key,
                             strength_svd_KL=float(rr.strength_svd_KL),
                             vector_kind='right_singular_vector_KL',**w))
            vectors[(sec,label,mode)]=f
    cov=[]
    for sec in sector_set:
        for src in ('M1','M2','M3'):
            dst=C3_NEXT[src]
            modes_src=[(m,f) for (s,q,m),f in vectors.items() if s==sec and q==src]
            modes_dst=[(m,f) for (s,q,m),f in vectors.items() if s==sec and q==dst]
            for i,f in modes_src:
                rotated=c3_rotate(f,R,at,c3,Q[dst])
                ovs={m:overlap(rotated,f2) for m,f2 in modes_dst}
                best=max(ovs,key=ovs.get) if ovs else None
                sw=weights(rotated,R,rev,nn,Q[dst])
                ov0=overlap(rotated,vectors[(sec,dst,0)]) if (sec,dst,0) in vectors else float('nan')
                cov.append(dict(case=case.name,sector=sec,q_src=src,q_dst=dst,mode_src=i,
                                best_dest_mode=best,best_mode_overlap=(ovs[best] if best is not None else float('nan')),
                                leading_dest_overlap=ov0,
                                rotated_imag_weight=sw['nn_imag_bond_weight'],
                                rotated_imag_fraction=sw['nn_imag_fraction_of_nn']))
    # Explicit rotational scalar test of our projector. Independent of eigenmodes at next q.
    maxcov=0.
    for (sec,label,m),f in vectors.items():
        dst=C3_NEXT[label]
        rr=c3_rotate(f,R,at,c3,Q[dst])
        a=weights(f,R,rev,nn,Q[label]); b=weights(rr,R,rev,nn,Q[dst])
        maxcov=max(maxcov,abs(a['nn_imag_bond_weight']-b['nn_imag_bond_weight']),
                   abs(a['nn_real_bond_weight']-b['nn_real_bond_weight']))
    if maxcov>1e-9:
        raise AssertionError(f'Corrected projector failed C3 covariance, difference {maxcov}')
    write_csv(out/'corrected_mode_weights.csv',rows)
    write_csv(out/'mode_rotation_overlap.csv',cov)
    report=dict(case=case.name,nff=len(rev),nn_bond_count=int(nn.sum()),modes=len(rows),
                sector=list(sector_set),c3_projector_max_abs_error=maxcov,
                note='Hermitian NN real/imag character of KL right singular vectors; NOT an LCO susceptibility')
    (out/'audit_report.json').write_text(json.dumps(report,indent=2))
    print(f"[PASS] {case.name}: nff={len(rev)}, modes={len(rows)}, C3 projector err={maxcov:.2e}",flush=True)
    for r in rows:
        if r['mode']==0:
            print(f" {r['sector']:7s} {r['q']:2s}: NN={r['nn_total_weight']:.5f}"
                  f" real={r['nn_real_bond_weight']:.5f} imag={r['nn_imag_bond_weight']:.5f}"
                  f" imag/NN={r['nn_imag_fraction_of_nn']:.4f}",flush=True)
    return rows


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--self-test',action='store_true')
    p.add_argument('--root',type=Path,default=Path('lnn0p5-prrflow-nk24-nkf12-11pts'))
    p.add_argument('--out',type=Path,default=Path('lco_corrected_05'))
    p.add_argument('--tags',nargs='+',default=['U1p0_V0p5','U2p0_V1p5','U3p0_V1p5'])
    p.add_argument('--n-modes',type=int,default=4)
    p.add_argument('--sectors',nargs='+',choices=SECTORS,default=list(SECTORS))
    args=p.parse_args()
    geom_self_test()
    if args.self_test:return
    root=args.root.resolve();out=args.out.resolve()
    if not root.is_dir():raise FileNotFoundError(root)
    out.mkdir(parents=True,exist_ok=True)
    allrows=[]
    for tag in args.tags:
        matches=[c for c in root.iterdir() if c.is_dir() and c.name.endswith('_'+tag)]
        if len(matches)!=1:raise RuntimeError(f'Expected one case for {tag}, got {len(matches)}')
        rows=collect_case(matches[0],out/tag,args.n_modes,tuple(args.sectors))
        allrows.extend(rows)
    write_csv(out/'all_cases_corrected_weights.csv',allrows)
    print('Saved:',out/'all_cases_corrected_weights.csv')

if __name__=='__main__':main()
