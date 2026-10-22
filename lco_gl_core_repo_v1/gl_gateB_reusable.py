#!/usr/bin/env python3
"""Reusable, metadata-checked, *restricted* GL quartic analysis for spinful 3-M LCO.

Inputs: the source-audited TU-FF O-header map and a Stage3/4B diagnosis ZIP.
No divERGe runtime and no O.dvg tensors are required. No FRG flow is started.

Outputs: the fermionic determinant's finite-T Z1,Z2 for *fixed gap
operators*, plus stringent symmetry checks.  These are NOT full interacting
Hubbard free-energy coefficients; the HS cost, secondary order relaxation,
and fluctuation renormalization are outside this solver.
"""
from __future__ import annotations
import argparse, json, zipfile, hashlib
from pathlib import Path
import numpy as np
import gl_gateA_build_W as build
from gl_quartic import quartic_loop
from gl_thermo import choose_mu

Q = build.Q

def read_parameters(z, snapshot):
    path = f'snapshots/{snapshot}/diagnosis.json'
    if path not in z.namelist():
        raise ValueError(f'Missing {path}; available snapshot dirs: ' + str(sorted({p.split('/')[1] for p in z.namelist() if p.startswith('snapshots/')})))
    d = json.loads(z.read(path))
    p = d['resolved_parameters']
    required = ('t', 'lambda_nn', 'lambda_nnn', 'target_filling')
    if any(k not in p for k in required):
        raise ValueError('Missing physical-model metadata: '+','.join(k for k in required if k not in p))
    p = {k: (float(v) if isinstance(v, (int,float)) else v) for k,v in p.items()}
    if p.get('lambda_rashba',0) != 0:
        raise ValueError('Rashba mixing requires a different spinful model and source basis')
    if not 0. < p['target_filling'] < 1.: raise ValueError('Invalid filling')
    if p.get('filling_mode') != 'fixed_filling':
        raise ValueError('Only fixed_filling normal-state metadata is accepted')
    return p

def read_modes(archive, snapshot, geometry):
    with zipfile.ZipFile(archive) as z:
        p = read_parameters(z, snapshot)
        fs = {q: build.load_source_vector(z,snapshot,q,geometry)[0] for q in Q}
    # Real signed basis + rotationally aligned gauge.  The sign of any
    # eigenvector is arbitrary; quartic self/mixed coefficients are invariant.
    ws0={q:build.build_W((0.,0.),geometry,q,f) for q,f in fs.items()}
    P=build.site_C6_perm()
    sgn=[1]
    scores=[]
    for src,tgt in zip(Q[:2],Q[1:]):
        X=P@ws0[src]@P.conj().T
        Y=ws0[tgt]
        c=np.vdot(X,Y).real/(np.linalg.norm(X)*np.linalg.norm(Y))
        if abs(abs(c)-1)>1e-7:
            raise ValueError(f'Input eigenvectors are not C6 symmetry partners: {src}->{tgt}: {c}')
        sgn.append(sgn[-1]*(1 if c>=0 else -1))
        scores.append(float(c))
    fs={q:fs[q]*sgn[i] for i,q in enumerate(Q)}
    # Full K-space Hermiticity, TR, primitive translations and 60-degree rotation.
    W0={q:build.build_W((0,0),geometry,q,fs[q]) for q in Q}
    closing=P@W0[Q[2]]@P.conj().T
    close_err=np.linalg.norm(closing-W0[Q[0]])/np.linalg.norm(W0[Q[0]])
    if close_err>1e-7:raise ValueError(f'Inconsistent C6 sign gauge: {close_err}')
    Utr=np.block([[np.zeros((12,12)),np.eye(12)],[-np.eye(12),np.zeros((12,12))]])
    symmetry_errors=[]
    for K in ((.137,.224),(.371,.119)):
        kp,Pbk=build.folded_point_group_C6(K)
        for i,q in enumerate(Q):
            W=build.build_W(K,geometry,q,fs[q]); denom=np.linalg.norm(W)
            Wm=build.build_W(tuple(-x for x in K),geometry,q,fs[q])
            Wrot=build.build_W(kp,geometry,Q[(i+1)%3],fs[Q[(i+1)%3]])
            err_herm=np.linalg.norm(W-W.conj().T)/denom
            err_tr=np.linalg.norm(W+Utr@Wm.conj()@Utr.conj().T)/denom
            err_c6=np.linalg.norm(Pbk@W@Pbk.conj().T-Wrot)/denom
            err_spin=(np.linalg.norm(W[:12,12:])+np.linalg.norm(W[12:,:12]))/denom
            symmetry_errors.append([float(err_herm),float(err_tr),float(err_c6),float(err_spin)])
    maxerrors=np.max(symmetry_errors,axis=0)
    if np.any(maxerrors>1e-8):raise ValueError(f'Mean-field operator audit failed (H/TR/C6/Sz): {maxerrors}')
    return p,fs,{'C6_input_signed_overlaps':scores,'gauge_signs':dict(zip(Q,sgn)),
                 'C6_closure_error':float(close_err),
                 'max_errors_Hermiticity_TR_C6_Sz':maxerrors.tolist()}

def normal_hoppings(p):
    # nff may change with Nk/Nkf, but H0 is independent of momentum grid.
    return build.pre.model.spinful_hoppings(t=p['t'],mu=0.,
        lambda_nn=p['lambda_nn'],lambda_nnn=p['lambda_nnn'],lambda_rashba=0.)

def make_grid(n,geometry,fs,p):
    if int(n)<2:raise ValueError('Folded momentum mesh must have >=2 points per direction')
    kgrid=[(i/n,j/n) for i in range(n) for j in range(n)]
    hops=normal_hoppings(p)
    H=np.asarray([build.pre.folded_h0(K,hops) for K in kgrid])
    W=np.asarray([[build.build_W(K,geometry,q,fs[q]) for q in Q] for K in kgrid])
    assert H.shape==(n*n,24,24) and W.shape==(n*n,3,24,24)
    assert np.linalg.norm(H-H.swapaxes(-1,-2).conj())<1e-8
    return H,W

def compute(map_path,archive,snapshot,T,mesh,nmat,mu_mode='normal_fixed_density',
            fixed_mu=None):
    if T<=0 or nmat<1:raise ValueError('Use T>0 and nmat>=1')
    geometry=build.load_geometry(map_path)
    params,fs,audit=read_modes(archive,snapshot,geometry)
    H,W=make_grid(mesh,geometry,fs,params)
    E=np.linalg.eigvalsh(H)
    target=float(params['target_filling'])*24
    mu=choose_mu(E,T,expected_per_sc=target) if mu_mode=='normal_fixed_density' else float(fixed_mu)
    filled=np.mean(np.sum(1/(1+np.exp(np.clip((E-mu)/T,-700,700))),axis=1))
    coef=quartic_loop(H,W,T,mu,nmat)
    z1=np.diag(coef)
    z2=np.array([coef[0,1],coef[1,2],coef[2,0]])
    maxabs=max(1.,np.max(abs(coef)))
    spread=max(np.ptp(z1),np.ptp(z2)) / maxabs
    if spread>1e-7: raise RuntimeError(f'Unexpected C6 symmetry breaking in GL coefficients: {spread}')
    zz1=float(np.mean(z1)); zz2=float(np.mean(z2))
    return {
        'status':'PASS_RESTRICTED_FERMIONIC_GL',
        'analysis_scope':'fixed_FRG_gap_subspace__fermionic_loop_only__no_HS_or_secondary_order',
        'provenance':{'map_file':str(map_path),'map_sha256':hashlib.sha256(Path(map_path).read_bytes()).hexdigest(),
                      'archive_file':str(archive),'archive_sha256':hashlib.sha256(Path(archive).read_bytes()).hexdigest(),
                      'snapshot':snapshot,'source_labels_validated':True},
        'model':params,'C6_symmetry_audit':audit,
        'sampling':{'T':T,'folded_kmesh':mesh,'n_positive_Matsubara':nmat,
                    'chemical_potential_mode':mu_mode,'mu':mu,
                    'target_electrons_per_2by2_supercell':target,'computed_electrons':float(filled)},
        'quartic_per_2by2_supercell':{'Z1':zz1,'Z2':zz2,'Z1_minus_Z2':zz1-zz2,
            'Z1_by_M':z1.tolist(),'Z2_by_pair':z2.tolist(),
            'C6_relative_coefficient_spread':float(spread),
            'F4_single_minus_triple_at_total_eta_1':float((zz1-zz2)/3),
            'restricted_local_preference':'triple_M' if zz1>zz2 else 'single_M' if zz1<zz2 else 'undetermined',
            'quartic_single_stable':zz1>0,'quartic_equal_triple_stable':zz1+2*zz2>0}}

def main():
    p=argparse.ArgumentParser(description='Source-verified reusable FRG spinful SOC 3M restricted GL solver')
    p.add_argument('--map',required=True,help='Exact FF map extracted from matching O.dvg header')
    p.add_argument('--archive',required=True,help='Stage3/Stage4B diagnosed snapshot ZIP with full gap vectors')
    p.add_argument('--snapshot',required=True,help='e.g. iter_0074')
    p.add_argument('--T',type=float,required=True)
    p.add_argument('--mesh',type=int,required=True)
    p.add_argument('--nmat',type=int,default=256)
    p.add_argument('--fixed-mu',type=float,default=None,help='Optional grand canonical fixed mu, otherwise fill normal state to target density')
    p.add_argument('--out',required=True)
    args=p.parse_args()
    mm='fixed_mu' if args.fixed_mu is not None else 'normal_fixed_density'
    report=compute(args.map,args.archive,args.snapshot,args.T,args.mesh,args.nmat,mm,args.fixed_mu)
    out=Path(args.out);out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(report,indent=2)+'\n')
    print('RESTRICTED GL PASS',report['quartic_per_2by2_supercell']['Z1_minus_Z2'], 'output:',out)
if __name__=='__main__':main()
