#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Single read-only TUfRG physical-diagnosis entry point, v1.0 (Stage 3).

Normal use (PACE):
  python TUfRG_diagnose.py --case /path/to/snapshots/final \
      --out-dir /path/to/safe/new/diagnosis \
      --diverge-src /path/to/official/diverge/src \
      --binary /path/to/installed/diverge/libdivERGe.so

The original T.dvg, O.dvg, and metadata.json are NEVER modified. The native
full-vertex projection is computed from raw P/C/D using the audited C bridge
unless an exact matching, provenance-checked --projected-prefix is supplied.

Only spin-conserving intrinsic KM (including zero SOC) is supported, on an
uncompressed 2D even square mesh. Longitudinal and transverse PH are checked
at configurable self-inverse q points (Gamma and/or M); Cooper is checked at Gamma. This is a sector comparison on
SIGNED flow-integrated loops, NOT an automatic phase-diagram classifier.

Companion files: TUfRG_archive.py, TUfRG_physics.py, TUfRG_restore.c.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
from typing import Any

import numpy as np

import TUfRG_archive as archive
import TUfRG_physics as ph


class DiagnosisError(RuntimeError):
    pass


def _read_archived_fixture(T: Path, destination: Path) -> None:
    """Reproduce the tested Stage-2 export_model_fixture.py ABI (v0.10 only)."""
    h = np.fromfile(T, dtype='<i8', count=128)
    if h.size != 128 or h[0] != 145 or h[1] != 2048 or h[2] != 1024:
        raise DiagnosisError('Unsupported T.dvg header/ABI')
    no, ns = int(h[25]), int(h[46])
    nk = list(map(int,h[4:7])); nkf = list(map(int,h[7:10]))
    if no != 3 or ns != 2 or h[45] != 0 or nk[2] != 1 or nkf[2] != 1:
        raise DiagnosisError('The native fixture exporter requires 2D explicit-spin kagome')
    lattice = np.frombuffer(h[26:35].tobytes(), dtype='<f8').reshape(3,3)
    pos = np.fromfile(T,dtype='<f8',count=3*no,offset=int(h[35])).reshape(no,3)
    if not np.isfinite(lattice).all() or not np.isfinite(pos).all():
        raise DiagnosisError('Invalid lattice/orbital positions')
    def q_at(b: bytes, offset: int) -> int:
        return struct.unpack_from('<q', b, int(offset))[0]
    def z_at(b: bytes, offset: int) -> tuple[float,float]:
        return struct.unpack_from('<2d', b, int(offset))
    hops=[];verts=[]
    with T.open('rb') as f:
        f.seek(int(h[43]));hb=f.read(int(h[44]))
        for i in range(int(h[42])):
            b=hb[i*int(h[108]):(i+1)*int(h[108])]
            R=struct.unpack_from('<3q',b,int(h[102]))
            fields=[q_at(b,h[j]) for j in (103,104,105,106)]
            hops.append((*R,*fields,*z_at(b,h[107])))
        f.seek(int(h[48]));vb=f.read(int(h[49]))
        for i in range(int(h[47])):
            b=vb[i*int(h[118]):(i+1)*int(h[118])]
            channel=chr(b[int(h[109])])
            R=struct.unpack_from('<3q',b,int(h[110]))
            fields=[q_at(b,h[j]) for j in range(111,117)]
            verts.append((channel,*R,*fields,*z_at(b,h[117])))
    with destination.open('w',encoding='utf8') as f:
        f.write(f'{nk[0]} {nkf[0]} {ns}\n')
        for a in lattice:f.write(' '.join(f'{v:.17g}' for v in a)+'\n')
        for a in pos:f.write(' '.join(f'{v:.17g}' for v in a)+'\n')
        f.write(f'{len(hops)}\n')
        for rec in hops:f.write(' '.join(str(x) if isinstance(x,int) else f'{x:.17g}' for x in rec)+'\n')
        f.write(f'{len(verts)}\n')
        for rec in verts:f.write(' '.join(str(x) if isinstance(x,(int,str)) else f'{x:.17g}' for x in rec)+'\n')


def _sha(path:Path)->str:
    return hashlib.sha256(path.read_bytes()).hexdigest() # only small T/library/C; NEVER multi-GB O


def _prefix_complete(prefix:Path, qmap:dict[str,int],nff:int)->bool:
    wanted=[('p',qmap['Gamma'])]
    for q in qmap.values():wanted.extend([('d',q),('m',q)])
    return all((Path(f'{prefix}_{c}_q{q}.bin').is_file() and
                Path(f'{prefix}_{c}_q{q}.bin').stat().st_size==16*(4*nff)**2)
               for c,q in wanted)


def _validated_external_prefix(prefix:Path,case:Path, archive_report:dict[str,Any],
                               binary:Path|None)->None:
    """External projected input must be tied to this T/O snapshot, not guessed."""
    manifest=prefix.parent/'manifest.json'
    if not manifest.is_file():
        raise DiagnosisError(f'Missing projection provenance manifest: {manifest}')
    data=json.loads(manifest.read_text())
    if Path(data['case_path']).resolve()!=case.resolve():
        raise DiagnosisError('Projection cache was generated for a different archived case')
    if data.get('T_sha256')!=archive_report['T']['T_sha256']:
        raise DiagnosisError('Projection cache T.dvg differs from the selected model')
    if data.get('O_size_bytes')!=archive_report['O']['O_size_bytes']:
        raise DiagnosisError('Projection cache O.dvg file size differs')
    if data.get('nff')!=archive_report['O']['nff']:
        raise DiagnosisError('Projection cache nff differs')
    if binary is not None and data.get('binary_sha256')!=archive_report['binary_sha256']:
        raise DiagnosisError('Projection cache runtime binary differs')
    if not _prefix_complete(prefix, archive_report['O']['q_indices'],archive_report['O']['nff']):
        raise DiagnosisError('Projection cache is incomplete')


def native_projection(case:Path,output:Path,preflight:dict[str,Any],
                      binary:Path,src:Path,c_source:Path)->Path:
    """Invoke audited source-backed divERGe projection. Never use raw P/C/D directly."""
    if not (src/'diverge_model.h').is_file():
        raise DiagnosisError(f'Invalid divERGe source include directory: {src}')
    if not c_source.is_file():
        raise DiagnosisError(f'Missing audited native C bridge: {c_source}')
    prefix=output/'restored'
    manifest=output/'manifest.json'
    # Reuse only if both manifest and all projected files are present and verified.
    if manifest.is_file():
        try:
            _validated_external_prefix(prefix,case,preflight,binary)
            return prefix
        except (DiagnosisError,ValueError,KeyError,TypeError):
            raise DiagnosisError('Existing projection cache does not match inputs. Use a new --out-dir; no overwrite.')
    if any(output.glob('restored_*.bin')):
        raise DiagnosisError('Unverified projected matrices exist in out-dir; do not overwrite')
    fixture=output/'historical_model.fixture'
    _read_archived_fixture(case/'T.dvg',fixture)
    exe=output/'restore_full_projected'
    # Build/link ONLY against the exact audited libdivERGe.so used in the Stage 2 gates.
    subprocess.run(['gcc','-O2','-std=gnu11','-fopenmp','-DCTYPESGEN=1',
                    '-I'+str(src),str(c_source),'-L'+str(binary.parent),
                    '-Wl,-rpath,'+str(binary.parent),'-ldivERGe','-lm','-o',str(exe)],check=True)
    qmap=preflight['O']['q_indices']
    qcsv=','.join(map(str,qmap.values()))
    with (output/'projection.log').open('w') as log:
        subprocess.run([str(exe),str(fixture),str(case/'O.dvg'),
                        str(preflight['mesh']['tu_distance']),str(prefix),qcsv],
                       check=True,stdout=log,stderr=subprocess.STDOUT)
    if not _prefix_complete(prefix,qmap,preflight['O']['nff']):
        raise DiagnosisError('Native projection finished without all required matrices')
    provenance={
        'case_path':str(case.resolve()),'T_sha256':preflight['T']['T_sha256'],
        'O_size_bytes':preflight['O']['O_size_bytes'],'nff':preflight['O']['nff'],
        'binary_sha256':preflight['binary_sha256'],
        'c_source_sha256':_sha(c_source), 'q_indices':qmap,
        'scope':'source-backed p/d/m projection; Gamma+3M only'
    }
    manifest.write_text(json.dumps(provenance,indent=2)+'\n')
    return prefix


def _sums(vec:np.ndarray,labels:tuple[str,...],geometry:ph.Geometry,
          lattice:np.ndarray,pos:np.ndarray, *, top:int=10)->dict[str,Any]:
    """Coefficient weights only. No claim of an order parameter or physical current."""
    if vec is None:
        return {'status':'no_vector'}
    frac=np.abs(vec)**2
    frac=frac/max(frac.sum(),1e-30)
    by_type={};by_distance={};components=[]
    for idx,(weight,label) in enumerate(zip(frac,labels)):
        text=label.split(':')
        if text[0] in ('0','1','Sx','Sy'):
            name={'0':'rho','1':'Sz','Sx':'Sx','Sy':'Sy'}[text[0]]
            term=text[1]
            ty=f'{name}_{"onsite" if term=="on" else "imag_bond" if term=="I" else "real_bond"}'
            ff=int(text[2]) if term=='on' else int(text[2].split(',')[0])
        else:
            name=text[0];term=text[1]
            ty=f'{name}_{"onsite" if term=="on" else "bond"}'
            ff=int(text[2]) if term=='on' else int(text[2].split(',')[0])
        by_type[ty]=by_type.get(ty,0.)+float(weight)
        disp=pos[geometry.to_orb[ff]]-pos[geometry.from_orb[ff]]-geometry.ff_R[ff]@lattice
        length=float(np.linalg.norm(disp))
        shell='onsite' if length<1e-7 else 'NN' if abs(length-.5)<1e-6 else 'NNN' if abs(length-np.sqrt(3)/2)<1e-6 else 'third_1' if abs(length-1)<1e-6 else f'd_{length:.6g}'
        by_distance[shell]=by_distance.get(shell,0.)+float(weight)
        components.append({'label':label,'weight':float(weight),'bond_length':length,'shell':shell})
    return {'by_source_type':dict(sorted(by_type.items())),
            'by_bond_shell':dict(sorted(by_distance.items())),
            'largest_components':sorted(components,key=lambda c:c['weight'],reverse=True)[:top]}


def _subspace_sums(basis:np.ndarray,labels:tuple[str,...],geometry:ph.Geometry,
                   lattice:np.ndarray,pos:np.ndarray)->dict[str,Any]:
    """Invariant weights for the complete leading eigenspace (not one vector)."""
    diagonal=ph.projector_diagonal(basis)
    ans=_sums(np.sqrt(diagonal),labels,geometry,lattice,pos)
    ans['definition']='diag(orthogonal eigenspace projector) / rank'
    ans['rank']=int(basis.shape[1])
    return ans


def select_ph_q(geometry:ph.Geometry, spec:str)->dict[str,int]:
    """Fail-closed selector for physical self-inverse q only.

    Arbitrary q are NOT supported by Hermitian source bases because
    O(q)^dagger=O(-q). The ordinary-q / complex-source problem is separate.
    """
    standard=geometry.high_symmetry_q()
    if spec.strip().lower() in ('trim','default'):
        return standard.copy()
    if spec.strip().lower() in ('all','full','full-bz','bz'):
        raise DiagnosisError('Full BZ PH analysis is NOT implemented: ordinary q '
            'requires q/-q complex-source pairing; do not apply Hermitian TRIM kernels.')
    names={key.lower():key for key in standard}
    names.update({'gamma':'Gamma','g':'Gamma','ma':'M_a','mb':'M_b','mc':'M_c'})
    selected={}
    tokens=[tok.strip() for tok in spec.split(',') if tok.strip()]
    if not tokens:
        raise DiagnosisError('Empty --ph-q selection')
    for token in tokens:
        if token.lower() in names:
            label=names[token.lower()]
            iq=standard[label]
        else:
            try:iq=int(token)
            except ValueError as exc:
                raise DiagnosisError(f'Invalid PH q token {token!r}') from exc
            if not geometry.is_ph_trim(iq):
                raise DiagnosisError(f'q index {iq} is not self-inverse (Gamma/M). '
                   'Ordinary q requires a q/-q complex-source implementation.')
            label=next(name for name,q in standard.items() if q==iq)
        selected[label]=iq
    if 'Gamma' not in selected:
        raise DiagnosisError('--ph-q must include Gamma in v1 (Gamma Cooper is mandatory)')
    return {k:v for k,v in standard.items() if k in selected}


def calculate(O:Path,prefix:Path,geom:ph.Geometry,lattice:np.ndarray,pos:np.ndarray,
              *,no_soc:bool=False,ph_q:str="trim")->tuple[dict[str,Any],dict[str,np.ndarray]]:
    """Compute selected TRIM PH (incl. transverse) and Gamma Cooper."""
    qmap=select_ph_q(geom,ph_q)
    result={'status':'PENDING','scope':'signed-integrated-kernel selected TRIM PH and Gamma PP',
            'nff':geom.nff,'nq':geom.nq,'q_indices':qmap,
            'q_coverage':{'PH':'selected self-inverse q only','PP':'Gamma only',
                          'selected_count':len(qmap),'total_BZ_points':geom.nq,
                          'is_full_BZ':False,'q_minus_q_paired_generic':False},
            'q':{},'sector_candidates':[],'blockers':[],'warnings':[],
            'normalization':{'PH':'K = T† full_d T; H = K @ L_PH',
                             'PP':'K = 0.5*T† full_p T; H = -K @ L_PP'}}
    modes={}
    for q_label,iq in qmap.items():
        b0=ph.longitudinal_basis(geom,iq)
        bx=ph.transverse_basis(b0,geom.nff)
        d=ph.native_full_matrix(prefix,'d',iq,geom.nff)
        m=ph.native_full_matrix(prefix,'m',iq,geom.nff)
        loop=ph.native_loop(O,'PH',iq,geom)
        item={'q_index':iq,'q_reduced':geom.reduced_q(iq).tolist()}
        for family,basis in [('longitudinal',b0),('transverse',bx)]:
            dm=ph.compare_d_m(d,m,basis)
            K,L=ph.physical_matrices(d,loop,basis,'PH')
            sanity=ph.kernel_sanity(K,L,basis,transverse=(family=='transverse'))
            block={'d_vs_m_relative':dm,'sanity':sanity,'sectors':{}}
            if dm>1e-7:result['blockers'].append(f'{q_label}/{family}/d_vs_m')
            for metric,value in sanity.items():
                if value>1e-7:result['blockers'].append(f'{q_label}/{family}/{metric}={value}')
            for sector,mat in ph.split_ph(K,L,basis).items():
                spec=ph.signed_spectrum(mat,hermitian_source=True)
                key=f'{q_label}_{family}_{sector}'
                multi=spec.leading_multiplicity
                report=spec.summary()
                if not spec.dominant_real_branch:
                    result['blockers'].append(f'{key}/unverified_leading_branch')
                if multi>1:
                    result['warnings'].append(f'{key}: {multi}-fold leading eigenspace; representative vector is non-unique; use projector-invariant weights')
                if spec.leading_vector is not None:
                    modes[key+'_gap']=spec.leading_vector
                    modes[key+'_response']=spec.leading_response
                    modes[key+'_gap_basis']=spec.leading_basis
                    if spec.response_basis is not None:
                        modes[key+'_response_basis']=spec.response_basis
                    else:
                        result['blockers'].append(f'{key}/rank_deficient_induced_response')
                    labels=mat.source_labels
                    report['gap_coefficients']=_subspace_sums(spec.leading_basis,labels,geom,lattice,pos)
                    if spec.response_basis is not None:
                        report['induced_response_coefficients']=_subspace_sums(spec.response_basis,labels,geom,lattice,pos)
                    else:
                        report['induced_response_coefficients']={'status':'rank_deficient'}
                    report['source_labels']=list(labels)
                    result['sector_candidates'].append({'q':q_label,'family':family,
                                          'sector':sector,'leading_real':spec.leading_real,
                                          'degeneracy':multi})
                block['sectors'][sector]=report
            item[family]=block
        item['long_trans_cross_K']=float(np.linalg.norm(b0.transform.conj().T@d@bx.transform)/max(np.linalg.norm(d),1e-30))
        item['long_trans_cross_L']=float(np.linalg.norm(b0.transform.conj().T@loop@bx.transform)/max(np.linalg.norm(loop),1e-30))
        for typ in ('long_trans_cross_K','long_trans_cross_L'):
            if item[typ]>1e-7:result['blockers'].append(f'{q_label}/{typ}')
        if no_soc:
            n=geom.nff
            for nm,A in [('K',ph.physical_matrices(d,loop,b0,'PH')[0]),
                         ('L',ph.physical_matrices(d,loop,b0,'PH')[1])]:
                ratio=float(np.linalg.norm(A[:n,n:])/max(np.linalg.norm(A),1e-30))
                item[f'noSOC_rho_Sz_cross_{nm}']=ratio
                if ratio>1e-7:result['blockers'].append(f'{q_label}/noSOC_rho_Sz_cross_{nm}={ratio}')
            item['SU2_regression']=ph.no_soc_su2_regression(d,loop,geom.nff)
            for nm,val in item['SU2_regression'].items():
                if val>1e-7:result['blockers'].append(f'{q_label}/{nm}={val}')
        if q_label=='Gamma':
            basis=ph.cooper_basis(geom)
            p=ph.native_full_matrix(prefix,'p',iq,geom.nff)
            lpp=ph.native_loop(O,'PP',iq,geom)
            K,L=ph.physical_matrices(p,lpp,basis,'PP')
            sanity=ph.kernel_sanity(K,L,basis)
            item['cooper']={'sanity':sanity,'sectors':{}}
            for metric,value in sanity.items():
                if value>1e-7:result['blockers'].append(f'Gamma/Cooper/{metric}={value}')
            for sector,mat in ph.split_pp(K,L,basis).items():
                spec=ph.signed_spectrum(mat,hermitian_source=False)
                key=f'Gamma_cooper_{sector.replace("+","plus").replace("-","minus")}'
                multi=spec.leading_multiplicity
                report=spec.summary()
                if not spec.dominant_real_branch:
                    result['blockers'].append(f'{key}/unverified_leading_branch')
                if multi>1:
                    result['warnings'].append(f'{key}: {multi}-fold leading eigenspace; representative vector is non-unique; use projector-invariant weights')
                if spec.leading_vector is not None:
                    modes[key+'_gap']=spec.leading_vector
                    modes[key+'_response']=spec.leading_response
                    modes[key+'_gap_basis']=spec.leading_basis
                    if spec.response_basis is not None:
                        modes[key+'_response_basis']=spec.response_basis
                    else:
                        result['blockers'].append(f'{key}/rank_deficient_induced_response')
                    labels=mat.source_labels
                    report['gap_coefficients']=_subspace_sums(spec.leading_basis,labels,geom,lattice,pos)
                    if spec.response_basis is not None:
                        report['induced_response_coefficients']=_subspace_sums(spec.response_basis,labels,geom,lattice,pos)
                    else:
                        report['induced_response_coefficients']={'status':'rank_deficient'}
                    report['source_labels']=list(labels)
                    result['sector_candidates'].append({'q':'Gamma','family':'cooper',
                                      'sector':sector,'leading_real':spec.leading_real,'degeneracy':multi})
                item['cooper']['sectors'][sector]=report
        result['q'][q_label]=item
    if all(m in qmap for m in ('M_a','M_b','M_c')):
        for fam in ('longitudinal','transverse'):
            for sec in ('TR_even','TR_odd'):
                x=[result['q'][m][fam]['sectors'][sec]['leading_real'] for m in ('M_a','M_b','M_c')]
                if any(v is None for v in x) or not np.allclose(x,x[0],rtol=1e-7,atol=1e-7):
                    result['blockers'].append(f'broken_C3_M_equivalence_{fam}_{sec}')
    result['sector_candidates'].sort(key=lambda entry:entry['leading_real'],reverse=True)
    result['status']='PASSED' if not result['blockers'] else 'BLOCKED'
    result['interpretation_limits']=[
       'Only selected self-inverse Gamma/M PH wavevectors; generic/incommensurate q are NOT assessed.',
       'At ordinary q, a valid Hermitian treatment requires q/-q pairing, not the Gamma/M source basis.',
       'Signed integrated-loop eigenvalues are diagnostics, not thermodynamic susceptibility divergence proofs.',
       'SOC mixes longitudinal rho and Sz; bond imaginary charge source alone is NOT physical electric current.',
       'Degenerate right-eigenspaces are saved in full; projector weights are basis invariant, but irrep labels still require symmetry analysis.',
       'Gap/source coefficients and loop-induced response coefficients must be interpreted separately.',
       'No FM, CBO, LCO, or A1/E2 phase label is assigned automatically from one coefficient threshold.'
    ]
    return result,modes


def _lattice_and_positions(T:Path)->tuple[np.ndarray,np.ndarray]:
    h=np.fromfile(T,dtype='<i8',count=128)
    lattice=np.frombuffer(h[26:35].tobytes(),dtype='<f8').reshape(3,3).copy()
    pos=np.fromfile(T,dtype='<f8',count=9,offset=int(h[35])).reshape(3,3)
    return lattice,pos


def main(argv:list[str]|None=None)->int:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case',type=Path,required=True)
    parser.add_argument('--out-dir',type=Path,required=True)
    parser.add_argument('--projected-prefix',type=Path,default=None,
                        help='Use a provenance-checked existing projected native cache')
    parser.add_argument('--diverge-src',type=Path,default=None,
                        help='Path to source directory with diverge_model.h (needed for fresh projection)')
    parser.add_argument('--binary',type=Path,required=True,
                        help='Audited libdivERGe.so; required also to verify cache provenance')
    parser.add_argument('--c-bridge',type=Path,default=Path(__file__).with_name('TUfRG_restore.c'))
    parser.add_argument('--require-soc',action='store_true')
    parser.add_argument('--ph-q',default='trim',
       help='Selected PH self-inverse q: trim (default), Gamma,M_a,M_b,M_c, or numeric TRIM indices. Full BZ unsupported.')
    args=parser.parse_args(argv)
    try:
        case=args.case.resolve(strict=True)
        out=args.out_dir.resolve()
        if case==out or case in out.parents:
            raise DiagnosisError('--out-dir must be outside the original archived case directory')
        binary=args.binary.resolve(strict=True) if args.binary is not None else None
        if args.projected_prefix is None and args.diverge_src is None:
            raise DiagnosisError('Fresh source-backed projection requires --diverge-src')
        preflight=archive.inspect_case(case,binary,args.require_soc)
        out.mkdir(parents=True,exist_ok=True)
        if args.projected_prefix is not None:
            prefix=args.projected_prefix.resolve()
            _validated_external_prefix(prefix,case,preflight,binary)
        else:
            prefix=native_projection(case,out,preflight,binary,args.diverge_src.resolve(strict=True),args.c_bridge.resolve(strict=True))
        geometry=ph.Geometry.from_O(case/'O.dvg')
        lattice,pos=_lattice_and_positions(case/'T.dvg')
        params=preflight['parameters']
        no_soc=abs(float(params.get('lambda_nn',0)))+abs(float(params.get('lambda_nnn',0)))==0
        result,modes=calculate(case/'O.dvg',prefix,geometry,lattice,pos,no_soc=no_soc,ph_q=args.ph_q)
        result['case_path']=str(case)
        result['preflight_status']=preflight['status']
        result['binary_sha256']=preflight['binary_sha256']
        result['T_sha256']=preflight['T']['T_sha256']
        result['parameters']=params
        result['resolved_parameters']=preflight.get('resolved_params')
        result['projected_prefix']=str(prefix)
        tmp=out/'diagnosis.json.tmp'
        tmp.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
        tmp.replace(out/'diagnosis.json')
        np.savez_compressed(out/'leading_modes.npz',**modes)
        print('STAGE3_DIAGNOSIS_'+result['status'], 'blockers=', result['blockers'],
              'report=',out/'diagnosis.json',flush=True)
        return 0 if result['status']=='PASSED' else 3
    except (DiagnosisError, ph.PhysicsInputError,
            OSError,ValueError,KeyError,subprocess.CalledProcessError) as exc:
        parser.exit(2,f'STAGE3_DIAGNOSIS_FAILED: {exc}\n')


if __name__=='__main__':
    raise SystemExit(main())
