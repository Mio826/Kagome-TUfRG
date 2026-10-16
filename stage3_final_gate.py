#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Independent Stage-3 v1.0 verification of a finished diagnosis report.

Input is SMALL diagnosis.json and leading_modes.npz (not 15-GB O.dvg).
Checks are independent of TUfRG_diagnose and TUfRG_physics imports.
Optional --soc141-reference checks the one Stage-2 audited SOC benchmark ONLY.
"""
import argparse
import json
from pathlib import Path
import numpy as np


def source_type(label):
    parts=label.split(':')
    component=parts[0]
    if component in ('0','1','Sx','Sy'):
        name={'0':'rho','1':'Sz','Sx':'Sx','Sy':'Sy'}[component]
        kind={'on':'onsite','I':'imag_bond','R':'real_bond'}[parts[1]]
        return f'{name}_{kind}'
    return f'{component}_{"onsite" if parts[1]=="on" else "bond"}'


def check(path:Path, *, soc141_reference=False):
    data=json.loads((path/'diagnosis.json').read_text())
    assert data['status']=='PASSED' and not data['blockers'],data['blockers']
    assert data['q_coverage']['PH']=='selected self-inverse q only'
    assert data['q_coverage']['PP']=='Gamma only'
    assert data['q_coverage']['is_full_BZ'] is False
    assert data['q_coverage']['q_minus_q_paired_generic'] is False
    assert data['q_coverage']['selected_count']==len(data['q_indices'])
    n_sectors=0; n_degenerate=0; maxweight_error=0.;maxortho=0.
    with np.load(path/'leading_modes.npz',allow_pickle=False) as stored:
        for q,datum in data['q'].items():
            for family in ('longitudinal','transverse','cooper'):
                if family not in datum:continue
                for sector,v in datum[family]['sectors'].items():
                    key=(f'{q}_{family}_{sector}' if family!='cooper' else
                         f'{q}_cooper_{sector.replace("+","plus").replace("-","minus")}')
                    n_sectors+=1
                    m=v['leading_multiplicity'];dim=v['dimension']
                    assert m>0 and v['dominant_real_branch'],key
                    n_degenerate+=(m>1)
                    labels=v['source_labels']
                    assert len(labels)==dim,key
                    for part,field in (('gap','gap_coefficients'),
                                       ('response','induced_response_coefficients')):
                        B=stored[key+'_'+part+'_basis']
                        assert B.shape==(dim,m),(key,part,B.shape,dim,m)
                        ortho=np.linalg.norm(B.conj().T@B-np.eye(m))
                        maxortho=max(maxortho,float(ortho))
                        assert ortho<1e-8,(key,part,ortho)
                        weights=np.sum(np.abs(B)**2,axis=1)/m
                        assert np.isclose(weights.sum(),1.,atol=1e-9),(key,part)
                        typ={}
                        for item,wt in zip(labels,weights):
                            type_key=source_type(item)
                            typ[type_key]=typ.get(type_key,0.)+float(wt)
                        recorded=v[field]
                        assert recorded['rank']==m,(key,part)
                        assert recorded['definition']=='diag(orthogonal eigenspace projector) / rank'
                        assert set(typ)==set(recorded['by_source_type']),(key,part)
                        err=max(abs(x-recorded['by_source_type'][k]) for k,x in typ.items())
                        maxweight_error=max(maxweight_error,err)
                        assert err<1e-9,(key,part,err)
                        # Rotate this basis by a deterministic complex/real unitary:
                        # The projector must not change (gauge-invariant summary).
                        if m>1:
                            rng=np.random.default_rng(2718+m)
                            rand=rng.normal(size=(m,m))
                            if np.iscomplexobj(B) and family=='cooper':
                                rand=rand+1j*rng.normal(size=(m,m))
                            rot,_=np.linalg.qr(rand)
                            rotated=B@rot
                            invariant=np.sum(np.abs(rotated)**2,axis=1)/m
                            assert np.max(np.abs(weights-invariant))<1e-12,key
                    gap=stored[key+'_gap']
                    response=stored[key+'_response']
                    assert gap.shape==(dim,) and response.shape==(dim,)
                    assert np.linalg.norm(gap)>0 and np.linalg.norm(response)>0
                    Q=stored[key+'_gap_basis']
                    relative=np.linalg.norm(gap-Q@(Q.conj().T@gap))/np.linalg.norm(gap)
                    assert relative<1e-8,(key,'representative gap not in span',relative)
                    Qr=stored[key+'_response_basis']
                    relative=np.linalg.norm(response-Qr@(Qr.conj().T@response))/np.linalg.norm(response)
                    assert relative<1e-8,(key,'representative response not in span',relative)
                    assert v['subspace_eigenpair_relative_residual']<1e-8,key
        assert len(stored.files)==4*n_sectors,(len(stored.files),n_sectors)
    if soc141_reference:
        assert data['nff']==141 and data['nq']==576
        assert len(data['q'])==4 and n_sectors==19
        assert n_degenerate==10,(n_degenerate,'unexpected multiplicities')
        references=[
          ('Gamma','longitudinal','TR_even',5.135272791561742),
          ('M_a','longitudinal','TR_even',5.038260359227392),
          ('M_b','longitudinal','TR_even',5.038260359227376),
          ('M_c','longitudinal','TR_even',5.038260359227396),
          ('M_a','longitudinal','TR_odd',1.1686931087435855),
          ('M_a','transverse','TR_odd',1.1666983943286884),
          ('Gamma','cooper','Sz0',1.9940449614324522),
          ('Gamma','cooper','+1',0.31654220874111155),
          ('Gamma','cooper','-1',0.316542208741111),
        ]
        for q,family,sector,expected in references:
            got=data['q'][q][family]['sectors'][sector]['leading_real']
            assert abs(got-expected)<1e-7,(q,family,sector,got,expected)
    return {'status':'STAGE3_FINAL_GATE_PASSED','case':data.get('case_path'),
            'n_sectors':n_sectors,'n_degenerate_sectors':n_degenerate,
            'saved_eigenmode_arrays':4*n_sectors,
            'max_orthonormality_error':maxortho,
            'max_projector_type_weight_error':maxweight_error,
            'momentum_coverage':data['q_coverage'],
            'soc141_reference_checked':soc141_reference}


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out-dir',type=Path,required=True)
    ap.add_argument('--soc141-reference',action='store_true')
    args=ap.parse_args()
    try:
        result=check(args.out_dir,soc141_reference=args.soc141_reference)
    except (AssertionError,ValueError,KeyError,OSError) as ex:
        ap.exit(2,f'STAGE3_FINAL_GATE_FAILED: {ex}\n')
    (args.out_dir/'stage3_final_gate.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
