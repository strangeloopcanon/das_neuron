"""Post-hoc discovery confirmation using fixed models and a fresh holdout.

--freeze writes selected VALIDATION features before --run generates any new data.
No training or changes to the primary protocol, models, or six-feature result.
"""
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np

ROOT=Path(__file__).resolve().parent
PREFIX='learned_superposition_v2'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def model_sha(m):
    value=dict(seed=m['seed'],W=m['W'],bias=m['bias'])
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def freeze():
    target=ROOT/(PREFIX+'_confirmation_protocol.json')
    if target.exists():raise FileExistsError('Confirmation protocol already frozen; do not replace it.')
    r=json.loads((ROOT/(PREFIX+'_results.json')).read_text());p=json.loads((ROOT/(PREFIX+'_protocol.json')).read_text())
    g=p['compressed_gates'];selected=[]
    for m in r['models']:
        if m['stage']!='compressed':continue
        keep=[t['feature'] for t in m['validation']['per_feature'] if t['active_relative_rmse']<=g['active_relative_rmse_max'] and t['combined_inactive_relative_rmse']<=g['combined_inactive_relative_rmse_max'] and t['active_amplitude_r2']>g['active_amplitude_r2_min_exclusive']]
        selected.append(dict(seed=m['seed'],model_sha256=model_sha(m),features_selected_on_validation=keep))
    protocol=dict(stage='Post-hoc discovery confirmation; not the original six-feature primary test',
        original_results_sha256=sha(ROOT/(PREFIX+'_results.json')),original_protocol_sha256=sha(ROOT/(PREFIX+'_protocol.json')),source_sha256=sha(__file__),
        model_selection='All five compressed seeds retained. For each frozen model, select features solely by the original compressed accuracy/leakage/active-amplitude-R2 criteria on VALIDATION.',
        selection=selected,new_holdout=dict(seed=978311,per_feature=4096,amplitude=[.8,1.2],feature_groups=6),
        criteria=dict(required_retained_features_per_model=5,required_model_count=5,active_relative_rmse_max=.25,combined_inactive_relative_rmse_max=.25,active_amplitude_r2_min_exclusive=0),
        scope='Confirm at least five specified response features per model are selectively recoverable from two latent coordinates. All six features evaluated; original all-six primary gate remains failed. No fitting or hyperparameter changes.',
        no_retraining=True)
    target.write_text(json.dumps(protocol,indent=2)+'\n')
    print(json.dumps(dict(frozen_protocol=str(target.name),selection=selected,new_data_generated=False)))


def run():
    p=json.loads((ROOT/(PREFIX+'_confirmation_protocol.json')).read_text());r=json.loads((ROOT/(PREFIX+'_results.json')).read_text())
    assert p['original_results_sha256']==sha(ROOT/(PREFIX+'_results.json'))
    assert p['source_sha256']==sha(__file__)
    spec=p['new_holdout'];rng=np.random.default_rng(spec['seed']);groups=[]
    for i in range(6):
        x=np.zeros((spec['per_feature'],6));x[:,i]=rng.uniform(*spec['amplitude'],len(x));groups.append(x)
    x=np.concatenate(groups)
    c0=1-.36*.96;c=c0*(1+x);x=(c-c0)/c0
    # Fresh nonzero rows are disjoint from every original partition, including TEST.
    with np.load(ROOT/(PREFIX+'_data.npz'),allow_pickle=False) as old:
        old_keys={tuple(row) for name in old.files for row in old[name] if np.any(row)}
    assert not {tuple(row) for row in x}&old_keys
    np.savez_compressed(ROOT/(PREFIX+'_confirmation_data.npz'),CONFIRMATION=x)
    result=dict(protocol_sha256=sha(ROOT/(PREFIX+'_confirmation_protocol.json')),archive_sha256=sha(ROOT/(PREFIX+'_confirmation_data.npz')),array_sha256=hashlib.sha256(np.asarray(x,dtype='<f8').tobytes()).hexdigest(),rows=len(x),models=[],primary_all_six_result_unchanged=r['summary'])
    for selected in p['selection']:
        m=next(m for m in r['models'] if m['stage']=='compressed' and m['seed']==selected['seed']);assert model_sha(m)==selected['model_sha256']
        w,b=np.array(m['W']),np.array(m['bias']);pred=np.maximum(0,(x@w.T)@w+b);pf=[]
        for i in range(6):
            rows=x[:,i]>0;target=x[rows,i];y=pred[rows];den=float(np.sum(target*target));error=y[:,i]-target
            record=dict(feature=i,selected_on_validation=i in selected['features_selected_on_validation'],active_relative_rmse=float(np.sqrt(np.sum(error*error)/den)),combined_inactive_relative_rmse=float(np.sqrt(np.sum(y[:,np.arange(6)!=i]**2)/den)),active_amplitude_r2=float(1-np.sum(error*error)/np.sum((target-target.mean())**2)))
            record['passes']=record['active_relative_rmse']<=.25 and record['combined_inactive_relative_rmse']<=.25 and record['active_amplitude_r2']>0
            pf.append(record)
        selected_pf=[t for t in pf if t['selected_on_validation']]
        passed=len(selected_pf)>=5 and all(t['passes'] for t in selected_pf)
        result['models'].append(dict(seed=m['seed'],model_sha256=model_sha(m),per_feature=pf,selected_count=len(selected_pf),confirmed_count=sum(t['passes'] for t in selected_pf),confirmation_pass=passed))
    result['confirmed_model_count']=sum(m['confirmation_pass'] for m in result['models']);result['confirmation_pass']=len(result['models'])==5 and result['confirmed_model_count']==5
    (ROOT/(PREFIX+'_confirmation_results.json')).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(confirmation_pass=result['confirmation_pass'],confirmed_models=result['confirmed_model_count'],rows=len(x),primary_study_pass=r['summary']['study_pass'])))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--freeze',action='store_true');ap.add_argument('--run',action='store_true');a=ap.parse_args()
    if a.freeze:freeze()
    elif a.run:run()
    else:ap.error('select --freeze or --run')
