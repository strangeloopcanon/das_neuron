"""Frozen-protocol task generation and automatic baseline evaluation. No training."""
import hashlib,json
from pathlib import Path
import numpy as np
from dsge_prediction_materials import forward
ROOT=Path(__file__).resolve().parent

def canonical(x): return json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def sha(x): return hashlib.sha256(x).hexdigest()
def read(name): return json.loads((ROOT/name).read_text())
def write(name,x): (ROOT/name).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def direction(x): return 0 if abs(x)<=.002 else (1 if x>0 else -1)
def truth(model,t): return forward(model,t['raw_input'],t['intervention'])['normalized_output'][t['output_index']]-t['baseline']['normalized_output'][t['output_index']]
def jacobian(model,t):
    w=model['network']; b=t['baseline']; e=t['intervention']; o=t['output_index']; u=e['unit']; layer=e['layer']
    if layer=='H2': return -e['fraction']*b['h2'][u]*w['W3'][u][o]
    dh1=[0.]*16; dp2=[0.]*16
    if layer=='H1': dh1[u]=-e['fraction']*b['h1'][u]
    elif layer=='W1': dh1[u]=e['delta']*t['normalized_input'][e['input_index']]*(b['p1'][u]>0)
    else: dp2[u]=e['delta']*b['h1'][e['input_index']]
    return sum((dp2[j]+sum(w['W2'][j][i]*dh1[i] for i in range(16)))*(b['p2'][j]>0)*w['W3'][j][o] for j in range(16))
def metrics(p,y):
    err=np.abs(np.array(p)-y)
    return dict(n=len(y),mae_sd=float(err.mean()),numeric_accuracy=float((err<=.02).mean()),direction_accuracy=float(np.mean([direction(a)==direction(b) for a,b in zip(p,y)])))

def run():
    protocol=read('dsge_prediction_protocol.json'); r=read('richer_results.json'); s=read('system_analysis.json')
    saved=next(m for m in r['models'] if m['width']==16 and m['seed']==113)
    labels=next(m for m in s['models'] if m['id']=='H16-seed113')['labels']
    model=dict(id='H16-seed113',parameter_count=387,network=saved['network'],normalization=r['provenance']['normalization'],inputs=s['inputs'],outputs=s['outputs'])
    assert sum(np.asarray(v).size for v in model['network'].values())==387
    from richer_teacher import load_teacher
    teacher=load_teacher(); seen=set()
    for spec in r['protocol']['data'].values():
        if isinstance(spec,dict):
            k,st=teacher.sample_states(spec['n'],spec['seed']); seen.update((float(a).hex(),int(b)) for a,b in zip(k,st))
    source_files=['richer_results.json','system_analysis.json','richer_teacher.npz','dsge_prediction_experiment.py','dsge_prediction_materials.py']
    meta=dict(protocol_sha256=sha((ROOT/'dsge_prediction_protocol.json').read_bytes()),model_sha256=sha(canonical(model)),labels_sha256=sha(canonical(labels)),source_sha256={f:sha((ROOT/f).read_bytes()) for f in source_files})
    banks={}
    for bank,count in protocol['banks'].items():
        rng=np.random.default_rng(protocol['seeds'][bank]); rows=[]
        schedule=[('h1_attenuation',0),('h2_attenuation',1)] if bank=='practice' else [(f,o) for _ in range(count//12) for f in protocol['families'] for o in range(3)]
        for idx,(family,o) in enumerate(schedule):
            while True:
                k=float(rng.uniform(.5,1.8)*teacher.parameters['kss']); st=int(rng.choice(21,p=teacher.stationary)); key=(k.hex(),st)
                if key not in seen: break
            seen.add(key); raw=[k,*teacher.state_features[st].tolist()]; b=forward(model,raw)
            layer='H2' if family=='h2_attenuation' else 'H1'
            if family=='incoming_weight': layer=str(rng.choice(['W1','W2']))
            active=b['h2'] if layer in ('H2','W2') else b['h1']
            u=int(rng.choice(np.flatnonzero(np.array(active)>0)))
            if family=='incoming_weight':
                upstream=int(rng.integers(3)) if layer=='W1' else int(rng.choice(np.flatnonzero(np.array(b['h1'])>0)))
                e=dict(layer=layer,unit=u,input_index=upstream,delta=float(rng.choice([-.5,.5])))
            else: e=dict(layer=layer,unit=u,fraction=0. if family=='sham' else (1. if bank=='practice' else float(rng.choice([.25,.5,1.]))))
            t=dict(task_id=f'{bank}-{idx+1:04d}',bank=bank,family=family,output_index=o,output_name=model['outputs'][o],raw_input=raw,normalized_input=b.pop('normalized_input'),state=st,baseline=b,intervention=e)
            if bank=='practice': t.update(answer_sd=truth(model,t),edited_trace=forward(model,raw,e))
            rows.append(t)
        banks[bank]=rows
    rng=np.random.default_rng(protocol['seeds']['allocation']); cells=[dict(condition=c,form=f) for c in protocol['conditions'] for f in ['A','B']]; allocations=[]
    for block in range(6):
        for j in rng.permutation(4): allocations.append(dict(slot=len(allocations),**cells[int(j)]))
    tasks=dict(schema_version=1,metadata=meta,model=model,labels=labels,label_scope=s['label_scope'],teacher_state_sampling=dict(kss=teacher.parameters['kss'],state_features=teacher.state_features.tolist(),stationary=teacher.stationary.tolist(),P=teacher.P.tolist()),allocations=allocations,banks=banks)
    write('dsge_prediction_tasks.json',tasks)
    results={}
    for bank,rows in banks.items():
        strata={}
        for family in ['all',*protocol['families']]:
            subset=[t for t in rows if family=='all' or t['family']==family]
            if not subset: continue
            y=[truth(model,t) for t in subset]
            strata[family]={name:metrics(p,y) for name,p in [('zero_change',[0.]*len(y)),('frozen_gate_jacobian',[jacobian(model,t) for t in subset]),('exact_edited_forward',y)]}
            h2=[t for t in subset if t['family']=='h2_attenuation']
            if h2: strata[family]['h2_direct_readout']=metrics([jacobian(model,t) for t in h2],[truth(model,t) for t in h2])
        results[bank]=strata
    write('dsge_prediction_baselines.json',dict(metadata={**meta,'tasks_sha256':sha((ROOT/'dsge_prediction_tasks.json').read_bytes())},human_completed=0,automated_baselines=results,scope=protocol['scope']))
    print(json.dumps(results['automated'],indent=2))
if __name__=='__main__': run()
