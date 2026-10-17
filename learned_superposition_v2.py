"""Fixed-protocol full-width learning, projection and learned sparse compression.

No existing experiment is edited. Only --run trains; --pilot does not read TEST.
"""
import os
for _key in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[_key]='1'
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import time
import numpy as np

ROOT=Path(__file__).resolve().parent
PREFIX='learned_superposition_v2'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def array_sha(x):
    return hashlib.sha256(np.asarray(x,dtype='<f8').tobytes()).hexdigest()


def protocol():
    return json.loads((ROOT/(PREFIX+'_protocol.json')).read_text())


def teacher(z):
    p=protocol()['teacher'];y=np.asarray(z,float)*p['capital']**p['alpha']
    return (1-p['alpha']*p['beta'])*y,p['alpha']*p['beta']*y


def dataset(spec,amplitude,kind='singleton'):
    rng=np.random.default_rng(spec['seed'])
    if kind=='singleton':
        groups=[]
        for i in range(6):
            x=np.zeros((spec['per_feature'],6));x[:,i]=rng.uniform(*amplitude,len(x));groups.append(x)
        groups.append(np.zeros((spec['zero'],6)))
        x=np.concatenate(groups);rng.shuffle(x)
    elif kind=='pair':
        groups=[]
        for i,j in itertools.combinations(range(6),2):
            x=np.zeros((spec['per_pair'],6));x[:,[i,j]]=rng.uniform(*amplitude,(len(x),2));groups.append(x)
        x=np.concatenate(groups);rng.shuffle(x)
    elif kind=='sparse':
        x=(rng.random((spec['n'],6))<spec['activity_probability'])*rng.uniform(*amplitude,(spec['n'],6))
    elif kind=='edit':
        x=np.zeros((6*spec['per_feature'],6))
        for i in range(6):
            lo=i*spec['per_feature'];hi=(i+1)*spec['per_feature']
            x[lo:hi,i]=rng.uniform(*amplitude,spec['per_feature'])
    else:
        raise ValueError(kind)
    # The neural inputs are numerical responses of the closed-form economic teacher.
    c,_=teacher(1+x);c0,_=teacher(np.ones_like(x))
    return (c-c0)/c0


def make_data():
    p=protocol()['data'];out={}
    for name,kind in [('TRAIN','singleton'),('VALIDATION','singleton'),('TEST','singleton'),('PAIR_TEST','pair'),('SPARSE_TEST','sparse'),('EDIT_TEST','edit')]:
        out[name]=dataset(p[name],p['amplitude'],kind)
    # Repeated all-zero baselines are deliberate. Positive examples are disjoint.
    seen=set()
    for name,x in out.items():
        keys={tuple(row) for row in x if np.any(row)}
        assert not keys&seen,name+' duplicate positive rows'
        seen|=keys
    return out


def forward(x,w,b):
    h=x@w.T;pre=h@w+b
    return np.maximum(pre,0),h,pre


def loss_gradient(x,w,b):
    out,h,pre=forward(x,w,b)
    g=2*(out-x)*(pre>0)/x.size
    return float(np.mean((out-x)**2)),h.T@g+(g@w.T).T@x,g.sum(0)


def gradient_check():
    rng=np.random.default_rng(7181);x=rng.uniform(.3,1.4,(13,6));w=rng.normal(0,.4,(3,6));b=rng.normal(.1,.1,6)
    assert np.min(abs(forward(x,w,b)[2]))>1e-5,'gradient check point too close to a ReLU kink'
    _,gw,gb=loss_gradient(x,w,b);errors=[]
    for param,g in [(w,gw),(b,gb)]:
        for ix in np.ndindex(param.shape):
            old=param[ix];param[ix]=old+1e-6;plus=loss_gradient(x,w,b)[0]
            param[ix]=old-1e-6;minus=loss_gradient(x,w,b)[0];param[ix]=old
            errors.append(abs((plus-minus)/2e-6-g[ix]))
    assert max(errors)<1e-7,max(errors)
    return float(max(errors))


def metrics(x,w,b):
    pred=forward(x,w,b)[0];den=float(np.sum(x*x));out=dict(mse=float(np.mean((pred-x)**2)),relative_rmse=float(np.sqrt(np.sum((pred-x)**2)/den)),per_feature=[])
    for i in range(6):
        rows=(x[:,i]>0)&(np.count_nonzero(x,axis=1)==1);target=x[rows,i];y=pred[rows];other=np.arange(6)!=i
        norm=float(np.sum(target*target));ss=float(np.sum((target-target.mean())**2))
        out['per_feature'].append(dict(feature=i,active_relative_rmse=float(np.sqrt(np.sum((y[:,i]-target)**2)/norm)),combined_inactive_relative_rmse=float(np.sqrt(np.sum(y[:,other]**2)/norm)),active_amplitude_r2=float(1-np.sum((y[:,i]-target)**2)/ss)))
    zero=~np.any(x,axis=1)
    out['zero_output_max']=float(np.max(abs(pred[zero]))) if zero.any() else float(np.max(abs(forward(np.zeros((1,6)),w,b)[0])))
    return out


def full_gate(m):
    g=protocol()['full_width_gates']
    return all(q['active_relative_rmse']<=g['active_relative_rmse_max'] and q['combined_inactive_relative_rmse']<=g['combined_inactive_relative_rmse_max'] and q['active_amplitude_r2']>=g['active_amplitude_r2_min'] for q in m['per_feature']) and m['zero_output_max']<=g['zero_output_max']


def train(x,val,w,b,steps,seed):
    p=protocol()['training'];rng=np.random.default_rng(seed);params=[w.copy(),b.copy()];moment=[np.zeros_like(q) for q in params];variance=[q.copy() for q in moment];curve=[];start=time.perf_counter()
    for step in range(1,steps+1):
        batch=x[rng.integers(len(x),size=p['batch_size'])];loss,gw,gb=loss_gradient(batch,*params)
        lr=p['learning_rate'] if step<=steps*p['decay_fraction'] else p['final_learning_rate']
        for q,m,v,grad in zip(params,moment,variance,[gw,gb]):
            m*=.9;m+=.1*grad;v*=.999;v+=.001*grad*grad
            q-=lr*(m/(1-.9**step))/(np.sqrt(v/(1-.999**step))+1e-8)
        if step==1 or step%500==0 or step==steps:
            curve.append(dict(step=step,batch_mse=loss,validation_mse=float(np.mean((forward(val,*params)[0]-val)**2))))
    return dict(W=params[0].tolist(),bias=params[1].tolist(),curve=curve,seconds=time.perf_counter()-start)


def pilot():
    p=protocol();error=gradient_check();spec=p['data'];x=dataset(spec['TRAIN'],spec['amplitude']);val=dataset(spec['VALIDATION'],spec['amplitude'])
    rng=np.random.default_rng(p['pilot']['seed']);w=rng.normal(0,.4,(6,6));b=np.zeros(6)
    row=train(x,val,w,b,p['pilot']['steps'],p['pilot']['seed']+1)
    projected_seconds=row['seconds']/p['pilot']['steps']*len(p['training']['seeds'])*(p['training']['full_steps']+p['training']['compressed_steps'])
    result=dict(protocol_sha256=sha(ROOT/(PREFIX+'_protocol.json')),source_sha256=sha(__file__),gradient_max_absolute_error=error,pilot_steps=p['pilot']['steps'],pilot_seconds=row['seconds'],projected_total_seconds=projected_seconds,budget_seconds=p['pilot']['runtime_budget_seconds'],test_evaluated=False,weights_discarded=True)
    (ROOT/(PREFIX+'_pilot.json')).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))
    return result


def geometry(w):
    norms=np.linalg.norm(w,axis=0);denom=np.outer(norms,norms);cos=np.divide(w.T@w,denom,out=np.zeros((6,6)),where=denom>0);mask=~np.eye(6,dtype=bool)
    return dict(rank=int(np.linalg.matrix_rank(w)),column_norms=norms.tolist(),cosines=cos.tolist(),maximum_positive_pair_cosine=float(np.max(cos[mask])))


def basal_metrics(x,pred):
    return dict(mse=float(np.mean((pred-x)**2)),relative_rmse=float(np.linalg.norm(pred-x)/np.linalg.norm(x)))


def baselines(train_x,test):
    mean=train_x.mean(0);_,_,vt=np.linalg.svd(train_x-mean,full_matrices=False);basis=vt[:2];coordinates=np.argsort(np.sum(train_x*train_x,axis=0),kind='stable')[-2:]
    coord=np.zeros_like(test);coord[:,coordinates]=test[:,coordinates]
    return dict(mean=mean.tolist(),basis=basis.tolist(),coordinates=coordinates.tolist(),pca=basal_metrics(test,np.maximum(0,(test-mean)@basis.T@basis+mean)),coordinates_reference=basal_metrics(test,coord))


def edit_metrics(edit_x,w,b):
    zero=forward(np.zeros((1,6)),w,b)[0][0];per=[];contexts=[]
    # The edit input is never used for training, validation or projection choice.
    for i in range(6):
        amplitudes=edit_x[edit_x[:,i]>0,i];latent=amplitudes[:,None]*w[:,i]
        decoded=np.maximum(0,latent@w+b);target=np.zeros((len(amplitudes),6));target[:,i]=amplitudes
        per.append(dict(feature=i,target_relative_rmse=float(np.linalg.norm(decoded-target)/np.linalg.norm(target)),active_relative_rmse=float(np.linalg.norm(decoded[:,i]-amplitudes)/np.linalg.norm(amplitudes)),inactive_relative_rmse=float(np.linalg.norm(decoded[:,np.arange(6)!=i])/np.linalg.norm(amplitudes)),removed_output_max=float(np.max(abs(np.maximum(0,(latent-latent)@w+b)-zero)))))
        for j in range(6):
            if j==i:continue
            before=np.zeros_like(target);before[:,j]=1.;after=before+target;h=before@w.T;edited=h+latent;out=np.maximum(0,edited@w+b)
            direct=forward(after,w,b)[0];base=forward(before,w,b)[0];removed=np.maximum(0,(edited-latent)@w+b)
            contexts.append(dict(inserted_feature=i,background_feature=j,target_relative_rmse=float(np.linalg.norm(out-after)/np.linalg.norm(after)),unaffected_change_relative_rms=float(np.linalg.norm((out-base)[:,np.arange(6)!=i])/np.linalg.norm(amplitudes)),encoding_identity_maxabs=float(np.max(abs(out-direct))),removal_identity_maxabs=float(np.max(abs(removed-base)))))
    single=edit_x
    original=forward(single,w,b)[0];coordinate=[]
    for j in range(len(w)):
        h=single@w.T;h[:,j]=0;edited=np.maximum(0,h@w+b)
        coordinate.append(dict(coordinate=j,per_feature_response_change=[float(np.linalg.norm((edited-original)[single[:,i]>0])/np.linalg.norm(single[single[:,i]>0,i])) for i in range(6)]))
    return dict(isolated=per,contexts=contexts,coordinate_ablations=coordinate)


def run():
    p=protocol();pilot_result=json.loads((ROOT/(PREFIX+'_pilot.json')).read_text())
    assert pilot_result['protocol_sha256']==sha(ROOT/(PREFIX+'_protocol.json'))
    assert pilot_result['source_sha256']==sha(__file__)
    assert pilot_result['projected_total_seconds']<=p['pilot']['runtime_budget_seconds']
    arrays=make_data();np.savez_compressed(ROOT/(PREFIX+'_data.npz'),**arrays)
    train_x,val=arrays['TRAIN'],arrays['VALIDATION'];models=[];start=time.perf_counter()
    result=dict(protocol_sha256=sha(ROOT/(PREFIX+'_protocol.json')),source_sha256=sha(__file__),data_archive_sha256=sha(ROOT/(PREFIX+'_data.npz')),array_hashes={k:array_sha(x) for k,x in arrays.items()},array_shapes={k:list(x.shape) for k,x in arrays.items()},default_seed=p['training']['default_seed'],models=models,pilot=pilot_result,scope=p['scope'])
    for seed in p['training']['seeds']:
        w=np.random.default_rng(seed).normal(0,.4,(6,6));b=np.zeros(6)
        model=train(train_x,val,w,b,p['training']['full_steps'],seed+1000)
        model.update(seed=seed,width=6,parameters=42,stage='full',initial_W=w.tolist(),initial_bias=b.tolist())
        model['validation']=metrics(val,np.array(model['W']),np.array(model['bias']));model['validation_pass']=full_gate(model['validation']);models.append(model)
        print('full',seed,'VALIDATION pass',model['validation_pass'],flush=True)
    result['full_validation_prerequisite']=all(m['validation_pass'] for m in models)
    if result['full_validation_prerequisite']:
        for full in list(models):
            seed=full['seed'];gaussian=np.random.default_rng(seed+10000).normal(size=(6,2));q,r=np.linalg.qr(gaussian,mode='reduced');signs=np.where(np.diag(r)>=0,1.,-1.);q=q*signs;projection=q.T
            w=projection@np.array(full['W']);b=np.zeros(6)
            model=train(train_x,val,w,b,p['training']['compressed_steps'],seed+2000)
            model.update(seed=seed,width=2,parameters=18,stage='compressed',parent_seed=seed,projection_gaussian=gaussian.tolist(),projection=projection.tolist(),initial_W=w.tolist(),initial_bias=b.tolist())
            model['validation']=metrics(val,np.array(model['W']),np.array(model['bias']));models.append(model)
            print('compressed',seed,'training complete',flush=True)
    # Every parameter is frozen before any evaluation below.
    result['training_seconds']=time.perf_counter()-start
    reference=baselines(train_x,arrays['TEST']);result['baselines']=reference
    for model in models:
        w,b=np.array(model['W']),np.array(model['bias']);model['geometry']=geometry(w)
        model['evaluation']={name:metrics(arrays[name],w,b) for name in ['TEST']}
        model['distribution_shifts']={name:basal_metrics(arrays[name],forward(arrays[name],w,b)[0]) for name in ['PAIR_TEST','SPARSE_TEST']}
        model['pair_groups']=[]
        pairs=arrays['PAIR_TEST']
        for i,j in itertools.combinations(range(6),2):
            selected=(pairs[:,i]>0)&(pairs[:,j]>0);x=pairs[selected]
            model['pair_groups'].append(dict(features=[i,j],**basal_metrics(x,forward(x,w,b)[0])))
        model['edits']=edit_metrics(arrays['EDIT_TEST'],w,b)
        model['unit_response_matrix']=forward(np.eye(6),w,b)[0].tolist()
        if model['stage']=='full':model['test_pass']=full_gate(model['evaluation']['TEST'])
        else:
            g=p['compressed_gates'];m=model['evaluation']['TEST'];geo=model['geometry']
            model['initial_test']=metrics(arrays['TEST'],np.array(model['initial_W']),np.array(model['initial_bias']))
            model['gates']=dict(all_features_active=all(t['active_relative_rmse']<=g['active_relative_rmse_max'] for t in m['per_feature']),all_features_selective=all(t['combined_inactive_relative_rmse']<=g['combined_inactive_relative_rmse_max'] for t in m['per_feature']),all_features_amplitude=all(t['active_amplitude_r2']>g['active_amplitude_r2_min_exclusive'] for t in m['per_feature']),beats_pca=m['mse']<=g['mse_ratio_to_train_fit_pca_max']*reference['pca']['mse'],beats_coordinates=m['mse']<=g['mse_ratio_to_train_fit_coordinate_reference_max']*reference['coordinates_reference']['mse'],rank=geo['rank']==2,nonzero=min(geo['column_norms'])>=g['column_norm_min'],distinct_signed_directions=geo['maximum_positive_pair_cosine']<g['positive_pair_cosine_max'])
            model['test_pass']=all(model['gates'].values())
        print(model['stage'],model['seed'],'TEST pass',model['test_pass'],'relative RMSE',model['evaluation']['TEST']['relative_rmse'],flush=True)
    fulls=[m for m in models if m['stage']=='full'];compressed=[m for m in models if m['stage']=='compressed']
    result['summary']=dict(full_validation_pass_count=sum(m['validation_pass'] for m in fulls),full_test_pass_count=sum(m['test_pass'] for m in fulls),compressed_test_pass_count=sum(m['test_pass'] for m in compressed),compressed_test_count=len(compressed),study_pass=len(compressed)==5 and all(m['test_pass'] for m in fulls) and sum(m['test_pass'] for m in compressed)>=4)
    result['total_seconds']=time.perf_counter()-start
    (ROOT/(PREFIX+'_results.json')).write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    report(result)
    print(json.dumps(result['summary']))


def report(result):
    s=result['summary'];lines=['# Learned economic response superposition, version 2','',f"Full-width TEST controls pass: {s['full_test_pass_count']}/5. Compressed TEST models pass: {s['compressed_test_pass_count']}/{s['compressed_test_count']}. Fixed study criterion: {s['study_pass']}.",'','The six-coordinate model has 42 trainable parameters. Its projected two-coordinate model has 18. Encoder and decoder share weights. Compressed models start from a random orthogonal projection of the trained full encoder, with decoder biases reset to zero, then fine-tune on TRAIN. No hexagon or sector directions are built in.','','## Frozen protocol and data','','Six known consumption-response features come from separate log-utility growth economies. The features reconstruct economic solution outputs, not raw states or policy functions. TRAIN/VALIDATION/TEST contain balanced singleton responses with amplitudes .8–1.2 and repeated zero baselines. All positive rows are disjoint. Pair and sparse tests are distribution shifts. No TEST results affect checkpoint, projection, optimizer, support or gates.','','The compressed 25% active-error and leakage targets are declared operational criteria. Each feature must also have positive amplitude R² on its active examples. Constant feature classification is insufficient. Full-width gates are 2% active error/leakage, amplitude R²≥.95 and zero-output bound. All five VALIDATION controls must pass before compression. All outcomes are retained.','','## All fixed seeds','','| Stage | Seed | Parameters | TEST response RMSE | Largest active error | Largest leakage | Lowest active amplitude R² | TEST gate | Pair RMSE |','|---|---:|---:|---:|---:|---:|---:|---|---:|']
    for m in result['models']:
        pf=m['evaluation']['TEST']['per_feature']
        lines.append(f"| {m['stage']} | {m['seed']} | {m['parameters']} | {m['evaluation']['TEST']['relative_rmse']:.4f} | {max(t['active_relative_rmse'] for t in pf):.4f} | {max(t['combined_inactive_relative_rmse'] for t in pf):.4f} | {min(t['active_amplitude_r2'] for t in pf):.4f} | {m['test_pass']} | {m['distribution_shifts']['PAIR_TEST']['relative_rmse']:.4f} |")
    lines+=['','The default seed is 1103, fixed before training. Every seed is shown, including failures. Relative RMSE is sqrt(sum squared reconstruction error / sum squared actual response), not percentage error in economic consumption levels. Leakage uses the combined squared error of all five inactive outputs, divided by active-response squared amplitude. Active amplitude R² uses only the feature\'s active examples.','','## Geometry, controls, and interventions','','Six encoder columns represent the six independently controlled features. Selective reconstruction of all six from rank-two coordinates, with distinct signed directions and nonzero amplitudes, is the operational evidence for learned superposition. Antipodal directions are allowed. Raw neuron correlations alone are not the criterion.','','The PCA and two-coordinate-retention references are fitted on TRAIN. Warm-start reconstructions are measured before fine-tuning. Each final model is also tested by adding and removing its feature direction in latent space, decoding and measuring target error and changes to unaffected outputs. Linear encoding equalities are implementation checks; decoded selectivity carries the measured content. Coordinate removals expose how individual latent coordinates participate in multiple features.','','The economic teacher stays exactly additive. Pair errors are outside singleton training support, and the full-width controls can also err there. Compare the matched control before attributing a pair effect to compression. This is a test of learned response compression, not a measured human understanding gap or a proof of intrinsic difficulty.','','## Execution and reproduction','',f"Measured CPU training time: {result['training_seconds']:.3f}s. Pilot projection: {result['pilot']['projected_total_seconds']:.3f}s. No paid computation. No retries or outcome-guided hyperparameter changes.",'','```sh','python learned_superposition_v2.py --pilot','python learned_superposition_v2.py --run','```','','Training writes only the version-2 result files; preserve the existing frozen evidence before rerunning. Independent verification is supplied separately.','']
    (ROOT/(PREFIX+'_report.md')).write_text('\n'.join(lines))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--pilot',action='store_true');parser.add_argument('--run',action='store_true');args=parser.parse_args()
    if args.pilot:pilot()
    elif args.run:run()
    else:parser.error('select --pilot or --run')
