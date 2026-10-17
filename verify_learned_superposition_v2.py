"""Read-only numerical audit. Does not import or execute the training source."""
import hashlib
import itertools
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parent
PREFIX='learned_superposition_v2'
checks=0
max_error=0.

def require(ok,message):
    global checks
    checks+=1
    if not ok: raise AssertionError(message)

def same(actual,expected,label):
    global max_error
    if isinstance(expected,dict):
        require(set(actual)==set(expected),label+' keys')
        for key in expected:same(actual[key],expected[key],label+'.'+key)
    elif isinstance(expected,list):
        require(len(actual)==len(expected),label+' length')
        for i,(a,b) in enumerate(zip(actual,expected)):same(a,b,label+str(i))
    elif isinstance(expected,(bool,str,int)):
        require(actual==expected,label)
    else:
        a,b=float(actual),float(expected)
        require(np.isfinite(a) and np.isfinite(b),label+' finite')
        max_error=max(max_error,abs(a-b))
        require(np.isclose(a,b,rtol=2e-8,atol=2e-10),f'{label}: {a} != {b}')

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def predict(x,w,b):return np.maximum(x@w.T@w+b,0)
def basic(x,y):return dict(mse=float(np.mean(np.square(x-y))),relative_rmse=float(np.linalg.norm(x-y)/np.linalg.norm(x)))
def measure(x,w,b):
    y=predict(x,w,b);out=basic(x,y);out['per_feature']=[]
    for i in range(6):
        select=(x[:,i]>0)&(np.count_nonzero(x,axis=1)==1);t=x[select,i];v=y[select];norm=np.linalg.norm(t)
        out['per_feature'].append(dict(feature=i,active_relative_rmse=float(np.linalg.norm(v[:,i]-t)/norm),combined_inactive_relative_rmse=float(np.linalg.norm(v[:,np.arange(6)!=i])/norm),active_amplitude_r2=float(1-np.sum((v[:,i]-t)**2)/np.sum((t-t.mean())**2))))
    out['zero_output_max']=float(np.max(np.abs(predict(np.zeros((1,6)),w,b))))
    return out

def fullpass(m,p):
    g=p['full_width_gates']
    return bool(all(f['active_relative_rmse']<=g['active_relative_rmse_max'] and f['combined_inactive_relative_rmse']<=g['combined_inactive_relative_rmse_max'] and f['active_amplitude_r2']>=g['active_amplitude_r2_min'] for f in m['per_feature']) and m['zero_output_max']<=g['zero_output_max'])

def regenerate(name,p):
    spec=p['data'][name];rng=np.random.default_rng(spec['seed']);lo,hi=p['data']['amplitude']
    if name in ['TRAIN','VALIDATION','TEST','EDIT_TEST']:
        blocks=[]
        for j in range(6):
            a=np.zeros((spec['per_feature'],6));a[:,j]=rng.uniform(lo,hi,len(a));blocks.append(a)
        if name!='EDIT_TEST':blocks.append(np.zeros((spec['zero'],6)))
        x=np.concatenate(blocks)
        if name!='EDIT_TEST':rng.shuffle(x)
    elif name=='PAIR_TEST':
        blocks=[]
        for pair in itertools.combinations(range(6),2):
            a=np.zeros((spec['per_pair'],6));a[:,pair]=rng.uniform(lo,hi,(len(a),2));blocks.append(a)
        x=np.concatenate(blocks);rng.shuffle(x)
    else:x=(rng.random((spec['n'],6))<spec['activity_probability'])*rng.uniform(lo,hi,(spec['n'],6))
    t=p['teacher'];z=1+x;k=t['capital'];output=z*k**t['alpha'];c=(1-t['alpha']*t['beta'])*output;kp=t['alpha']*t['beta']*output
    require(np.allclose(c+kp,output,atol=1e-14),'teacher resource')
    # Euler equation: 1/c = beta E[alpha z_next kp^(alpha-1)/c_next].
    znext=np.full_like(z,1.17);cnext=(1-t['alpha']*t['beta'])*znext*kp**t['alpha']
    require(np.allclose(1/c,t['beta']*t['alpha']*znext*kp**(t['alpha']-1)/cnext),'teacher Euler')
    response=(c-(1-t['alpha']*t['beta'])*k**t['alpha'])/((1-t['alpha']*t['beta'])*k**t['alpha'])
    require(np.allclose(response,x,rtol=1e-13,atol=1e-14),'teacher normalized response')
    return response

def edits(x,w,b):
    isolated=[];contexts=[];zero=predict(np.zeros((1,6)),w,b)[0]
    for i in range(6):
        a=x[x[:,i]>0,i];target=np.zeros((len(a),6));target[:,i]=a;delta=np.outer(a,w[:,i]);y=np.maximum(delta@w+b,0);den=np.linalg.norm(a)
        isolated.append(dict(feature=i,target_relative_rmse=float(np.linalg.norm(y-target)/den),active_relative_rmse=float(np.linalg.norm(y[:,i]-a)/den),inactive_relative_rmse=float(np.linalg.norm(y[:,np.arange(6)!=i])/den),removed_output_max=float(np.max(np.abs(np.maximum((delta-delta)@w+b,0)-zero)))))
        for j in range(6):
            if i==j:continue
            base=np.zeros_like(target);base[:,j]=1;h=base@w.T;edited=h+delta;y=np.maximum(edited@w+b,0);before=predict(base,w,b)
            contexts.append(dict(inserted_feature=i,background_feature=j,target_relative_rmse=float(np.linalg.norm(y-base-target)/np.linalg.norm(base+target)),unaffected_change_relative_rms=float(np.linalg.norm((y-before)[:,np.arange(6)!=i])/den),encoding_identity_maxabs=float(np.max(abs(y-predict(base+target,w,b)))),removal_identity_maxabs=float(np.max(abs(np.maximum((edited-delta)@w+b,0)-before)))))
    ablations=[];original=predict(x,w,b)
    for j in range(len(w)):
        h=x@w.T;h[:,j]=0;change=np.maximum(h@w+b,0)-original
        ablations.append(dict(coordinate=j,per_feature_response_change=[float(np.linalg.norm(change[x[:,i]>0])/np.linalg.norm(x[x[:,i]>0,i])) for i in range(6)]))
    return dict(isolated=isolated,contexts=contexts,coordinate_ablations=ablations)

def confirmation(original,p,data):
    prefix=PREFIX+'_confirmation'
    cp=json.loads((ROOT/(prefix+'_protocol.json')).read_text());cr=json.loads((ROOT/(prefix+'_results.json')).read_text())
    require(cp['original_results_sha256']==sha(ROOT/(PREFIX+'_results.json')),'confirmation original results hash')
    require(cp['original_protocol_sha256']==sha(ROOT/(PREFIX+'_protocol.json')),'confirmation original protocol hash')
    require(cp['source_sha256']==sha(ROOT/(prefix+'.py')),'confirmation source hash')
    require(cr['protocol_sha256']==sha(ROOT/(prefix+'_protocol.json')),'confirmation protocol hash')
    require(cr['archive_sha256']==sha(ROOT/(prefix+'_data.npz')),'confirmation archive hash')
    require(cp['no_retraining'] is True,'confirmation no retraining declaration')
    with np.load(ROOT/(prefix+'_data.npz'),allow_pickle=False) as archive:
        require(archive.files==['CONFIRMATION'],'confirmation archive members');x=archive['CONFIRMATION']
    require(cr['array_sha256']==hashlib.sha256(np.asarray(x,dtype='<f8').tobytes()).hexdigest(),'confirmation canonical hash')
    spec=cp['new_holdout'];rng=np.random.default_rng(spec['seed']);expected=np.zeros((spec['per_feature']*6,6))
    for i in range(6):expected[i*spec['per_feature']:(i+1)*spec['per_feature'],i]=rng.uniform(*spec['amplitude'],spec['per_feature'])
    c0=1-p['teacher']['alpha']*p['teacher']['beta'];expected=(c0*(1+expected)-c0)/c0
    require(x.shape==expected.shape and np.allclose(x,expected,rtol=1e-13,atol=1e-14),'confirmation regenerated data')
    require(spec['feature_groups']==6 and spec['per_feature']==4096 and cr['rows']==24576==len(x),'confirmation row/group counts')
    old={tuple(row) for values in data.values() for row in values if np.any(row)};new={tuple(row) for row in x}
    require(not old&new and len(new)==len(x),'confirmation positive rows unique/disjoint')
    same(cr['primary_all_six_result_unchanged'],original['summary'],'confirmation preserves primary summary')
    full_models=[m for m in original['models'] if m['stage']=='compressed'];require([m['seed'] for m in cp['selection']]==p['training']['seeds'],'confirmation all original seeds retained')
    require([m['seed'] for m in cr['models']]==p['training']['seeds'],'confirmation result seeds')
    criteria=cp['criteria'];g=p['compressed_gates']
    for key in ['active_relative_rmse_max','combined_inactive_relative_rmse_max','active_amplitude_r2_min_exclusive']:same(criteria[key],g[key],'confirmation original criterion '+key)
    require(criteria['required_retained_features_per_model']==5 and criteria['required_model_count']==5,'confirmation required counts')
    def passes(f):return bool(f['active_relative_rmse']<=criteria['active_relative_rmse_max'] and f['combined_inactive_relative_rmse']<=criteria['combined_inactive_relative_rmse_max'] and f['active_amplitude_r2']>criteria['active_amplitude_r2_min_exclusive'])
    expected_models=[]
    for selected,m in zip(cp['selection'],full_models):
        payload=dict(seed=m['seed'],W=m['W'],bias=m['bias']);digest=hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(',',':')).encode()).hexdigest()
        require(selected['model_sha256']==digest,'confirmation frozen model hash')
        val=measure(data['VALIDATION'],np.array(m['W']),np.array(m['bias']))
        retained=[f['feature'] for f in val['per_feature'] if passes(f)]
        same(selected['features_selected_on_validation'],retained,'confirmation VALIDATION-only selection')
        metrics=measure(x,np.array(m['W']),np.array(m['bias']))['per_feature'];records=[]
        for f in metrics:
            i=f['feature'];require(np.sum(x[:,i]>0)==4096,'confirmation per-feature rows')
            records.append(dict(**f,selected_on_validation=i in retained,passes=passes(f)))
        active=[f for f in records if f['selected_on_validation']];count=sum(f['passes'] for f in active)
        expected_models.append(dict(seed=m['seed'],model_sha256=digest,per_feature=records,selected_count=len(active),confirmed_count=count,confirmation_pass=len(active)>=5 and all(f['passes'] for f in active)))
    same(cr['models'],expected_models,'confirmation all model metrics and flags')
    count=sum(m['confirmation_pass'] for m in expected_models);passed=len(expected_models)==5 and count==5
    same(cr['confirmed_model_count'],count,'confirmation confirmed count');same(cr['confirmation_pass'],passed,'confirmation overall pass')
    return dict(confirmed_model_count=count,confirmation_pass=passed,rows=len(x),selected_features_per_model=[m['selected_count'] for m in expected_models],primary_study_pass=original['summary']['study_pass'])

def main():
    p=json.loads((ROOT/(PREFIX+'_protocol.json')).read_text());r=json.loads((ROOT/(PREFIX+'_results.json')).read_text());pilot=json.loads((ROOT/(PREFIX+'_pilot.json')).read_text())
    for obj in [r,pilot]:
        require(obj['protocol_sha256']==sha(ROOT/(PREFIX+'_protocol.json')),'protocol hash')
        require(obj['source_sha256']==sha(ROOT/(PREFIX+'.py')),'source hash')
    same(r['pilot'],pilot,'pilot');require(pilot['test_evaluated'] is False and pilot['weights_discarded'] is True,'pilot authority');require(pilot['projected_total_seconds']<=p['pilot']['runtime_budget_seconds'],'runtime budget')
    require(r['data_archive_sha256']==sha(ROOT/(PREFIX+'_data.npz')),'archive hash');require(p['frozen_before_training'] is True,'frozen flag')
    with np.load(ROOT/(PREFIX+'_data.npz'),allow_pickle=False) as archive:data={k:archive[k] for k in archive.files}
    require(set(data)=={'TRAIN','VALIDATION','TEST','PAIR_TEST','SPARSE_TEST','EDIT_TEST'},'data names');seen=set()
    for name,x in data.items():
        require(hashlib.sha256(np.asarray(x,dtype='<f8').tobytes()).hexdigest()==r['array_hashes'][name],name+' canonical hash')
        same(r['array_shapes'][name],list(x.shape),name+' shape');require(np.allclose(x,regenerate(name,p),rtol=1e-13,atol=1e-14),name+' regenerated')
        positive={tuple(row) for row in x if np.any(row)};require(not positive&seen,name+' positive split disjoint');seen|=positive
        spec=p['data'][name]
        if name in ['TRAIN','VALIDATION','TEST','EDIT_TEST']:
            for i in range(6):require(np.sum((x[:,i]>0)&(np.count_nonzero(x,axis=1)==1))==spec['per_feature'],name+' group')
            require(np.sum(~np.any(x,axis=1))==spec.get('zero',0),name+' zero count')
        elif name=='PAIR_TEST':
            for i,j in itertools.combinations(range(6),2):require(np.sum((x[:,i]>0)&(x[:,j]>0))==spec['per_pair'],'pair count')
    train=data['TRAIN'];test=data['TEST'];require(np.array_equal(predict(test,np.eye(6),np.zeros(6)),test),'analytic six-dimensional identity control');same(r['default_seed'],p['training']['default_seed'],'default seed');mean=train.mean(0);_,_,vt=np.linalg.svd(train-mean,full_matrices=False);basis=vt[:2];coords=np.argsort(np.sum(train**2,axis=0),kind='stable')[-2:];yp=np.maximum((test-mean)@basis.T@basis+mean,0);yc=np.zeros_like(test);yc[:,coords]=test[:,coords]
    base=dict(mean=mean.tolist(),basis=basis.tolist(),coordinates=coords.tolist(),pca=basic(test,yp),coordinates_reference=basic(test,yc));same(r['baselines'],base,'baseline')
    full=[m for m in r['models'] if m['stage']=='full'];comp=[m for m in r['models'] if m['stage']=='compressed'];require([m['seed'] for m in full]==p['training']['seeds'],'full seeds');prereq=all(fullpass(measure(data['VALIDATION'],np.array(m['W']),np.array(m['bias'])),p) for m in full);same(r['full_validation_prerequisite'],prereq,'prerequisite');require([m['seed'] for m in comp]==(p['training']['seeds'] if prereq else []),'compression authorization')
    for m in r['models']:
        w=np.array(m['W']);b=np.array(m['bias']);seed=m['seed'];width=6 if m['stage']=='full' else 2;require(w.shape==(width,6) and b.shape==(6,),'parameter shape');require(m['parameters']==w.size+b.size==p['architecture']['full_parameters' if width==6 else 'compressed_parameters'],'parameter count');same(m['width'],width,'width');v=measure(data['VALIDATION'],w,b);same(m['validation'],v,'validation')
        require(m['curve'][-1]['step']==p['training']['full_steps' if width==6 else 'compressed_steps'],'final scheduled checkpoint')
        if width==6:
            same(m['initial_W'],np.random.default_rng(seed).normal(0,.4,(6,6)).tolist(),'Gaussian init');same(m['initial_bias'],[0.]*6,'full bias');same(m['validation_pass'],fullpass(v,p),'validation gate')
        else:
            require(m['parent_seed']==seed,'parent');parent=next(q for q in full if q['seed']==seed);g=np.random.default_rng(seed+10000).normal(size=(6,2));q,t=np.linalg.qr(g,mode='reduced');projection=(q*np.where(np.diag(t)>=0,1.,-1.)).T;same(m['projection_gaussian'],g.tolist(),'projection Gaussian');same(m['projection'],projection.tolist(),'QR projection');same(m['initial_W'],(projection@np.array(parent['W'])).tolist(),'projected parent');same(m['initial_bias'],[0.]*6,'reset bias');same(m['initial_test'],measure(test,np.array(m['initial_W']),np.zeros(6)),'warm start')
        norms=np.linalg.norm(w,axis=0);cos=(w.T@w)/np.outer(norms,norms);geo=dict(rank=int(np.linalg.matrix_rank(w)),column_norms=norms.tolist(),cosines=cos.tolist(),maximum_positive_pair_cosine=float(cos[~np.eye(6,dtype=bool)].max()));same(m['geometry'],geo,'geometry');metric=measure(test,w,b);same(m['evaluation'],{'TEST':metric},'test');same(m['unit_response_matrix'],predict(np.eye(6),w,b).tolist(),'unit outputs')
        same(m['distribution_shifts'],{name:basic(data[name],predict(data[name],w,b)) for name in ['PAIR_TEST','SPARSE_TEST']},'shift');pairs=data['PAIR_TEST'];groups=[]
        for i,j in itertools.combinations(range(6),2):
            x=pairs[(pairs[:,i]>0)&(pairs[:,j]>0)];groups.append(dict(features=[i,j],**basic(x,predict(x,w,b))))
        same(m['pair_groups'],groups,'pair groups');same(m['edits'],edits(data['EDIT_TEST'],w,b),'interventions')
        if width==6:passed=fullpass(metric,p)
        else:
            g=p['compressed_gates'];pf=metric['per_feature'];flags=dict(all_features_active=all(t['active_relative_rmse']<=g['active_relative_rmse_max'] for t in pf),all_features_selective=all(t['combined_inactive_relative_rmse']<=g['combined_inactive_relative_rmse_max'] for t in pf),all_features_amplitude=all(t['active_amplitude_r2']>g['active_amplitude_r2_min_exclusive'] for t in pf),beats_pca=metric['mse']<=g['mse_ratio_to_train_fit_pca_max']*base['pca']['mse'],beats_coordinates=metric['mse']<=g['mse_ratio_to_train_fit_coordinate_reference_max']*base['coordinates_reference']['mse'],rank=geo['rank']==2,nonzero=min(norms)>=g['column_norm_min'],distinct_signed_directions=geo['maximum_positive_pair_cosine']<g['positive_pair_cosine_max']);same(m['gates'],flags,'compressed flags');passed=all(flags.values())
        same(m['test_pass'],passed,'test pass')
    summary=dict(full_validation_pass_count=sum(m['validation_pass'] for m in full),full_test_pass_count=sum(m['test_pass'] for m in full),compressed_test_pass_count=sum(m['test_pass'] for m in comp),compressed_test_count=len(comp),study_pass=len(comp)==5 and len(full)==5 and all(m['test_pass'] for m in full) and sum(m['test_pass'] for m in comp)>=4);same(r['summary'],summary,'summary')
    confirmation_summary=confirmation(r,p,data)
    print(json.dumps(dict(confirmation=confirmation_summary,verification='PASS',checks=checks,max_absolute_recomputation_difference=max_error,summary=summary,source_sha256=r['source_sha256'],protocol_sha256=r['protocol_sha256'],data_archive_sha256=r['data_archive_sha256'],limitation='Saved artifact and source audit; no training replay or independent timestamp proof of freeze.'),indent=2))

if __name__=='__main__':main()
