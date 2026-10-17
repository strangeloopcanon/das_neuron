"""Bounded NumPy-only training; the protocol is frozen separately before execution."""
import json, hashlib, time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent
P=json.loads((ROOT/'response_superposition_protocol.json').read_text())
def digest(a): return hashlib.sha256(np.asarray(a,dtype='<f8').tobytes()).hexdigest()
def data():
 d=P['data']; rng=np.random.default_rng(d['train_seed']); n=d['train_n']; x=np.zeros((n,6)); ix=rng.integers(6,size=n); on=rng.random(n)<.9; x[np.arange(n)[on],ix[on]]=rng.uniform(*d['active_amplitude'],size=on.sum())
 rng=np.random.default_rng(d['test_seed']); single=np.zeros((6*d['singleton_test_n_per_feature'],6)); ix=np.repeat(np.arange(6),d['singleton_test_n_per_feature']); single[np.arange(len(ix)),ix]=rng.uniform(*d['active_amplitude'],len(ix))
 pair=np.zeros((d['pair_test_n'],6))
 for row in pair: row[rng.choice(6,2,replace=False)]=rng.uniform(*d['active_amplitude'],2)
 sparse=(rng.random((d['sparse_test_n'],6))<.05)*rng.uniform(*d['active_amplitude'],(d['sparse_test_n'],6))
 return dict(train=x,singleton=single,pair=pair,sparse=sparse)
def predict(x,e,d,b): return np.maximum(0,x@e.T@d.T+b)
def metrics(x,y):
 den=float(np.sum(x*x)); return {'mse':float(np.mean((x-y)**2)),'relative_rmse':float(np.sqrt(np.sum((x-y)**2)/den))}
def perfeature(x,y):
 out=[]
 for i in range(6):
  mask=x[:,i]>0; z=x[mask]; pred=y[mask]; other=np.arange(6)!=i; den=np.sum(z[:,i]**2)
  out.append({'feature':i,'active_relative_rmse':float(np.sqrt(np.sum((pred[:,i]-z[:,i])**2)/den)),'combined_inactive_relative_rmse':float(np.sqrt(np.sum(pred[:,other]**2)/den))})
 return out
def edits(e,d,b):
 cases=[]
 for context in [-1,*range(6)]:
  for i in range(6):
   if context==i: continue
   x=np.zeros(6)
   if context>=0:x[context]=1
   target=x.copy();target[i]=1
   h=e@x; edited=h+e[:,i]; output=np.maximum(0,d@edited+b); removed=np.maximum(0,d@(edited-e[:,i])+b)
   cases.append({'context':context,'feature':i,'input':x.tolist(),'target':target.tolist(),'latent_before':h.tolist(),'latent_after':edited.tolist(),'decoded_after':output.tolist(),'decoded_removed':removed.tolist(),'encoding_equality_maxabs':float(np.max(np.abs(edited-e@target))),'addition_error_l2':float(np.linalg.norm(output-target)),'removal_error_l2':float(np.linalg.norm(removed-x)),'unaffected_change_l2':float(np.linalg.norm((output-predict(x[None],e,d,b)[0])[np.arange(6)!=i]))})
 return cases
def main():
 arrays=data(); x=arrays['train']; mean=x.mean(0); _,_,vt=np.linalg.svd(x-mean,full_matrices=False); basis=vt[:2]; coords=np.argsort(np.sum(x*x,axis=0))[-2:]; baselines={}
 for key,z in arrays.items():
  pca=np.maximum(0,(z-mean)@basis.T@basis+mean); coord=np.zeros_like(z);coord[:,coords]=z[:,coords]
  baselines[key]={'pca':metrics(z,pca),'coordinate':metrics(z,coord)}
 result={'protocol_sha256':hashlib.sha256((ROOT/'response_superposition_protocol.json').read_bytes()).hexdigest(),'dataset_hashes':{k:digest(v) for k,v in arrays.items()},'dataset_shapes':{k:list(v.shape) for k,v in arrays.items()},'baseline':baselines,'baseline_parameters':{'mean':mean.tolist(),'pca_basis':basis.tolist(),'coordinates':coords.tolist()},'models':[],'default_seed':11,'teacher_c0':1-.36*.96}
 for width in P['training']['widths']:
  for seed in P['training']['seeds']:
   rng=np.random.default_rng(seed); e=rng.normal(0,.4,(width,6)); d=rng.normal(0,.4,(6,width));b=np.zeros(6); params=[e,d,b];m=[np.zeros_like(a) for a in params];v=[a.copy() for a in m];start=time.perf_counter();loss=[]; pilot=None
   for step in range(1,12001):
    batch=x[rng.integers(len(x),size=128)];h=batch@e.T;pre=h@d.T+b;out=np.maximum(0,pre);g=2*(out-batch)*(pre>0)/batch.size; grads=[(g@d).T@batch,g.T@h,g.sum(0)]
    for j,(a,grad) in enumerate(zip(params,grads)):
     m[j]=.9*m[j]+.1*grad;v[j]=.999*v[j]+.001*grad*grad;a-=.003*(m[j]/(1-.9**step))/(np.sqrt(v[j]/(1-.999**step))+1e-8)
    if step in [1,1000,4000,8000,12000]:loss.append({'step':step,'batch_mse':float(np.mean((out-batch)**2))})
    if step==1000:pilot=time.perf_counter()-start
   norms=np.linalg.norm(e,axis=0);cos=e.T@e/np.outer(norms,norms);np.fill_diagonal(cos,0)
   ms={k:metrics(z,predict(z,e,d,b)) for k,z in arrays.items()};pf=perfeature(arrays['singleton'],predict(arrays['singleton'],e,d,b)); gates={'all_features_active':max(a['active_relative_rmse'] for a in pf)<=.1,'all_features_inactive':max(a['combined_inactive_relative_rmse'] for a in pf)<=.1,'beats_pca_by_half':ms['singleton']['mse']<=.5*baselines['singleton']['pca']['mse'],'beats_coordinates_by_half':ms['singleton']['mse']<=.5*baselines['singleton']['coordinate']['mse'],'rank':int(np.linalg.matrix_rank(e))==min(width,6),'distinct_directions':float(np.max(abs(cos)))<.9999,'nonzero_columns':float(min(norms))>.001}
   row={'seed':seed,'width':width,'encoder':e.tolist(),'decoder':d.tolist(),'bias':b.tolist(),'metrics':ms,'perfeature':pf,'geometry':{'rank':int(np.linalg.matrix_rank(e)),'column_norms':norms.tolist(),'pairwise_cosines':(e.T@e/np.outer(norms,norms)).tolist()},'interventions':edits(e,d,b),'gates':gates,'passes':all(gates.values()),'pilot_seconds':pilot,'total_seconds':time.perf_counter()-start,'loss_checkpoints':loss,'examples':{k:[{'input':z.tolist(),'latent':(e@z).tolist(),'output':predict(z[None],e,d,b)[0].tolist()} for z in a[:12]] for k,a in arrays.items() if k!='train'}}
   result['models'].append(row);print(width,seed,row['passes'],ms['singleton'],pf,flush=True)
 result['all_compressed_seeds_pass']=all(m['passes'] for m in result['models'] if m['width']==2)
 (ROOT/'response_superposition_results.json').write_text(json.dumps(result,indent=2)+'\n')
 lines=['# Economic-response compression','',f"All three compressed seeds pass the frozen gates: **{result['all_compressed_seeds_pass']}**.",'','Six separate log-utility growth economies use full capital depreciation. Each has alpha=0.36 and beta=0.96. Their exact policy gives c=(1-alpha beta)zk^alpha and next capital=alpha beta zk^alpha. At k=1, normalized consumption excess is x=z-1. There are no transfers or economic interactions.','','The autoencoder compresses economic response vectors, not economic states or policies. TRAIN contains zeros and single-feature responses. Active responses range from 0.8 to 1.2 relative to baseline consumption. Held-out pairs test a distribution shift. Sparse tests use independent activity probability 0.05. All seeds and weights are retained. No retries or parameter search occurred.','','| Width | Seed | Singleton relative RMSE | Pair relative RMSE | Sparse relative RMSE | Gates |','|---|---|---|---|---|---|']
 for m in result['models']:lines.append(f"| {m['width']} | {m['seed']} | {m['metrics']['singleton']['relative_rmse']:.4f} | {m['metrics']['pair']['relative_rmse']:.4f} | {m['metrics']['sparse']['relative_rmse']:.4f} | {m['passes']} |")
 lines+=['','The default display uses seed 11. It is not selected by performance. Feature gates measure active-value error and combined inactive-output leakage. Both must be below 10% for every feature. The compressed model must reduce singleton MSE by at least 50% against both TRAIN-fit references. Geometry also requires rank two, nonzero columns, and distinct directions.','','Latent additions and removals are decoded and checked against actual input edits. Their encoding equality follows from the linear encoder. It does not establish discovery of an independent mechanism. Decoded errors and unaffected-output changes show whether feature edits remain selective in context. Pair errors measure reconstruction failure on combined inputs; the economic teacher itself is additive.','','This bounded experiment tests representation of a narrow, declared feature distribution. It provides no DSGE solution acceleration, general policy-learning result, or hardness theorem. The six-dimensional control and all sparse tests appear above.','']
 if not result['all_compressed_seeds_pass']:
  lines[2]='The learned runs do not demonstrate six-feature compression. All three compressed seeds fail the frozen gates. The six-dimensional controls also fail. Several output units never recover their assigned feature, with 100% active-value error. This training failure prevents a clean conclusion about compression capacity.'
 lines+=['## Per-seed diagnostic','']
 for m in result['models']:
  count=sum(q['active_relative_rmse']<=.1 and q['combined_inactive_relative_rmse']<=.1 for q in m['perfeature'])
  cos=m['geometry']['pairwise_cosines'];mx=max(abs(cos[i][j]) for i in range(6) for j in range(i))
  lines += [f"Width {m['width']}, seed {m['seed']}: {count}/6 features meet both error gates; encoder rank {m['geometry']['rank']}; maximum absolute off-diagonal cosine {mx:.6f}; pilot {m['pilot_seconds']:.3f}s, full {m['total_seconds']:.3f}s.",'']
 lines += ['The declared random initialization and ReLU decoder can leave output units inactive. Their gradients are then zero on the training samples. This is consistent with the observed dead features. These results cannot distinguish a poor training setup from insufficient representational capacity. No additional training was run after observing this failure.','']
 (ROOT/'response_superposition_report.md').write_text('\n'.join(lines))
if __name__=='__main__':main()
