"""Independent verification of frozen teacher, datasets, weights, and reported checks."""
import json, hashlib
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parent
p=json.loads((R/'response_superposition_protocol.json').read_text());r=json.loads((R/'response_superposition_results.json').read_text())
assert hashlib.sha256((R/'response_superposition_protocol.json').read_bytes()).hexdigest()==r['protocol_sha256']
a=.36;b=.96;rng=np.random.default_rng(909);k=np.exp(rng.normal(size=100));z=np.exp(rng.normal(size=100));zn=np.exp(rng.normal(size=100));y=z*k**a;c=(1-a*b)*y;kp=a*b*y;cn=(1-a*b)*zn*kp**a
assert np.allclose(c+kp,y)
assert np.allclose(1/c,b*a*zn*kp**(a-1)/cn)
assert np.allclose((c/((1-a*b)*k**a))-1,z-1)
d=p['data'];rng=np.random.default_rng(d['train_seed']);n=d['train_n'];train=np.zeros((n,6));ids=rng.integers(6,size=n);on=rng.random(n)<.9;train[np.arange(n)[on],ids[on]]=rng.uniform(.8,1.2,on.sum())
rng=np.random.default_rng(d['test_seed']);ids=np.repeat(np.arange(6),128);single=np.zeros((768,6));single[np.arange(768),ids]=rng.uniform(.8,1.2,768);pair=np.zeros((1024,6))
for row in pair:row[rng.choice(6,2,replace=False)]=rng.uniform(.8,1.2,2)
sparse=(rng.random((4096,6))<.05)*rng.uniform(.8,1.2,(4096,6));datasets=dict(train=train,singleton=single,pair=pair,sparse=sparse)
for name,x in datasets.items():assert hashlib.sha256(np.asarray(x,dtype='<f8').tobytes()).hexdigest()==r['dataset_hashes'][name]
mean=train.mean(0);_,_,vt=np.linalg.svd(train-mean,full_matrices=False);basis=vt[:2];coord=np.argsort(np.sum(train*train,0))[-2:]
def met(x,y):return {'mse':float(np.mean((x-y)**2)),'relative_rmse':float(np.linalg.norm(x-y)/np.linalg.norm(x))}
for name,x in datasets.items():
 yy=np.zeros_like(x);yy[:,coord]=x[:,coord]
 for key,y in [('pca',np.maximum(0,(x-mean)@basis.T@basis+mean)),('coordinate',yy)]:
  for key2,value in met(x,y).items():assert np.isclose(value,r['baseline'][name][key][key2],atol=1e-12)
assert [(m['width'],m['seed']) for m in r['models']]==[(w,s) for w in [2,6] for s in [11,23,47]]
for m in r['models']:
 e=np.array(m['encoder']);dec=np.array(m['decoder']);bias=np.array(m['bias']);f=lambda x:np.maximum(0,x@e.T@dec.T+bias)
 assert e.shape==(m['width'],6) and dec.shape==(6,m['width']) and bias.shape==(6,)
 for name,x in datasets.items():
  for key,v in met(x,f(x)).items():assert np.isclose(v,m['metrics'][name][key],atol=1e-12)
 norms=np.linalg.norm(e,axis=0);cos=e.T@e/np.outer(norms,norms);assert np.allclose(cos,m['geometry']['pairwise_cosines']);assert m['geometry']['rank']==np.linalg.matrix_rank(e)
 for pf in m['perfeature']:
  i=pf['feature'];x=single[single[:,i]>0];y=f(x);den=np.sum(x[:,i]**2)
  assert np.isclose(np.sqrt(np.sum((y[:,i]-x[:,i])**2)/den),pf['active_relative_rmse'])
  assert np.isclose(np.sqrt(np.sum(y[:,np.arange(6)!=i]**2)/den),pf['combined_inactive_relative_rmse'])
 for q in m['interventions']:
  x=np.array(q['input']);target=np.array(q['target']);i=q['feature'];h=e@x;hp=h+e[:,i];out=np.maximum(0,dec@hp+bias);removed=np.maximum(0,dec@(hp-e[:,i])+bias)
  assert np.allclose(hp,e@target) and np.allclose(out,q['decoded_after']) and np.allclose(removed,q['decoded_removed'])
  assert np.isclose(np.max(abs(hp-e@target)),q['encoding_equality_maxabs'],atol=1e-12)
  assert np.isclose(np.linalg.norm(out-target),q['addition_error_l2']) and np.isclose(np.linalg.norm(removed-x),q['removal_error_l2'])
  assert np.isclose(np.linalg.norm((out-f(x))[np.arange(6)!=i]),q['unaffected_change_l2'])
 for name,examples in m['examples'].items():
  for q in examples:assert np.allclose(e@q['input'],q['latent']) and np.allclose(f(np.array(q['input'])),q['output'])
 cc=cos.copy();np.fill_diagonal(cc,0)
 gates={'all_features_active':max(q['active_relative_rmse'] for q in m['perfeature'])<=.1,'all_features_inactive':max(q['combined_inactive_relative_rmse'] for q in m['perfeature'])<=.1,'beats_pca_by_half':m['metrics']['singleton']['mse']<=.5*r['baseline']['singleton']['pca']['mse'],'beats_coordinates_by_half':m['metrics']['singleton']['mse']<=.5*r['baseline']['singleton']['coordinate']['mse'],'rank':int(np.linalg.matrix_rank(e))==min(m['width'],6),'distinct_directions':float(np.max(abs(cc)))<.9999,'nonzero_columns':float(min(norms))>.001}
 assert gates==m['gates'] and all(gates.values())==m['passes']
assert r['all_compressed_seeds_pass']==all(m['passes'] for m in r['models'] if m['width']==2)
# Teacher negative control: edits add exactly and affect only their own separate economy.
for i in range(6):
 x=np.zeros(6);x[(i+1)%6]=1;edit=np.eye(6)[i];base=1-a*b
 assert np.allclose((base*(1+x+edit)-base)/base,x+edit)
print('PASS: resource constraint, Euler equation, normalized teacher, split hashes, all six models, baselines, geometry, decoded edits, examples, and frozen gates; experiment success =',r['all_compressed_seeds_pass'])
