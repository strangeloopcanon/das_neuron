"""Independent exact integer verification. Does not import generator."""
import json,itertools,math
from pathlib import Path
import numpy as np
r=json.loads((Path(__file__).resolve().parent/'constructive_superposition.json').read_text())
ring=(2,1,-1,-2,-1,1)
g=[[ring[(i-j)%6] for j in range(6)] for i in range(6)]
assert g==r['exact_gram2']
e=np.asarray(r['encoder']);d=np.asarray(r['decoder']);b=np.asarray(r['bias'])
expected=np.asarray([[math.cos(i*math.pi/3) for i in range(6)],[math.sin(i*math.pi/3) for i in range(6)]])
assert np.allclose(e,expected,atol=1e-12,rtol=0)
assert np.allclose(d,2*e.T,atol=1e-12,rtol=0) and np.array_equal(b,-np.ones(6))
assert np.linalg.matrix_rank(e)==2 and np.allclose(d@e,g,atol=1e-12,rtol=0)
assert np.array_equal(r['control']['encoder'],np.eye(6)) and np.array_equal(r['control']['decoder'],np.eye(6)) and np.array_equal(r['control']['bias'],np.zeros(6))
a=.36;beta=.96;c0=1-a*beta
assert r['teacher']['baseline_consumption']==c0
for k,z,zn in [(0.7,1.3,2.1),(1,2,1),(2.3,.8,1.7)]:
 y=z*k**a;c=c0*y;kp=a*beta*y;cn=c0*zn*kp**a
 assert math.isclose(c+kp,y,rel_tol=1e-12)
 assert math.isclose(1/c,beta*a*zn*kp**(a-1)/cn,rel_tol=1e-12)
cases=r['all_binary_cases'];assert len(cases)==64
counts={};exact_count=0
for bits,q in zip(itertools.product([0,1],repeat=6),cases):
 assert q['input']==list(bits) and q['id']==''.join(map(str,bits))
 active=[i for i,v in enumerate(bits) if v]
 assert q['active_features']==active and q['active_count']==len(active)
 out=[max(0,sum(g[i][j]*bits[j] for j in range(6))-1) for i in range(6)]
 assert out==q['exact_output']
 f=np.maximum(0,d@e@bits+b)
 assert np.max(abs(f-out))<1e-12
 assert np.allclose(f,q['output'],atol=1e-12,rtol=0) and np.allclose(e@bits,q['latent'],atol=1e-12,rtol=0)
 assert np.array_equal(q['control_output'],bits)
 assert np.allclose(q['productivity'],1+np.array(bits),atol=1e-12,rtol=0)
 consumption=c0*(1+np.array(bits))
 assert np.allclose(q['consumption'],consumption,atol=1e-12,rtol=0)
 assert np.allclose(q['next_capital'],a*beta*(1+np.array(bits)),atol=1e-12,rtol=0)
 assert np.max(abs(consumption-c0-c0*np.array(bits)))<1e-12
 assert np.isclose(q['teacher_additivity_residual'],np.max(abs(consumption-c0-c0*np.array(bits))),atol=1e-12)
 good=out==list(bits)
 assert q['exact_recovery']==good
 exact_count+=good
 assert np.isclose(q['maxabs_error'],np.max(abs(f-bits)),atol=1e-12)
 if len(active)<=1:assert good
 if len(active)==2:
  i,j=active;sep=min((i-j)%6,(j-i)%6);counts[sep]=counts.get(sep,0)+1
  if sep==1: expected_out=[2 if k in active else 0 for k in range(6)]
  elif sep==2:
   intermediate=next(k for k in range(6) if g[k][i]==1 and g[k][j]==1)
   expected_out=[int(k==intermediate) for k in range(6)]
  else:expected_out=[0]*6
  assert out==expected_out and not good
assert counts=={1:6,2:6,3:3} and exact_count==7
assert [q['id'] for q in r['support_cases']]==[q['id'] for q in cases if q['active_count']<=1]
assert len(r['pair_cases'])==15
for q in r['pair_cases']:
 source=next(c for c in cases if c['id']==q['id'])
 assert all(q[k]==v for k,v in source.items())
 i,j=q['active_features']
 assert q['angular_separation_degrees']==60*min((i-j)%6,(j-i)%6)
m=r['metrics']
assert m['support_exact_count']==m['support_count']==7 and m['pair_count']==15 and m['pair_exact_count']==0 and m['all_binary_exact_count']==7 and m['all_binary_count']==m['control_all_binary_exact_count']==64 and m['encoder_rank']==2
assert m['pair_classes']=={'adjacent':6,'120_degrees':6,'opposite':3}
assert np.allclose(m['gram_cosines'],e.T@e,atol=1e-12,rtol=0)
assert np.isclose(m['support_maxabs_error'],max(q['maxabs_error'] for q in cases if q['active_count']<=1),atol=1e-12)
print('PASS: exact integer Gram proof; 7 support cases; all 15 pair failures; all 64 binary inputs; 6D control; teacher resource/Euler and additivity.')
