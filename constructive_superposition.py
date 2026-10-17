"""Deterministic six-feature, two-coordinate constructed response code. No training."""
from pathlib import Path
import itertools,json
import numpy as np
R=Path(__file__).resolve().parent

def build():
 angles=np.arange(6)*np.pi/3;e=np.array([np.cos(angles),np.sin(angles)]);d=2*e.T;b=-np.ones(6);a=.36;beta=.96;c0=1-a*beta
 # 2*cos(j*pi/3), represented as exact integers.
 ring=[2,1,-1,-2,-1,1];gram=np.array([[ring[(i-j)%6] for j in range(6)] for i in range(6)])
 cases=[]
 for bits in itertools.product([0,1],repeat=6):
  x=np.array(bits);h=e@x;y=np.maximum(0,d@h+b);exact=np.maximum(0,gram@x-1);teacher=c0*(1+x);control=np.maximum(0,np.eye(6)@x)
  assert np.max(abs(y-exact))<1e-12
  cases.append({'id':''.join(map(str,bits)),'input':list(bits),'active_features':np.flatnonzero(x).tolist(),'active_count':int(sum(bits)),'productivity':(1+x).tolist(),'consumption':teacher.tolist(),'next_capital':(a*beta*(1+x)).tolist(),'latent':h.tolist(),'output':y.tolist(),'exact_output':exact.tolist(),'control_output':control.tolist(),'maxabs_error':float(np.max(abs(y-x))),'exact_recovery':bool(np.array_equal(exact,x)),'teacher_additivity_residual':float(np.max(abs(teacher-c0-c0*x)))})
 support=[q for q in cases if q['active_count']<=1];pairs=[dict(q) for q in cases if q['active_count']==2]
 for q in pairs:
  i,j=q['active_features'];sep=min((i-j)%6,(j-i)%6);q['angular_separation_degrees']=60*sep;q['class']={1:'adjacent: both active outputs double',2:'120 degrees: intermediate sector only',3:'opposite: all outputs zero'}[sep]
 r={'kind':'deterministic constructive witness; no training','scope':'Exact recovery of zero and six singleton binary consumption-response inputs. Independent economies, no transfers; no learned policy or DSGE solver claim. Binary amplitude only.','teacher':{'alpha':a,'beta':beta,'depreciation':1,'capital':1,'baseline_consumption':c0,'utility':'six separate log-utility economies with equal weights','policy_consumption':'(1-alpha*beta)*z*k**alpha','policy_next_capital':'alpha*beta*z*k**alpha','productivity_values':[1,2],'response':'x=(c-c0)/c0=z-1'},'feature_names':[f'Economy {i+1}' for i in range(6)],'encoder':e.tolist(),'decoder':d.tolist(),'bias':b.tolist(),'angles_degrees':(np.arange(6)*60).tolist(),'exact_gram2':gram.tolist(),'support_cases':support,'pair_cases':pairs,'all_binary_cases':cases,'control':{'encoder':np.eye(6).tolist(),'decoder':np.eye(6).tolist(),'bias':np.zeros(6).tolist()},'metrics':{'support_exact_count':sum(q['exact_recovery'] for q in support),'support_count':len(support),'support_maxabs_error':max(q['maxabs_error'] for q in support),'pair_exact_count':sum(q['exact_recovery'] for q in pairs),'pair_count':len(pairs),'all_binary_exact_count':sum(q['exact_recovery'] for q in cases),'all_binary_count':len(cases),'control_all_binary_exact_count':64,'pair_classes':{'adjacent':6,'120_degrees':6,'opposite':3},'encoder_rank':int(np.linalg.matrix_rank(e)),'gram_cosines':(e.T@e).tolist()}}
 (R/'constructive_superposition.json').write_text(json.dumps(r,indent=2)+'\n')
 return r
if __name__=='__main__':
 r=build();print(json.dumps(r['metrics'],indent=2))
