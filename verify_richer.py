"""Read-only reconstruction of saved richer DSGE evidence; never trains or writes.

Teacher policy interpolation/static recovery is reused, but every checked FOC,
network forward pass, intervention, response metric, split, and fit is recomputed.
This verifies saved artifacts; it does not replay optimizer updates or prove
Bellman optimality.
"""
import os
for _key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[_key] = '1'
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from richer_teacher import load_teacher, RicherTeacher

ROOT = Path(__file__).resolve().parent
KEYS = ('W1','b1','W2','b2','W3','b3')
OUTPUTS = ('c','labor','investment')


def close(actual, expected, label, atol=2e-10, rtol=2e-8):
    np.testing.assert_allclose(actual, expected, atol=atol, rtol=rtol, err_msg=label)


def check(condition, label):
    if not condition:
        raise AssertionError(label)


def sha(path):
    return hashlib.sha256((ROOT/path).read_bytes()).hexdigest()


def array_sha(*arrays):
    h = hashlib.sha256()
    for array in arrays:
        a = np.ascontiguousarray(array)
        h.update(str(a.dtype).encode()); h.update(str(a.shape).encode()); h.update(a.tobytes())
    return h.hexdigest()


def forward(x, net, layer=0, unit=0, fraction=1.):
    first = np.maximum(x@net['W1'].T+net['b1'],0)
    if layer == 1:
        first[:,unit] *= 1-fraction
    gates = first@net['W2'].T+net['b2']
    final = np.maximum(gates,0)
    if layer == 2:
        final[:,unit] *= 1-fraction
    return final@net['W3']+net['b3'], (first,gates,final)


def response(actual,predicted):
    denominator = float(np.sqrt(np.mean(actual**2)))
    error = float(np.sqrt(np.mean((actual-predicted)**2)))
    active = abs(actual)>1e-10
    return dict(response_rms_train_sd=denominator,prediction_error_rms_train_sd=error,
        normalized_response_rmse=error/max(denominator,1e-30),
        sign_accuracy=float(np.mean(np.sign(actual[active])==np.sign(predicted[active]))) if active.any() else None,
        active_output_effects=int(active.sum()))


def compare_metrics(actual,saved,label):
    check(actual.keys() <= saved.keys(),label+' missing metrics')
    for key,value in actual.items():
        if value is None or isinstance(value,(bool,str)):
            check(value == saved[key],label+' '+key)
        else:
            close(value,saved[key],label+' '+key)


def fit(prediction,truth):
    difference=prediction-truth
    return dict(normalized_rmse=np.sqrt(np.mean(difference**2,axis=0)),
        normalized_max_error=np.max(abs(difference),axis=0))


def independent_chain(count,rho,innovation):
    probability=(1+rho)/2
    matrix=np.array([[probability,1-probability],[1-probability,probability]])
    for size in range(3,count+1):
        old=matrix;matrix=np.zeros((size,size))
        matrix[:-1,:-1]+=probability*old;matrix[:-1,1:]+=(1-probability)*old
        matrix[1:,:-1]+=(1-probability)*old;matrix[1:,1:]+=probability*old
        matrix[1:-1]/=2
    from math import comb
    stationary=np.array([comb(count-1,j)/2**(count-1) for j in range(count)])
    levels=np.linspace(-1,1,count)*innovation/np.sqrt(1-rho*rho)*np.sqrt(count-1)
    return matrix,levels,stationary


def foc_residuals(teacher,k,state,policy=None):
    p=teacher.parameters
    current=teacher.policy(k,state) if policy is None else policy(k,state)
    c,n,i=current['c'],current['labor'],current['investment']
    kp=(1-p['delta'])*k+i
    check(np.all(c>0)&np.all(n>0)&np.all(kp>teacher.k_grid[0])&np.all(kp<teacher.k_grid[-1]),'FOC positive/interior states')
    y=np.exp(teacher.state_features[state,0])*k**p['alpha']*n**(1-p['alpha'])
    adjustment=.5*p['kappa']*k*(i/k-p['delta'])**2
    q=1+p['kappa']*(i/k-p['delta'])
    check(np.all(q>0),'FOC positive q')
    labor=p['chi']*np.exp(teacher.state_features[state,1])*n**p['phi']
    wage_mu=c**(-p['sigma'])*(1-p['alpha'])*y/n
    expectation=np.zeros(len(k))
    for future_state in range(len(teacher.P)):
        s=np.full(len(k),future_state,dtype=int)
        future=teacher.policy(kp,s) if policy is None else policy(kp,s)
        cf,nf,iv=future['c'],future['labor'],future['investment']
        check(np.all(cf>0)&np.all(nf>0),'future FOC positivity')
        mpk=p['alpha']*np.exp(teacher.state_features[s,0])*kp**(p['alpha']-1)*nf**(1-p['alpha'])
        qf=1+p['kappa']*(iv/kp-p['delta'])
        phik=.5*p['kappa']*(p['delta']**2-(iv/kp)**2)
        expectation+=teacher.P[state,future_state]*cf**(-p['sigma'])*(mpk+(1-p['delta'])*qf-phik)
    euler=p['beta']*expectation/(c**(-p['sigma'])*q)-1
    return dict(resource_relative_rms=float(np.sqrt(np.mean(((c+i+adjustment-y)/y)**2))),
        resource_relative_max=float(np.max(abs((c+i+adjustment-y)/y))),
        labor_foc_relative_rms=float(np.sqrt(np.mean((labor/wage_mu-1)**2))),
        labor_foc_relative_max=float(np.max(abs(labor/wage_mu-1))),
        euler_relative_rms=float(np.sqrt(np.mean(euler**2))),euler_relative_max=float(np.max(abs(euler))))


def verify_teacher(teacher,report):
    p=teacher.parameters
    for key,value in report['parameters'].items():close(p[key],value,'teacher parameters '+key)
    expected=dict(alpha=.36,beta=.96,delta=.08,sigma=2.,phi=1.,kappa=2.,rho=.9,innovation_std=.03,labor_rho=.7,labor_innovation_std=.04,nss=.33)
    for key,value in expected.items():close(p[key],value,'calibration '+key)
    kss=p['nss']*(p['alpha']/(1/p['beta']-1+p['delta']))**(1/(1-p['alpha']))
    yss=kss**p['alpha']*p['nss']**(1-p['alpha']);css=yss-p['delta']*kss
    close([kss,yss,css,(1-p['alpha'])*yss/(p['nss']**(1+p['phi'])*css**p['sigma'])],
        [p['kss'],p['yss'],p['css'],p['chi']],'steady state')
    A,z,az=independent_chain(7,p['rho'],p['innovation_std'])
    B,eta,be=independent_chain(3,p['labor_rho'],p['labor_innovation_std'])
    close(teacher.P,np.kron(A,B),'independent joint Markov chain')
    close(teacher.state_features,[(a,b) for a in z for b in eta],'joint shock ordering')
    close(teacher.stationary,np.kron(az,be),'stationary joint law')
    close(teacher.stationary@teacher.P,teacher.stationary,'stationarity')
    close(teacher.P.sum(1),1,'transition mass');check(np.all(teacher.P>=0),'negative transition probability')
    check(np.all(np.diff(teacher.k_grid)>0),'capital grid order')
    check(teacher.next_capital_grid.shape==(21,960),'fine policy shape')
    for array in (teacher.k_grid,teacher.next_capital_grid,teacher.P,teacher.state_features):check(np.isfinite(array).all(),'nonfinite teacher')
    check(sha(report['provenance']['source'])==report['provenance']['source_sha256'],'teacher source provenance')
    check(sha(report['provenance']['artifact'])==report['provenance']['artifact_sha256'],'teacher archive provenance')
    check(sha(report['provenance']['rouwenhorst_source'])==report['provenance']['rouwenhorst_source_sha256'],'teacher dependency provenance')
    # Derivatives of installation cost: hold the other argument fixed.
    rng=np.random.default_rng(714018)
    capital=rng.uniform(.5,1.8,128)*kss;investment=rng.uniform(-.03,.2,128)*capital;step=1e-5*capital
    def adjustment(k,i):return .5*p['kappa']*k*(i/k-p['delta'])**2
    derivative_k=(adjustment(capital+step,investment)-adjustment(capital-step,investment))/(2*step)
    derivative_i=(adjustment(capital,investment+step)-adjustment(capital,investment-step))/(2*step)
    close(derivative_k,.5*p['kappa']*(p['delta']**2-(investment/capital)**2),'installation partial capital',atol=1e-8)
    close(derivative_i,p['kappa']*(investment/capital-p['delta']),'installation partial investment',atol=1e-8)
    rng=np.random.default_rng(714019)
    k=rng.uniform(.5,1.8,1536)*kss;s=rng.integers(21,size=len(k))
    metrics=foc_residuals(teacher,k,s)
    check(all(np.isfinite(v) for v in metrics.values()),'nonfinite fresh teacher residuals')
    check(metrics['resource_relative_max']<1e-10,'fresh teacher resource residual')
    check(metrics['labor_foc_relative_max']<1e-5,'fresh teacher labor residual')
    check(metrics['euler_relative_max']<.002,'fresh teacher Euler residual')
    with np.load(ROOT/'richer_teacher.npz',allow_pickle=False) as saved:
        coarse=RicherTeacher(saved['coarse_k_grid'],saved['coarse_next_capital_grid'],teacher.P,teacher.state_features,teacher.stationary,p)
    # Reconstruct the original refinement sample, not just the reported scalar.
    rng=np.random.default_rng(81006);k=rng.uniform(.5,1.8,12000)*kss;s=rng.choice(21,12000,p=teacher.stationary)
    fine_pol=teacher.policy(k,s);coarse_pol=coarse.policy(k,s)
    for key,saved in report['grid_refinement'].items():
        for label,array in [('absolute',coarse_pol[key]-fine_pol[key]),('relative',coarse_pol[key]/fine_pol[key]-1)]:
            close(np.max(abs(array)),saved[label]['maximum'],'refinement '+key+' '+label)
            close(np.sqrt(np.mean(array**2)),saved[label]['rms'],'refinement RMS '+key+' '+label)
    return metrics


def partitions(protocol,teacher,provenance):
    data={};seen=set()
    for name,spec in protocol['data'].items():
        if not isinstance(spec,dict):continue
        rng=np.random.default_rng(spec['seed']);k=rng.uniform(.5,1.8,spec['n'])*teacher.parameters['kss']
        state=rng.choice(21,spec['n'],p=teacher.stationary)
        raw=np.column_stack((k,teacher.state_features[state]));pol=teacher.policy(k,state)
        y=np.column_stack([pol[key] for key in OUTPUTS]);membership={(float(a).hex(),int(b)) for a,b in zip(k,state)}
        check(len(membership)==len(k) and not membership&seen,'partition disjointness '+name);seen|=membership
        check(array_sha(k,state.astype(np.int64),y)==provenance['partitions'][name]['sha256'],'partition digest '+name)
        close(np.bincount(state,minlength=21),provenance['partitions'][name]['shock_state_counts'],'partition state counts '+name)
        check(spec['n']==provenance['partitions'][name]['n'] and spec['seed']==provenance['partitions'][name]['seed'],'partition specification '+name)
        data[name]=dict(raw=raw,state=state,y=y)
    train=data['TRAIN'];norm=dict(input_mean=train['raw'].mean(0),input_std=train['raw'].std(0),output_mean=train['y'].mean(0),output_std=train['y'].std(0))
    for key,value in norm.items():close(value,provenance['normalization'][key],'TRAIN normalization '+key)
    for split in data.values():
        split['x']=(split['raw']-norm['input_mean'])/norm['input_std']
        split['yn']=(split['y']-norm['output_mean'])/norm['output_std']
    return data,norm


def interventions(net,x,saved):
    baseline,(h1,gates,h2)=forward(x,net);width=len(net['b1'])
    # Derivative of each output with respect to each first hidden activation.
    jac=np.empty((len(x),width,3))
    for output in range(3):jac[:,:,output]=((gates>0)*net['W3'][:,output])@net['W2']
    check(len(saved['first_layer'])==3,'intervention fraction count')
    for fraction,record in zip((.01,.1,1.),saved['first_layer']):
        check(len(record['per_output'])==3,'intervention output count')
        close(fraction,record['fraction'],'intervention fraction')
        actual=[];predicted=[];switches=0
        for unit in range(width):
            edited,(_,edited_gates,_)=forward(x,net,1,unit,fraction)
            actual.append(edited-baseline);predicted.append(-fraction*h1[:,unit,None]*jac[:,unit])
            switches+=int(np.any((edited_gates>0)!=(gates>0),axis=1).sum())
        actual=np.stack(actual,1);predicted=np.stack(predicted,1)
        compare_metrics(response(actual,predicted),record,'all-unit local prediction')
        close(switches/(len(x)*width),record['downstream_gate_switch_fraction'],'gate switches')
        for output in range(3):compare_metrics(response(actual[:,:,output],predicted[:,:,output]),record['per_output'][output],'per-output intervention')
    exact=0.
    for unit in range(width):
        actual=forward(x,net,2,unit)[0]-baseline
        exact=max(exact,float(np.max(abs(actual+h2[:,unit,None]*net['W3'][unit]))))
    close(exact,saved['final_layer_max_absolute_prediction_error'],'exact signed-readout ablations',atol=2e-12)
    check(exact<1e-10,'last-layer ablation exactness')


def labels(net,data,norm,teacher,saved):
    x=data['DISCOVERY']['x'];h=forward(x,net)[1][-1]
    h=h-h.mean(0);xc=x-x.mean(0)
    denom=np.sqrt(np.sum(h*h,0)[:,None]*np.sum(xc*xc,0)[None,:])
    corr=np.divide(h.T@xc,denom,out=np.zeros_like(denom),where=denom>1e-14)
    selected=int(np.argmax(abs(corr[:,1])))
    check(selected==saved['unit'],'DISCOVERY label selection')
    close(corr,saved['correlations'],'DISCOVERY correlations');close(corr[selected],saved['selected_correlations'],'selected correlations')
    close(net['W3'][selected],saved['readout_train_output_sd'],'selected signed readout')
    background=data['INTERVENTION_TEST']['raw'];pairs=[]
    lo=background.copy();hi=background.copy()
    lo[:,0]=np.maximum(.5*teacher.parameters['kss'],.9*background[:,0]);hi[:,0]=np.minimum(1.8*teacher.parameters['kss'],1.1*background[:,0]);pairs.append((lo,hi))
    for column in (1,2):
        levels=np.unique(teacher.state_features[:,column-1]);index=int(np.argmin(abs(levels)))
        lo=background.copy();hi=background.copy();lo[:,column]=levels[index];hi[:,column]=levels[index+1];pairs.append((lo,hi))
    check(len(saved['responses'])==3,'input response count')
    for name,(lo,hi),record in zip(('capital','productivity','labor_disutility'),pairs,saved['responses']):
        check(record['input']==name,'response pair order')
        check(np.all(lo[:,0]>=.5*teacher.parameters['kss']) and np.all(hi[:,0]<=1.8*teacher.parameters['kss']),'response support')
        a=(lo-norm['input_mean'])/norm['input_std'];b=(hi-norm['input_mean'])/norm['input_std']
        original=forward(b,net)[0]-forward(a,net)[0]
        edited=forward(b,net,2,selected)[0]-forward(a,net,2,selected)[0]
        rms=float(np.sqrt(np.mean(original**2)))
        compare_metrics(dict(original_rms_train_sd=rms,retained_response_norm_ratio=float(np.sqrt(np.mean(edited**2)))/max(rms,1e-30),
            response_distortion_ratio=float(np.sqrt(np.mean((edited-original)**2)))/max(rms,1e-30),original_mean_response=original.mean(0),edited_mean_response=edited.mean(0)),record,'label response '+name)


def design(x,quadratic=False):
    columns=[np.ones(len(x)),*x.T]
    if quadratic:columns.extend(x[:,i]*x[:,j] for i in range(3) for j in range(i,3))
    return np.column_stack(columns)


def main():
    started=time.perf_counter()
    protocol=json.loads((ROOT/'richer_protocol.json').read_text());result=json.loads((ROOT/'richer_results.json').read_text())
    report=json.loads((ROOT/'richer_teacher.json').read_text());teacher=load_teacher()
    check(result['protocol']==protocol,'saved protocol differs')
    provenance=result['provenance'];check(sha('richer_protocol.json')==provenance['protocol_sha256'],'protocol provenance')
    for group in ('source_sha256','teacher_sha256'):
        for path,digest in provenance[group].items():check(sha(path)==digest,'provenance '+path)
    fresh=verify_teacher(teacher,report);data,norm=partitions(protocol,teacher,provenance)
    check(provenance['raw_states_disjoint'] is True,'raw state disjointness claim')
    # Compare numerical loss derivatives to the actual training gradient implementation.
    from richer_experiment import loss_grad
    rng=np.random.default_rng(714020);width=5
    small=dict(W1=rng.normal(size=(width,3)),b1=rng.normal(size=width),W2=rng.normal(size=(width,width)),b2=rng.normal(size=width),W3=rng.normal(size=(width,3)),b3=rng.normal(size=3))
    x=rng.normal(size=(17,3));y=rng.normal(size=(17,3));loss,gradient=loss_grad(x,y,small)
    close(loss,np.mean((forward(x,small)[0]-y)**2),'independent training loss')
    gradient_error=0.
    for key,index in [('W1',(1,2)),('b1',(2,)),('W2',(2,1)),('b2',(1,)),('W3',(2,1)),('b3',(1,))]:
        original=small[key][index];small[key][index]=original+1e-6;plus=np.mean((forward(x,small)[0]-y)**2)
        small[key][index]=original-1e-6;minus=np.mean((forward(x,small)[0]-y)**2);small[key][index]=original
        gradient_error=max(gradient_error,abs((plus-minus)/2e-6-gradient[key][index]))
    check(gradient_error<1e-7,'training gradient finite difference')
    tr=data['TRAIN'];te=data['TEST'];va=data['VALIDATION'];it=data['INTERVENTION_TEST']
    for name,quadratic in [('affine',False),('quadratic',True)]:
        coef=np.linalg.lstsq(design(tr['x'],quadratic),tr['yn'],rcond=None)[0]
        close(coef,result['baselines'][name]['coefficients'],'baseline coefficients '+name)
        compare_metrics(fit(design(te['x'],quadratic)@coef,te['yn']),result['baselines'][name],'baseline TEST '+name)
    expected={(w,s) for w in protocol['architecture']['widths'] for s in protocol['architecture']['seeds']}
    check(len(result['models'])==len(expected),'model count')
    seen=set();k=np.tile(np.linspace(.5,1.8,301)*teacher.parameters['kss'],21);state=np.repeat(np.arange(21),301)
    dense_raw=np.column_stack((k,teacher.state_features[state]));pol=teacher.policy(k,state)
    dense_y=(np.column_stack([pol[key] for key in OUTPUTS])-norm['output_mean'])/norm['output_std']
    dense_x=(dense_raw-norm['input_mean'])/norm['input_std']
    for model in result['models']:
        width=model['width'];seed=model['seed'];seen.add((width,seed));net={key:np.asarray(model['network'][key],dtype=float) for key in KEYS}
        shapes=dict(W1=(width,3),b1=(width,),W2=(width,width),b2=(width,),W3=(width,3),b3=(3,))
        for key,array in net.items():check(array.shape==shapes[key] and np.isfinite(array).all(),'network shape/finite '+key)
        check(sum(a.size for a in net.values())==width*width+8*width+3==model['training']['parameter_count'],'parameter count')
        compare_metrics(fit(forward(te['x'],net)[0],te['yn']),model['fit']['TEST'],'model TEST')
        compare_metrics(fit(forward(dense_x,net)[0],dense_y),model['fit']['dense_21_state_grid'],'model dense fit')
        rng=np.random.default_rng(seed)
        initial=dict(W1=rng.normal(0,np.sqrt(2/3),(width,3)),b1=np.zeros(width),W2=rng.normal(0,np.sqrt(2/width),(width,width)),b2=np.zeros(width),W3=rng.normal(0,np.sqrt(1/width),(width,3)),b3=np.zeros(3))
        compare_metrics(fit(forward(te['x'],initial)[0],te['yn']),model['fit']['INITIAL_TEST'],'random initial TEST')
        meta=model['training'];curve=meta['curve'];scheduled=list(range(0,protocol['training']['steps']+1,protocol['training']['validation_frequency']))
        check([c['step'] for c in curve]==scheduled,'complete validation schedule')
        passing=[c for c in curve if max(c['normalized_rmse'])<=protocol['training']['gate_each_output']]
        chosen=passing[0] if passing else min(curve,key=lambda c:max(c['normalized_rmse']))
        check(bool(passing)==meta['eligible'] and chosen['step']==meta['selected_step'],'checkpoint selection rule')
        close(fit(forward(va['x'],net)[0],va['yn'])['normalized_rmse'],chosen['normalized_rmse'],'selected checkpoint validation')
        interventions(net,it['x'],model['interventions']);labels(net,data,norm,teacher,model['labels'])
        def neural_policy(k,s):
            raw=np.column_stack((k,teacher.state_features[s]));physical=forward((raw-norm['input_mean'])/norm['input_std'],net)[0]*norm['output_std']+norm['output_mean']
            return dict(zip(OUTPUTS,physical.T))
        saved=model['economic_residuals']
        audit_k=te['raw'][:1024,0];audit_s=te['state'][:1024]
        audit=neural_policy(audit_k,audit_s);c,n,i=[audit[key] for key in OUTPUTS]
        kp=(1-teacher.parameters['delta'])*audit_k+i
        q=1+teacher.parameters['kappa']*(i/audit_k-teacher.parameters['delta'])
        counts=dict(states=len(audit_k),nonpositive_consumption=int(np.sum(c<=0)),nonpositive_labor=int(np.sum(n<=0)),
            nonpositive_q=int(np.sum(q<=0)),next_capital_outside_grid=int(np.sum((kp<teacher.k_grid[0])|(kp>teacher.k_grid[-1]))))
        compare_metrics(counts,saved,'neural feasibility counts')
        check(saved['all_residuals_available'] is True,'unexpected unavailable saved neural residuals')
        if saved['all_residuals_available']:

            compare_metrics(foc_residuals(teacher,te['raw'][:1024,0],te['state'][:1024],neural_policy),saved,'network economic residuals')
    check(seen==expected,'width/seed membership')
    print(json.dumps(dict(status='PASS',models=len(seen),fresh_teacher_states=1536,fresh_teacher_residuals=fresh,training_gradient_max_absolute_error=gradient_error,
        verified='saved hashes; independent joint Markov reconstruction; fresh teacher equations; stored-coarse refinement; split digests/disjointness; TRAIN normalization; all TEST/dense/initial fits; checkpoint rule; all first/final-layer interventions; selected labels and response pairs; available neural economic residuals',
        limits='No optimizer replay, retraining, or global Bellman optimality proof. Teacher interpolation/static policy is reused; equations and neural evaluation are independently implemented.',seconds=time.perf_counter()-started),indent=2))


if __name__=='__main__':main()
