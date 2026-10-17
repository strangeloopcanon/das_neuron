"""Larger nonlinear DSGE, three-output neural policies and full-weight audits.

Import-safe. Run --pilot, inspect its measured cost, then --run.
"""
import os
for _name in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[_name] = '1'
import argparse
import hashlib
import json
import time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
KEYS = ('W1', 'b1', 'W2', 'b2', 'W3', 'b3')
OUTPUTS = ('c', 'labor', 'investment')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def arrays_digest(*arrays):
    h = hashlib.sha256()
    for a in arrays:
        a = np.ascontiguousarray(a)
        h.update(str(a.dtype).encode()); h.update(str(a.shape).encode()); h.update(a.tobytes())
    return h.hexdigest()


def clone(net):
    return {key: value.copy() for key, value in net.items()}


def export(net):
    return {key: value.tolist() for key, value in net.items()}


def initialize(width, seed):
    rng = np.random.default_rng(seed)
    return dict(W1=rng.normal(0, np.sqrt(2/3), (width, 3)), b1=np.zeros(width),
                W2=rng.normal(0, np.sqrt(2/width), (width, width)), b2=np.zeros(width),
                W3=rng.normal(0, np.sqrt(1/width), (width, 3)), b3=np.zeros(3))


def evaluate(x, net, layer=None, unit=None, fraction=1.):
    p1 = x @ net['W1'].T+net['b1']
    h1 = np.maximum(p1, 0.)
    if layer == 1:
        h1[:, unit] *= 1-fraction
    p2 = h1 @ net['W2'].T+net['b2']
    h2 = np.maximum(p2, 0.)
    if layer == 2:
        h2[:, unit] *= 1-fraction
    return h2 @ net['W3']+net['b3'], (p1, h1, p2, h2)


def loss_grad(x, y, net):
    prediction, (p1, h1, p2, h2) = evaluate(x, net)
    error = prediction-y
    d3 = 2*error/error.size
    d2 = (d3 @ net['W3'].T)*(p2 > 0.)
    d1 = (d2 @ net['W2'])*(p1 > 0.)
    return float(np.mean(error**2)), dict(W1=d1.T@x, b1=d1.sum(0), W2=d2.T@h1,
        b2=d2.sum(0), W3=h2.T@d3, b3=d3.sum(0))


def gradient_check():
    net = initialize(6, 123)
    rng = np.random.default_rng(789)
    x, y = rng.normal(size=(19,3)), rng.normal(size=(19,3))
    _, grads = loss_grad(x, y, net)
    errors = []
    for key, index in [('W1',(2,1)),('b1',(3,)),('W2',(2,3)),('b2',(2,)),('W3',(2,1)),('b3',(1,))]:
        a, b = clone(net), clone(net)
        a[key][index] += 1e-6; b[key][index] -= 1e-6
        numerical = (loss_grad(x,y,a)[0]-loss_grad(x,y,b)[0])/2e-6
        errors.append(abs(numerical-grads[key][index]))
    assert max(errors) < 1e-7, errors
    return max(errors)


def context():
    from richer_teacher import load_teacher, sample_states
    protocol = json.loads((ROOT/'richer_protocol.json').read_text())
    teacher = load_teacher()
    data, membership, seen = {}, {}, set()
    for name, spec in protocol['data'].items():
        if not isinstance(spec, dict):
            continue
        k, state = sample_states(spec['n'], spec['seed'])
        raw = np.column_stack((k, teacher.state_features[state]))
        pol = teacher.policy(k, state)
        y = np.column_stack([pol[key] for key in OUTPUTS])
        keys = {(float(kk).hex(), int(ss)) for kk,ss in zip(k,state)}
        assert len(keys) == len(k) and not keys & seen
        seen |= keys
        data[name] = dict(raw=raw, state=state, y=y)
        membership[name] = dict(**spec, sha256=arrays_digest(k,state.astype(np.int64),y),
            shock_state_counts=np.bincount(state,minlength=len(teacher.P)).tolist())
    tr = data['TRAIN']
    norm = dict(input_mean=tr['raw'].mean(0), input_std=tr['raw'].std(0),
                output_mean=tr['y'].mean(0), output_std=tr['y'].std(0))
    for d in data.values():
        d['x'] = (d['raw']-norm['input_mean'])/norm['input_std']
        d['yn'] = (d['y']-norm['output_mean'])/norm['output_std']
    return dict(protocol=protocol,teacher=teacher,data=data,norm=norm,membership=membership)


def score(pred, truth, sd):
    return dict(normalized_rmse=np.sqrt(np.mean(((pred-truth)/sd)**2,axis=0)).tolist(),
                normalized_max_error=np.max(abs((pred-truth)/sd),axis=0).tolist())


def train(width,seed,ctx,steps):
    started = time.perf_counter()
    net = initialize(width,seed)
    m = {key:np.zeros_like(v) for key,v in net.items()}; v = clone(m)
    rng = np.random.default_rng(seed+1000000)
    protocol = ctx['protocol']['training']; tr,va = ctx['data']['TRAIN'],ctx['data']['VALIDATION']
    best,selected,best_error,best_step,selected_step = None,None,np.inf,0,None
    curve = []
    for step in range(steps+1):
        if step % protocol['validation_frequency'] == 0:
            errors = np.sqrt(np.mean((evaluate(va['x'],net)[0]-va['yn'])**2,axis=0))
            curve.append(dict(step=step,normalized_rmse=errors.tolist()))
            if errors.max() < best_error:
                best_error,best_step,best = float(errors.max()),step,clone(net)
            if selected is None and errors.max() <= protocol['gate_each_output']:
                selected, selected_step = clone(net),step
        if step == steps:
            break
        batch = rng.integers(0,len(tr['y']),protocol['batch_size'])
        _,grad = loss_grad(tr['x'][batch],tr['yn'][batch],net)
        t = step+1
        lr = next(item['value'] for item in protocol['learning_rates'] if t <= item['through'])
        for key in KEYS:
            m[key] = .9*m[key]+.1*grad[key]
            v[key] = .999*v[key]+.001*grad[key]**2
            net[key] -= lr*(m[key]/(1-.9**t))/(np.sqrt(v[key]/(1-.999**t))+1e-8)
    chosen = selected if selected is not None else best
    return chosen,dict(eligible=selected is not None,selected_step=selected_step if selected is not None else best_step,
        selection='first VALIDATION checkpoint meeting every-output gate' if selected is not None else 'best worst-output VALIDATION checkpoint, INELIGIBLE',
        seconds=time.perf_counter()-started,steps=steps,curve=curve,
        parameter_count=sum(a.size for a in chosen.values()))


def response_metrics(actual, predicted):
    rms = float(np.sqrt(np.mean(actual**2)))
    active = abs(actual) > 1e-10
    return dict(response_rms_train_sd=rms,
        prediction_error_rms_train_sd=float(np.sqrt(np.mean((actual-predicted)**2))),
        normalized_response_rmse=float(np.sqrt(np.mean((actual-predicted)**2)))/max(rms,1e-30),
        sign_accuracy=float(np.mean(np.sign(actual[active])==np.sign(predicted[active]))) if active.any() else None,
        active_output_effects=int(active.sum()))


def intervention_audit(net,x):
    baseline,(p1,h1,p2,h2) = evaluate(x,net)
    width = h1.shape[1]
    # n x first-layer-unit x output, using ALL signed weights.
    jacobian = np.einsum('lj,nl,lo->njo',net['W2'],p2>0,net['W3'])
    reports = []
    for fraction in (.01,.1,1.):
        changes,predictions,switches = [],[],0
        for j in range(width):
            edited,hidden = evaluate(x,net,1,j,fraction)
            changes.append(edited-baseline)
            predictions.append(jacobian[:,j,:]*(-fraction*h1[:,j,None]))
            switches += int(np.sum(np.any((hidden[2]>0)!=(p2>0),axis=1)))
        actual,predicted = np.stack(changes,1),np.stack(predictions,1)
        reports.append(dict(fraction=fraction,**response_metrics(actual,predicted),
            downstream_gate_switch_fraction=switches/(len(x)*width),
            per_output=[response_metrics(actual[:,:,o],predicted[:,:,o]) for o in range(3)]))
    exact_errors=[]
    for j in range(width):
        actual = evaluate(x,net,2,j)[0]-baseline
        predicted = -h2[:,j,None]*net['W3'][j]
        exact_errors.append(float(np.max(abs(actual-predicted))))
    return dict(first_layer=reports,final_layer_max_absolute_prediction_error=max(exact_errors),
                information='All weights and baseline gates supplied; actual edited forward propagation is the reference.')


def input_response_pairs(ctx, raw):
    teacher = ctx['teacher']; features = teacher.state_features
    # Equal weighting of backgrounds; contrasts stay within the declared domain.
    pairs=[]
    low,high=raw.copy(),raw.copy()
    low[:,0]=np.maximum(.5*teacher.parameters['kss'],.9*raw[:,0])
    high[:,0]=np.minimum(1.8*teacher.parameters['kss'],1.1*raw[:,0])
    pairs.append((low,high))
    for column in (0,1):
        levels=np.unique(features[:,column]); mid=int(np.argmin(abs(levels)))
        assert mid+1<len(levels)
        low,high=raw.copy(),raw.copy()
        low[:,column+1]=levels[mid];high[:,column+1]=levels[mid+1]
        pairs.append((low,high))
    norm=ctx['norm']
    return [((a-norm['input_mean'])/norm['input_std'],(b-norm['input_mean'])/norm['input_std']) for a,b in pairs]


def label_audit(net,ctx):
    disc=ctx['data']['DISCOVERY']; h=evaluate(disc['x'],net)[1][-1]
    hc=h-h.mean(0); xc=disc['x']-disc['x'].mean(0)
    denominators=np.sqrt((hc**2).sum(0)[:,None]*(xc**2).sum(0)[None,:])
    correlations=np.divide(hc.T@xc,denominators,out=np.zeros((h.shape[1],3)),where=denominators>1e-14)
    j=int(np.argmax(abs(correlations[:,1])))
    background=ctx['data']['INTERVENTION_TEST']['raw']
    records=[]
    for label,(a,b) in zip(('capital','productivity','labor_disutility'),input_response_pairs(ctx,background)):
        base=evaluate(b,net)[0]-evaluate(a,net)[0]
        edited=evaluate(b,net,2,j)[0]-evaluate(a,net,2,j)[0]
        norm=np.sqrt(np.mean(base**2))
        records.append(dict(input=label,original_rms_train_sd=float(norm),
            retained_response_norm_ratio=float(np.sqrt(np.mean(edited**2)))/max(norm,1e-30),
            response_distortion_ratio=float(np.sqrt(np.mean((edited-base)**2)))/max(norm,1e-30),
            original_mean_response=base.mean(0).tolist(),edited_mean_response=edited.mean(0).tolist()))
    return dict(unit=j,selection='highest absolute DISCOVERY productivity correlation',
        correlations=correlations.tolist(),selected_correlations=correlations[j].tolist(),
        readout_train_output_sd=net['W3'][j].tolist(),responses=records)


def economic_residuals(net,ctx,raw,state):
    teacher=ctx['teacher'];p=teacher.parameters;norm=ctx['norm']
    def physical(k,s):
        inputs=np.column_stack((k,teacher.state_features[s]))
        return evaluate((inputs-norm['input_mean'])/norm['input_std'],net)[0]*norm['output_std']+norm['output_mean']
    y=physical(raw[:,0],state); c,n,i=y.T;k=raw[:,0]
    positive=(c>0)&(n>0)
    report=dict(states=len(k),nonpositive_consumption=int(np.sum(c<=0)),nonpositive_labor=int(np.sum(n<=0)))
    if not positive.all():
        report['all_residuals_available']=False;return report
    output=np.exp(raw[:,1])*k**p['alpha']*n**(1-p['alpha'])
    cost=p['kappa']/2*k*(i/k-p['delta'])**2
    resource=c+i+cost-output
    labor_lhs=p['chi']*np.exp(raw[:,2])*n**p['phi']
    labor_rhs=c**(-p['sigma'])*(1-p['alpha'])*output/n
    q=1+p['kappa']*(i/k-p['delta']);kp=(1-p['delta'])*k+i
    valid=(kp>=teacher.k_grid[0])&(kp<=teacher.k_grid[-1])&(q>0)
    report.update(resource_relative_rms=float(np.sqrt(np.mean((resource/output)**2))),
        resource_relative_max=float(np.max(abs(resource/output))),
        labor_foc_relative_rms=float(np.sqrt(np.mean((labor_lhs/labor_rhs-1)**2))),
        labor_foc_relative_max=float(np.max(abs(labor_lhs/labor_rhs-1))),
        nonpositive_q=int(np.sum(q<=0)),next_capital_outside_grid=int(np.sum(~((kp>=teacher.k_grid[0])&(kp<=teacher.k_grid[-1])))))
    if not valid.all():
        report['all_residuals_available']=False;return report
    rhs=np.zeros(len(k));future_valid=True
    for s in range(len(teacher.P)):
        cf,nf,iv=physical(kp,np.full(len(kp),s,dtype=int)).T
        if np.any(cf<=0) or np.any(nf<=0):
            future_valid=False;break
        z=np.exp(teacher.state_features[s,0])
        mpk=p['alpha']*z*kp**(p['alpha']-1)*nf**(1-p['alpha'])
        qf=1+p['kappa']*(iv/kp-p['delta'])
        costk=p['kappa']/2*(p['delta']**2-(iv/kp)**2)
        rhs+=teacher.P[state,s]*cf**(-p['sigma'])*(mpk+(1-p['delta'])*qf-costk)
    report['all_residuals_available']=future_valid
    if future_valid:
        rel=p['beta']*rhs/(c**(-p['sigma'])*q)-1
        report.update(euler_relative_rms=float(np.sqrt(np.mean(rel**2))),euler_relative_max=float(np.max(abs(rel))))
    return report


def baseline_design(x,quadratic=False):
    columns=[np.ones(len(x)),*x.T]
    if quadratic:
        columns.extend(x[:,i]*x[:,j] for i in range(3) for j in range(i,3))
    return np.column_stack(columns)


def run(ctx):
    started=time.perf_counter();models=[]
    tr=ctx['data']['TRAIN'];te=ctx['data']['TEST'];it=ctx['data']['INTERVENTION_TEST']
    baselines={}
    for name,quadratic in [('affine',False),('quadratic',True)]:
        coef=np.linalg.lstsq(baseline_design(tr['x'],quadratic),tr['yn'],rcond=None)[0]
        baselines[name]=dict(coefficients=coef.tolist(),**score(baseline_design(te['x'],quadratic)@coef,te['yn'],np.ones(3)))
    for width in ctx['protocol']['architecture']['widths']:
        for seed in ctx['protocol']['architecture']['seeds']:
            net,meta=train(width,seed,ctx,ctx['protocol']['training']['steps'])
            pred=evaluate(te['x'],net)[0]
            initial=initialize(width,seed)
            # Dense state-stratified checks expose rare-state fit hidden by averages.
            k=np.tile(np.linspace(.5,1.8,301)*ctx['teacher'].parameters['kss'],len(ctx['teacher'].P))
            state=np.repeat(np.arange(len(ctx['teacher'].P)),301)
            raw=np.column_stack((k,ctx['teacher'].state_features[state]))
            pol=ctx['teacher'].policy(k,state)
            y=np.column_stack([pol[key] for key in OUTPUTS])
            dense_pred=evaluate((raw-ctx['norm']['input_mean'])/ctx['norm']['input_std'],net)[0]
            truth=(y-ctx['norm']['output_mean'])/ctx['norm']['output_std']
            fit=dict(TEST=score(pred,te['yn'],np.ones(3)),dense_21_state_grid=score(dense_pred,truth,np.ones(3)),
                INITIAL_TEST=score(evaluate(te['x'],initial)[0],te['yn'],np.ones(3)))
            model=dict(width=width,seed=seed,network=export(net),training=meta,fit=fit,
                interventions=intervention_audit(net,it['x']),labels=label_audit(net,ctx),
                economic_residuals=economic_residuals(net,ctx,te['raw'][:1024],te['state'][:1024]))
            models.append(model)
            print(json.dumps(dict(width=width,seed=seed,eligible=meta['eligible'],parameters=meta['parameter_count'],
                test_rmse=fit['TEST']['normalized_rmse'],seconds=meta['seconds'])),flush=True)
    provenance=dict(protocol_sha256=digest(ROOT/'richer_protocol.json'),
        source_sha256={f:digest(ROOT/f) for f in ('richer_experiment.py','richer_teacher.py')},
        teacher_sha256={f:digest(ROOT/f) for f in ('richer_teacher.npz','richer_teacher.json')},
        partitions=ctx['membership'],normalization={k:v.tolist() for k,v in ctx['norm'].items()},
        raw_states_disjoint=True,numpy_version=np.__version__)
    result=dict(protocol=ctx['protocol'],provenance=provenance,baselines=baselines,models=models,
        seconds=time.perf_counter()-started,scope='A larger solved DSGE and learned three-output circuits; no general incomprehensibility theorem.')
    (ROOT/'richer_results.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--pilot',action='store_true');parser.add_argument('--run',action='store_true')
    args=parser.parse_args();assert args.pilot != args.run,'Choose --pilot or --run'
    gradient_error=gradient_check();ctx=context()
    if args.pilot:
        p=ctx['protocol']['training']['pilot'];_,meta=train(p['width'],p['seed'],ctx,p['steps'])
        result=dict(gradient_max_error=gradient_error,pilot=meta,
            estimated_sweep_seconds=meta['seconds']*ctx['protocol']['training']['steps']/p['steps']*9,
            scope='Pilot timing and gradient verification, not an experiment result.')
        (ROOT/'richer_pilot.json').write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps({k:v for k,v in result.items() if k!='pilot'},indent=2));print('Measured pilot seconds:',meta['seconds'])
    else:
        assert (ROOT/'richer_pilot.json').exists(),'Run bounded pilot first'
        run(ctx)


if __name__=='__main__':
    main()
