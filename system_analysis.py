"""Read-only analysis of frozen richer networks; no training or teacher solve.

Writes system_analysis.json and system_analysis_report.md. All labels describe
DISCOVERY associations, while finite changes use the complete saved network.
"""
import os
for _key in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[_key] = '1'
import hashlib
import json
from pathlib import Path
import numpy as np
from richer_experiment import context, evaluate

ROOT = Path(__file__).resolve().parent
INPUTS = ['capital', 'log productivity', 'labor-disutility shock']
OUTPUTS = ['consumption', 'labor', 'investment']
STATES = dict(baseline=10, productivity=19, cost=11, both=20)
CONTRASTS = dict(productivity='productivity', cost='cost', joint='both')


def sha(path):
    return hashlib.sha256((ROOT / path).read_bytes()).hexdigest()


def plain(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value))


def neuron_labels(hidden, x, layer):
    h = hidden - hidden.mean(0)
    centered = x - x.mean(0)
    denom = np.sqrt((h*h).sum(0)[:, None]*(centered*centered).sum(0)[None, :])
    correlations = np.divide(h.T @ centered, denom, out=np.zeros((h.shape[1], 3)), where=denom > 1e-14)
    labels = []
    for unit, corr in enumerate(correlations):
        active = float(np.mean(hidden[:, unit] > 0))
        variable = bool(np.std(hidden[:, unit]) > 1e-14)
        strongest = int(np.argmax(abs(corr))) if variable else None
        label = (f'{INPUTS[strongest]} association (r={corr[strongest]:+.3f})' if variable else
                 'inactive on DISCOVERY' if active == 0 else 'constant on DISCOVERY; correlation undefined')
        definition = (f'h1[{unit}]=max(0, b1[{unit}]+sum_i W1[{unit},i]*x[i])' if layer == 1 else
                      f'h2[{unit}]=max(0, b2[{unit}]+sum_j W2[{unit},j]*h1[j])')
        labels.append(dict(unit=unit, correlations=corr, correlation_defined=variable,
                           activation_frequency=active, strongest_input_index=strongest,
                           label=label, definition=definition,
                           incoming_weight_reference=f'network.W{layer}[{unit}]',
                           bias_reference=f'network.b{layer}[{unit}]'))
    return labels


def route_counts(net, h1, h2):
    """Structural nonzero, open-gate paths; counts do not measure influence."""
    a1, a2 = h1 > 0, h2 > 0
    edges = (net['W2'] != 0)[None, :, :] & a2[:, :, None] & a1[:, None, :]
    first_input_degree = np.count_nonzero(net['W1'], axis=1)
    paths_to_h2 = np.sum(edges*first_input_degree[None, None, :], axis=2)
    return dict(open_h1_h2_edges=edges.sum((1, 2)),
                input_to_output_paths=paths_to_h2 @ (net['W3'] != 0).astype(int))


def metrics(contributions, delta):
    positive = np.maximum(contributions, 0).sum(1)
    negative = np.minimum(contributions, 0).sum(1)
    gross = positive-negative
    ratio = np.divide(gross, abs(delta), out=np.full(delta.shape, np.nan), where=abs(delta) > 1e-14)
    fraction = np.divide(gross-abs(delta), gross, out=np.zeros_like(gross), where=gross > 1e-14)
    top = np.sort(abs(contributions), axis=1)[:, -5:, :].sum(1)
    concentration = np.divide(top, gross, out=np.zeros_like(gross), where=gross > 1e-14)
    return dict(positive_sum=positive, negative_sum=negative, gross_sum=gross,
                gross_over_abs_net=ratio, cancellation_fraction=fraction,
                top5_gross_share=concentration)


def top_indices(contribution):
    return np.argsort(-abs(contribution), kind='stable')[:5].tolist()


def run():
    frozen = json.loads((ROOT/'richer_results.json').read_text())
    ctx = context()
    norm, teacher = ctx['norm'], ctx['teacher']
    assert all(np.allclose(norm[key], frozen['provenance']['normalization'][key], rtol=0, atol=0)
               for key in norm)
    assert ctx['membership'] == frozen['provenance']['partitions']
    ratios = np.linspace(.5, 1.8, 301)
    capital = ratios*teacher.parameters['kss']
    state_data, teacher_y = {}, {}
    for name, state in STATES.items():
        raw = np.column_stack((capital, np.tile(teacher.state_features[state], (301, 1))))
        state_data[name] = dict(raw=raw, x=(raw-norm['input_mean'])/norm['input_std'])
        pol = teacher.policy(capital, np.full(301, state))
        teacher_y[name] = np.column_stack([pol[k] for k in ('c', 'labor', 'investment')])
    teacher_interaction = teacher_y['both']-teacher_y['productivity']-teacher_y['cost']+teacher_y['baseline']
    example_indices = [int(np.argmin(abs(ratios-target))) for target in (.6, 1., 1.6)]
    result = dict(schema_version=1,
        scope='Frozen-network descriptive analysis; finite shock counterfactuals, not realized time paths or randomized economic treatments.',
        inputs=INPUTS, outputs=OUTPUTS, default_model_id='H16-seed113',
        default_selection='Width 16, seed 113 fixed before this analysis; not chosen by fit, cancellation, interaction, or labels.',
        label_scope='Every hidden unit is labeled by strongest absolute DISCOVERY Pearson association. This is a descriptive tag, not an exclusive semantic role or causal explanation. Zero variance has undefined correlation and stored numeric zeros.',
        attribution_definition='For output o, contribution[j,o] = TRAIN_output_sd[o] * W3[j,o] * (h2_target[j]-h2_base[j]); sum_j contributions equals the finite output change. Bias cancels.',
        route_count_definition='Count nonzero input→H1→H2→output paths whose two ReLU gates are open in each snapshot. These structural counts are not causal effect sizes. Final contribution routes identify H2 readout units, not full individual input paths.',
        interaction_definition='I = y_both - y_productivity - y_cost + y_baseline, at fixed capital. Network I minus teacher I is approximation error in interaction; network I alone does not establish economic interaction.',
        qualifications=['All shock corners are actual states in the 21-state numerical teacher.',
            'Labor cost is shorthand for the labor-disutility shock eta: it increases the utility cost of work, not a wage or accounting cost.',
            'The numerical teacher solves checked first-order equations; global Bellman optimality was not established.',
            'Associations alone do not specify nonlinear computation. The full network, normalization and forward equations do.',
            'No human comprehension test, hardness theorem, scaling law, or inference of polysemanticity is supplied.'],
        provenance=dict(source_sha256={f:sha(f) for f in ('system_analysis.py', 'richer_experiment.py', 'richer_teacher.py', 'richer_protocol.json', 'richer_results.json', 'richer_teacher.npz', 'richer_teacher.json')},
            partitions=ctx['membership'], normalization=norm,
            labels_membership='DISCOVERY only; 2048 states, seed 112003, no label fitting or selection on TEST.',
            analysis_grid_membership='New deterministic diagnostic grid: 301 capital points per four shock corners. This is descriptive analysis of frozen models, not a new held-out model-selection test.',
            execution='Local CPU forward passes only. No training, refitting, teacher solve, new paid work or edits to source evidence.'),
        states={name:dict(index=state,productivity_index=state//3,labor_disutility_index=state%3,
                         features=teacher.state_features[state]) for name,state in STATES.items()},
        capital_grid_over_kss=ratios, kss=teacher.parameters['kss'], example_indices=example_indices,
        example_selection='Prespecified requested capital ratios .6, 1.0, 1.6; nearest of 301 fixed grid points.',
        teacher=dict(physical_outputs=teacher_y, interaction=teacher_interaction), models=[])
    max_residual = 0.
    for saved in frozen['models']:
        net = {k:np.asarray(v) for k,v in saved['network'].items()}
        disc_h = evaluate(ctx['data']['DISCOVERY']['x'], net)[1]
        labels = dict(H1=neuron_labels(disc_h[1],ctx['data']['DISCOVERY']['x'],1),
                      H2=neuron_labels(disc_h[3],ctx['data']['DISCOVERY']['x'],2))
        snapshots = {}
        for name, d in state_data.items():
            out, hidden = evaluate(d['x'], net)
            snapshots[name] = dict(raw=d['raw'], x=d['x'], physical_outputs=out*norm['output_std']+norm['output_mean'],
                                   p1=hidden[0],h1=hidden[1],p2=hidden[2],h2=hidden[3],
                                   route_counts=route_counts(net,hidden[1],hidden[3]))
        base = snapshots['baseline']
        contrasts, grid = {}, {}
        for contrast,target_name in CONTRASTS.items():
            target = snapshots[target_name]
            contribution = ((target['h2']-base['h2'])[:,:,None]*net['W3'][None,:,:]*norm['output_std'])
            delta = target['physical_outputs']-base['physical_outputs']
            residual = float(np.max(abs(contribution.sum(1)-delta)))
            max_residual=max(max_residual,residual)
            m=metrics(contribution, delta)
            contrasts[contrast] = dict(target=target_name, delta=delta,
                teacher_delta=teacher_y[target_name]-teacher_y['baseline'], contribution=contribution, **m,
                h1_gate_changes=np.count_nonzero((target['p1']>0)!=(base['p1']>0),axis=1),
                h2_gate_changes=np.count_nonzero((target['p2']>0)!=(base['p2']>0),axis=1),
                attribution_max_absolute_residual=residual)
            grid[contrast]=dict(delta=delta, teacher_delta=teacher_y[target_name]-teacher_y['baseline'],
                **m,h1_gate_changes=contrasts[contrast]['h1_gate_changes'],
                h2_gate_changes=contrasts[contrast]['h2_gate_changes'],
                attribution_max_absolute_residual=residual)
        interaction = (snapshots['both']['physical_outputs']-snapshots['productivity']['physical_outputs']-
                       snapshots['cost']['physical_outputs']+base['physical_outputs'])
        interaction_contributions = (snapshots['both']['h2']-snapshots['productivity']['h2']-
                                    snapshots['cost']['h2']+base['h2'])[:,:,None]*net['W3'][None,:,:]*norm['output_std']
        residual = float(np.max(abs(interaction_contributions.sum(1)-interaction)))
        max_residual=max(max_residual,residual)
        grid['interaction']=dict(network=interaction,teacher=teacher_interaction,
                                 error=interaction-teacher_interaction,
                                 attribution_max_absolute_residual=residual)
        examples=[]
        for index in example_indices:
            ex=dict(grid_index=index,capital_over_kss=float(ratios[index]),capital=float(capital[index]),
                    snapshots={name:{key:(dict(open_h1_h2_edges=value['open_h1_h2_edges'][index],input_to_output_paths=value['input_to_output_paths'][index]) if key=='route_counts' else value[index]) for key,value in snap.items()} for name,snap in snapshots.items()},
                    contrasts={})
            for name,c in contrasts.items():
                ex['contrasts'][name]={key:(value[index] if isinstance(value,np.ndarray) else value) for key,value in c.items()}
                ex['contrasts'][name]['top5_units_by_output']=[top_indices(c['contribution'][index,:,o]) for o in range(3)]
            ex['interaction']=dict(network=interaction[index],teacher=teacher_interaction[index],
                                   error=(interaction-teacher_interaction)[index],contribution=interaction_contributions[index])
            examples.append(ex)
        overlap={}
        for name,c in contrasts.items():
            rows=[]
            for o in range(3):
                low=top_indices(c['contribution'][example_indices[0],:,o]);high=top_indices(c['contribution'][example_indices[-1],:,o])
                common=len(set(low)&set(high))
                rows.append(dict(output=OUTPUTS[o],low_units=low,high_units=high,
                                 intersection_count=common,overlap_fraction=common/5,
                                 jaccard=common/(10-common)))
            overlap[name]=rows
        # Exploratory witness is clearly separated from the fixed teaching cases.
        candidates=[]
        for contrast,c in contrasts.items():
            scores=c['gross_over_abs_net'].copy()
            scores[abs(c['delta']) < 1e-6*norm['output_std']]=-np.inf
            index,o=np.unravel_index(np.nanargmax(scores),scores.shape)
            candidates.append((scores[index,o],contrast,index,o))
        _,cn,ix,o=max(candidates)
        c=contrasts[cn]
        witness=dict(selection='Exploratory: maximum gross/absolute-net ratio across 301 capital points, 3 contrasts and 3 outputs; excludes |net| < 1e-6 TRAIN output SD.',
            contrast=cn,output=OUTPUTS[o],output_index=int(o),grid_index=int(ix),capital_over_kss=float(ratios[ix]),
            delta=float(c['delta'][ix,o]),teacher_delta=float(c['teacher_delta'][ix,o]),
            positive_sum=float(c['positive_sum'][ix,o]),negative_sum=float(c['negative_sum'][ix,o]),
            gross_over_abs_net=float(c['gross_over_abs_net'][ix,o]),
            cancellation_fraction=float(c['cancellation_fraction'][ix,o]),
            h1_gate_changes=int(c['h1_gate_changes'][ix]),h2_gate_changes=int(c['h2_gate_changes'][ix]))
        result['models'].append(dict(id=f"H{saved['width']}-seed{saved['seed']}",width=saved['width'],seed=saved['seed'],
            network=saved['network'],training=saved['training'],fit=saved['fit'],labels=labels,examples=examples,
            grid=grid,top5_route_overlap_low_high=overlap,exploratory_witness=witness))
    assert max_residual < 1e-12, max_residual
    result['verification']=dict(maximum_absolute_attribution_residual=max_residual,
        attribution_tolerance=1e-12,model_count=len(result['models']),
        labeled_hidden_unit_count=sum(2*m['width'] for m in result['models']),
        all_frozen_weights_preserved=True)
    # JSON has no NaN; undefined ratios are explicitly null.
    encoded=json.dumps(result,default=plain,separators=(',',':'),allow_nan=True)
    encoded=encoded.replace('NaN','null')
    (ROOT/'system_analysis.json').write_text(encoded+'\n')
    write_report(result)
    default=result['models'][0]
    print(json.dumps(dict(verification=result['verification'],default_witness=default['exploratory_witness'],
                          default_examples=[dict(k=e['capital_over_kss'],interaction=e['interaction'],
                            joint_delta=e['contrasts']['joint']['delta'],positive=e['contrasts']['joint']['positive_sum'],
                            negative=e['contrasts']['joint']['negative_sum']) for e in default['examples']]),default=plain,indent=2))
    return result


def write_report(result):
    default=next(m for m in result['models'] if m['id']==result['default_model_id'])
    lines=['# Joint computation in the frozen economic networks','',
        'Every hidden unit has a qualified association tag and an exact weighted-input definition. All saved weights are retained. The tags do not specify the computation by themselves. Exact forward evaluation does.','',
        'No training, refitting, teacher solution, or publication occurred. The default is H16 seed 113, fixed before these diagnostic calculations.','',
        'The four shock corners have indices 10 (center), 19 (high productivity), 11 (high labor disutility), and 20 (both). Capital is held fixed. Labor disutility increases the utility cost of work; it is not a wage.','',
        'The analysis evaluates each of nine networks at 301 capital points in [0.5, 1.8] times steady-state capital. Three teaching examples use the nearest points to 0.6, 1.0, and 1.6. These are new descriptive diagnostics, not a held-out model-selection test.','',
        '## Exact accounting','',
        '`delta output[o] = TRAIN_sd[o] * sum_j W3[j,o] * (h2_target[j] - h2_base[j])`.','',
        f"The largest absolute residual across every model, grid point, contrast, and interaction is {result['verification']['maximum_absolute_attribution_residual']:.3g}. Contributions can have opposite signs. Their gross sum and their net sum answer different questions.",'',
        'Open path counts describe nonzero connections with open gates. They do not measure causal importance. A final-hidden contribution identifies a readout unit and aggregates its upstream paths.','',
        '## Fixed default teaching cases','',
        '| Capital/kss | Output | Joint net | Positive terms | Negative terms | NN interaction | Teacher interaction | Interaction error |','|---:|---|---:|---:|---:|---:|---:|---:|']
    for ex in default['examples']:
        c=ex['contrasts']['joint'];i=ex['interaction']
        for o,name in enumerate(OUTPUTS):
            lines.append(f"| {ex['capital_over_kss']:.4f} | {name} | {c['delta'][o]:+.6g} | {c['positive_sum'][o]:+.6g} | {c['negative_sum'][o]:+.6g} | {i['network'][o]:+.6g} | {i['teacher'][o]:+.6g} | {i['error'][o]:+.6g} |")
    actual=np.asarray(result['teacher']['interaction'])
    estimated=np.asarray(default['grid']['interaction']['network'])
    relative=np.sqrt(np.mean((estimated-actual)**2,axis=0))/np.sqrt(np.mean(actual**2,axis=0))
    agreement=np.mean(np.sign(estimated)==np.sign(actual),axis=0)
    lines+=['','**Interaction fidelity fails:** exact neural accounting succeeds, but these networks do not accurately recover the teacher\'s small cross-shock interaction. Across the 301 capital points, default-model interaction error RMS / economic interaction RMS is '+', '.join(f'{name} {relative[o]:.2f}x' for o,name in enumerate(OUTPUTS))+'. Sign agreement is '+', '.join(f'{name} {100*agreement[o]:.1f}%' for o,name in enumerate(OUTPUTS))+'. Every saved model has consumption and labor interaction error RMS greater than the corresponding economic interaction RMS. These are descriptive diagnostics of already selected checkpoints.','']
    lines+=['','Interaction means `both - productivity - cost + baseline`. The teacher can have genuine interaction. The network can also have interaction approximation error. Report both; do not call all network interaction an economic effect.','',
        '## Route changes across capital','',
        'The top five routes rank final-hidden units by absolute contribution. This table compares the fixed low and high capital examples. Ties use ascending unit index.','',
        '| Shock | Consumption overlap | Labor overlap | Investment overlap |','|---|---:|---:|---:|']
    for name,rows in default['top5_route_overlap_low_high'].items():
        lines.append('| '+name+' | '+' | '.join(f"{r['intersection_count']}/5" for r in rows)+' |')
    w=default['exploratory_witness']
    lines+=['','## Exploratory witness','',
        f"A separate post hoc search finds {w['contrast']} → {w['output']} at capital/kss={w['capital_over_kss']:.6g}: positive terms {w['positive_sum']:+.6g}, negative terms {w['negative_sum']:+.6g}, net {w['delta']:+.6g}. Gross/absolute-net={w['gross_over_abs_net']:.4g}. This witness was chosen for cancellation and is not a typical-case estimate.",'',
        '## What these labels establish','',
        'Labels use all 2048 DISCOVERY examples (seed 112003). Each tag reports the largest absolute Pearson correlation with the three known inputs. Every correlation and activation frequency is available. Inactive or constant units have no defined Pearson correlation; their numeric placeholders are zero.','',
        'The same fixed tags accompany different gate patterns, signed magnitudes and route rankings across states. A tag therefore omits information needed for a finite shock calculation. This is a limitation of the tag, not proof that the circuit has no explanation.','',
        'The complete equations and weights reproduce every response. No human comprehension test, general hardness claim, size-scaling claim, or inference of polysemanticity follows from these measurements. The numerical teacher has checked first-order equations, not established global Bellman optimality.','',
        '## Reproduction and evidence','',
        'Run `python3 system_analysis.py` for local forward-only recomputation. The JSON includes SHA-256 hashes of the saved networks, protocol, teacher artifacts, and source files. It includes exact partition memberships and TRAIN normalization. No pre-existing evidence file is written.','']
    (ROOT/'system_analysis_report.md').write_text('\n'.join(lines))


if __name__=='__main__':
    run()
