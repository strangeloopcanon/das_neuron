"""Independently reconstruct the saved finite-shock system analysis.

Forward-only and read-only: no generator imports, training, teacher solution,
or writes. Reuses the frozen teacher's policy interpolation/static recovery.
Checks every saved numeric grid, teaching example, hidden-unit tag and weight.
"""
import os
for _name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[_name] = '1'
import hashlib
import json
import sys
from pathlib import Path
sys.dont_write_bytecode = True
import numpy as np
from richer_teacher import load_teacher

ROOT = Path(__file__).resolve().parent
INPUTS = ['capital', 'log productivity', 'labor-disutility shock']
OUTPUTS = ['consumption', 'labor', 'investment']
CORNERS = {'baseline': 10, 'productivity': 19, 'cost': 11, 'both': 20}
CONTRASTS = {'productivity': 'productivity', 'cost': 'cost', 'joint': 'both'}
WEIGHTS = ('W1', 'b1', 'W2', 'b2', 'W3', 'b3')


def check(condition, message):
    if not condition:
        raise AssertionError(message)


def close(actual, expected, message, atol=5e-13, rtol=1e-10):
    np.testing.assert_allclose(np.asarray(actual, float), np.asarray(expected, float),
                               atol=atol, rtol=rtol, equal_nan=True, err_msg=message)


def sha(name):
    return hashlib.sha256((ROOT/name).read_bytes()).hexdigest()


def array_sha(*arrays):
    digest = hashlib.sha256()
    for a in arrays:
        a = np.ascontiguousarray(a)
        digest.update(str(a.dtype).encode())
        digest.update(str(a.shape).encode())
        digest.update(a.tobytes())
    return digest.hexdigest()


def forward(x, net):
    # Independently derived affine/ReLU/affine/ReLU/readout equations.
    p1 = x @ net['W1'].T + net['b1']
    h1 = np.clip(p1, 0, None)
    p2 = h1 @ net['W2'].T + net['b2']
    h2 = np.clip(p2, 0, None)
    return h2 @ net['W3'] + net['b3'], dict(p1=p1, h1=h1, p2=p2, h2=h2)


def summaries(terms, delta):
    pos = np.where(terms > 0, terms, 0).sum(axis=1)
    neg = np.where(terms < 0, terms, 0).sum(axis=1)
    gross = np.abs(terms).sum(axis=1)
    ratio = np.full_like(delta, np.nan)
    np.divide(gross, np.abs(delta), out=ratio, where=np.abs(delta) > 1e-14)
    fraction = np.zeros_like(gross)
    np.divide(gross-np.abs(delta), gross, out=fraction, where=gross > 1e-14)
    top = np.sort(np.abs(terms), axis=1)[:, -5:, :].sum(axis=1)
    share = np.zeros_like(gross)
    np.divide(top, gross, out=share, where=gross > 1e-14)
    return dict(positive_sum=pos, negative_sum=neg, gross_sum=gross,
                gross_over_abs_net=ratio, cancellation_fraction=fraction,
                top5_gross_share=share)


def top5(terms):
    return sorted(range(len(terms)), key=lambda j: (-abs(terms[j]), j))[:5]


def route_counts(net, hidden):
    # Count per state with explicit matrix products, separate from generator.
    g1 = hidden['h1'] > 0
    g2 = hidden['h2'] > 0
    edge = (net['W2'] != 0).astype(np.int64)
    upstream_edges = g1.astype(np.int64) @ edge.T
    open_edges = (upstream_edges * g2).sum(axis=1)
    input_degree = np.count_nonzero(net['W1'], axis=1)
    to_h2 = (g1 * input_degree) @ edge.T
    paths = (to_h2 * g2) @ (net['W3'] != 0).astype(np.int64)
    return dict(open_h1_h2_edges=open_edges, input_to_output_paths=paths)


def verify():
    saved = json.loads((ROOT/'system_analysis.json').read_text())
    frozen = json.loads((ROOT/'richer_results.json').read_text())
    protocol = json.loads((ROOT/'richer_protocol.json').read_text())
    teacher_report = json.loads((ROOT/'richer_teacher.json').read_text())
    report = (ROOT/'system_analysis_report.md').read_text()
    teacher = load_teacher()
    check(saved['schema_version'] == 1, 'schema')
    check(saved['inputs'] == INPUTS and saved['outputs'] == OUTPUTS, 'input/output names')
    required_hashes = {'system_analysis.py', 'richer_experiment.py', 'richer_teacher.py',
                       'richer_protocol.json', 'richer_results.json', 'richer_teacher.npz', 'richer_teacher.json'}
    check(set(saved['provenance']['source_sha256']) == required_hashes, 'complete analysis hashes')
    for name, expected in saved['provenance']['source_sha256'].items():
        check(sha(name) == expected, 'analysis source hash '+name)
    check(sha('richer_protocol.json') == frozen['provenance']['protocol_sha256'], 'frozen protocol hash')
    check(frozen['protocol'] == protocol, 'frozen protocol content')
    for group in ('source_sha256', 'teacher_sha256'):
        for name, expected in frozen['provenance'][group].items():
            check(sha(name) == expected, 'frozen source hash '+name)
    tp = teacher_report['provenance']
    for source_key, hash_key in [('source', 'source_sha256'), ('artifact', 'artifact_sha256'),
                                  ('rouwenhorst_source', 'rouwenhorst_source_sha256')]:
        check(sha(tp[source_key]) == tp[hash_key], 'teacher provenance '+source_key)
    for key, value in teacher_report['parameters'].items():
        close(teacher.parameters[key], value, 'teacher parameter '+key, atol=0, rtol=0)

    # Recreate all split memberships and TRAIN normalization from saved protocol.
    data, membership, all_states = {}, {}, set()
    for name, spec in protocol['data'].items():
        if not isinstance(spec, dict):
            continue
        rng = np.random.default_rng(spec['seed'])
        capital = rng.uniform(.5, 1.8, spec['n']) * teacher.parameters['kss']
        states = rng.choice(len(teacher.P), spec['n'], p=teacher.stationary)
        raw = np.column_stack((capital, teacher.state_features[states]))
        pol = teacher.policy(capital, states)
        y = np.column_stack([pol[k] for k in ('c', 'labor', 'investment')])
        keys = {(float(k).hex(), int(s)) for k, s in zip(capital, states)}
        check(len(keys) == len(capital) and not keys & all_states, 'unique disjoint '+name)
        all_states |= keys
        membership[name] = dict(**spec, sha256=array_sha(capital, states.astype(np.int64), y),
                               shock_state_counts=np.bincount(states, minlength=len(teacher.P)).tolist())
        data[name] = dict(raw=raw, y=y)
    check(membership == saved['provenance']['partitions'] == frozen['provenance']['partitions'], 'all split receipts')
    tr = data['TRAIN']
    norm = dict(input_mean=tr['raw'].mean(0), input_std=tr['raw'].std(0),
                output_mean=tr['y'].mean(0), output_std=tr['y'].std(0))
    for key, value in norm.items():
        close(value, saved['provenance']['normalization'][key], 'analysis normalization '+key, atol=0, rtol=0)
        close(value, frozen['provenance']['normalization'][key], 'frozen normalization '+key, atol=0, rtol=0)
    discovery = (data['DISCOVERY']['raw']-norm['input_mean'])/norm['input_std']
    ratios = np.linspace(.5, 1.8, 301)
    close(saved['capital_grid_over_kss'], ratios, 'capital grid', atol=0, rtol=0)
    close(saved['kss'], teacher.parameters['kss'], 'kss', atol=0, rtol=0)
    capital = ratios * teacher.parameters['kss']
    indices = [int(np.argmin(abs(ratios-k))) for k in (.6, 1, 1.6)]
    check(saved['example_indices'] == indices, 'fixed example selection')
    corner_raw, truth = {}, {}
    for name, state in CORNERS.items():
        s = saved['states'][name]
        check((s['index'], s['productivity_index'], s['labor_disutility_index']) ==
              (state, state//3, state%3), 'corner index '+name)
        close(s['features'], teacher.state_features[state], 'corner features '+name, atol=0, rtol=0)
        corner_raw[name] = np.column_stack((capital, np.tile(teacher.state_features[state], (len(ratios), 1))))
        pol = teacher.policy(capital, np.full(len(ratios), state))
        truth[name] = np.column_stack([pol[k] for k in ('c', 'labor', 'investment')])
        close(truth[name], saved['teacher']['physical_outputs'][name], 'teacher grid '+name)
    teacher_interaction = truth['both']-truth['productivity']-truth['cost']+truth['baseline']
    close(teacher_interaction, saved['teacher']['interaction'], 'teacher interaction')
    expected_ids = {f'H{w}-seed{s}' for w in protocol['architecture']['widths'] for s in protocol['architecture']['seeds']}
    check({m['id'] for m in saved['models']} == expected_ids and len(saved['models']) == 9, 'nine distinct model identities')
    check(saved['default_model_id'] == 'H16-seed113' and 'fixed before' in saved['default_selection'], 'fixed default disclosure')
    total_labels, undefined, max_residual, max_parameters = 0, 0, 0., 0
    interaction_fidelity = {}
    for model in saved['models']:
        name = model['id']
        original = next(m for m in frozen['models'] if (m['width'], m['seed']) == (model['width'], model['seed']))
        check(model['network'] == original['network'], name+' exact frozen coefficients')
        check(model['training'] == original['training'] and model['fit'] == original['fit'], name+' frozen training/fit')
        check(set(model['network']) == set(WEIGHTS), name+' complete coefficients')
        net = {key: np.asarray(value, float) for key, value in model['network'].items()}
        width = model['width']
        shapes = dict(W1=(width, 3), b1=(width,), W2=(width, width), b2=(width,), W3=(width, 3), b3=(3,))
        for key, a in net.items():
            check(a.shape == shapes[key] and np.isfinite(a).all(), name+' coefficient shape '+key)
        max_parameters = max(max_parameters, sum(a.size for a in net.values()))
        _, dh = forward(discovery, net)
        for layer, key in [(1, 'H1'), (2, 'H2')]:
            acts = dh[f'h{layer}']
            check(len(model['labels'][key]) == width, name+' all labels '+key)
            for j, label in enumerate(model['labels'][key]):
                variable = bool(acts[:, j].std() > 1e-14)
                freq = np.count_nonzero(acts[:, j] > 0) / len(acts)
                if variable:
                    corr = [np.corrcoef(acts[:, j], discovery[:, i])[0, 1] for i in range(3)]
                    strongest = int(np.argmax(np.abs(corr)))
                    text = f'{INPUTS[strongest]} association (r={corr[strongest]:+.3f})'
                else:
                    corr, strongest = [0., 0., 0.], None
                    text = 'inactive on DISCOVERY' if freq == 0 else 'constant on DISCOVERY; correlation undefined'
                    undefined += 1
                check(label['unit'] == j and label['correlation_defined'] == variable, name+' label identity/definedness')
                close(label['correlations'], corr, name+' all input correlations')
                close(label['activation_frequency'], freq, name+' activation frequency', atol=0, rtol=0)
                check(label['strongest_input_index'] == strongest and label['label'] == text, name+' qualified tag')
                definition = (f'h1[{j}]=max(0, b1[{j}]+sum_i W1[{j},i]*x[i])' if layer == 1 else
                              f'h2[{j}]=max(0, b2[{j}]+sum_j W2[{j},j]*h1[j])')
                check(label['definition'] == definition, name+' exact unit definition')
                check(label['incoming_weight_reference'] == f'network.W{layer}[{j}]' and
                      label['bias_reference'] == f'network.b{layer}[{j}]', name+' coefficient references')
                total_labels += 1
        snaps = {}
        for corner, raw in corner_raw.items():
            x = (raw-norm['input_mean'])/norm['input_std']
            yn, h = forward(x, net)
            snaps[corner] = dict(raw=raw, x=x, physical_outputs=yn*norm['output_std']+norm['output_mean'],
                                 **h, route_counts=route_counts(net, h))
        base = snaps['baseline']
        calculations = {}
        for contrast, corner in CONTRASTS.items():
            target = snaps[corner]
            delta = target['physical_outputs']-base['physical_outputs']
            terms = np.einsum('nj,jo,o->njo', target['h2']-base['h2'], net['W3'], norm['output_std'])
            residual = float(np.max(abs(terms.sum(1)-delta)))
            max_residual = max(max_residual, residual)
            calc = dict(delta=delta, teacher_delta=truth[corner]-truth['baseline'], **summaries(terms, delta),
                        h1_gate_changes=np.sum((target['h1'] > 0) != (base['h1'] > 0), axis=1),
                        h2_gate_changes=np.sum((target['h2'] > 0) != (base['h2'] > 0), axis=1),
                        attribution_max_absolute_residual=residual)
            for metric, values in calc.items():
                close(model['grid'][contrast][metric], values, name+' complete '+contrast+' grid '+metric)
            calculations[contrast] = dict(**calc, contribution=terms)
        interaction = snaps['both']['physical_outputs']-snaps['productivity']['physical_outputs']-snaps['cost']['physical_outputs']+base['physical_outputs']
        terms_i = np.einsum('nj,jo,o->njo', snaps['both']['h2']-snaps['productivity']['h2']-snaps['cost']['h2']+base['h2'], net['W3'], norm['output_std'])
        residual = float(np.max(abs(terms_i.sum(1)-interaction)))
        max_residual = max(max_residual, residual)
        for key, values in dict(network=interaction, teacher=teacher_interaction, error=interaction-teacher_interaction,
                                attribution_max_absolute_residual=residual).items():
            close(model['grid']['interaction'][key], values, name+' full interaction grid '+key)
        check(len(model['examples']) == 3, name+' three examples')
        for example, index in zip(model['examples'], indices):
            check(example['grid_index'] == index, name+' example grid index')
            close(example['capital_over_kss'], ratios[index], name+' example ratio', atol=0, rtol=0)
            close(example['capital'], capital[index], name+' example capital', atol=0, rtol=0)
            for corner, snap in snaps.items():
                for key, value in snap.items():
                    if key == 'route_counts':
                        for route, counts in value.items():
                            close(example['snapshots'][corner][key][route], counts[index], name+' exact open path counts', atol=0, rtol=0)
                    else:
                        close(example['snapshots'][corner][key], value[index], name+' saved snapshot '+key)
            for contrast, calc in calculations.items():
                ex = example['contrasts'][contrast]
                check(ex['target'] == CONTRASTS[contrast], name+' contrast target')
                for key, values in calc.items():
                    close(ex[key], values[index] if isinstance(values, np.ndarray) else values, name+' example '+contrast+' '+key)
                check(ex['top5_units_by_output'] == [top5(calc['contribution'][index, :, o]) for o in range(3)], name+' top-five tie rule')
            for key, values in dict(network=interaction, teacher=teacher_interaction, error=interaction-teacher_interaction, contribution=terms_i).items():
                close(example['interaction'][key], values[index], name+' example interaction '+key)
        for contrast, calc in calculations.items():
            for o, row in enumerate(model['top5_route_overlap_low_high'][contrast]):
                low, high = [top5(calc['contribution'][ix, :, o]) for ix in (indices[0], indices[-1])]
                common = len(set(low) & set(high))
                check(row['output'] == OUTPUTS[o] and row['low_units'] == low and row['high_units'] == high, name+' low/high unit identities')
                check(row['intersection_count'] == common, name+' overlap count')
                close([row['overlap_fraction'], row['jaccard']], [common/5, common/(10-common)], name+' overlap fractions')
        # Reproduce the post hoc exhaustive witness search, including exclusion.
        candidates = []
        for contrast, calc in calculations.items():
            ratio = calc['gross_over_abs_net'].copy()
            ratio[abs(calc['delta']) < 1e-6*norm['output_std']] = -np.inf
            index, o = np.unravel_index(np.nanargmax(ratio), ratio.shape)
            candidates.append((ratio[index, o], contrast, int(index), int(o)))
        _, contrast, ix, o = max(candidates)
        witness = model['exploratory_witness']
        check('Exploratory' in witness['selection'] and '301' in witness['selection'] and 'excludes' in witness['selection'], name+' post hoc disclosure')
        check((witness['contrast'], witness['grid_index'], witness['output_index'], witness['output']) ==
              (contrast, ix, o, OUTPUTS[o]), name+' exhaustive witness identity')
        close(witness['capital_over_kss'], ratios[ix], name+' witness ratio')
        for key in ('delta', 'teacher_delta', 'positive_sum', 'negative_sum', 'gross_over_abs_net', 'cancellation_fraction'):
            close(witness[key], calculations[contrast][key][ix, o], name+' witness '+key)
        for key in ('h1_gate_changes', 'h2_gate_changes'):
            check(witness[key] == calculations[contrast][key][ix], name+' witness gate changes')
        rms_truth = np.sqrt(np.mean(teacher_interaction**2, axis=0))
        error = interaction-teacher_interaction
        interaction_fidelity[name] = dict(error_rms_over_teacher_rms=(np.sqrt(np.mean(error**2, axis=0))/rms_truth).tolist(),
            sign_agreement=np.mean(np.sign(interaction) == np.sign(teacher_interaction), axis=0).tolist())
        if name == saved['default_model_id']:
            for example in model['examples']:
                c, i = example['contrasts']['joint'], example['interaction']
                for o, output in enumerate(OUTPUTS):
                    row = f"| {example['capital_over_kss']:.4f} | {output} | {c['delta'][o]:+.6g} | {c['positive_sum'][o]:+.6g} | {c['negative_sum'][o]:+.6g} | {i['network'][o]:+.6g} | {i['teacher'][o]:+.6g} | {i['error'][o]:+.6g} |"
                    check(row in report, 'report teaching row '+output)
            for contrast, overlap in model['top5_route_overlap_low_high'].items():
                row = '| '+contrast+' | '+' | '.join(f"{r['intersection_count']}/5" for r in overlap)+' |'
                check(row in report, 'report overlap '+contrast)
            for value in (f"{witness['capital_over_kss']:.4g}", f"{witness['positive_sum']:+.6g}",
                          f"{witness['negative_sum']:+.6g}", f"{witness['delta']:+.6g}",
                          f"{witness['gross_over_abs_net']:.1f}"):
                check(value in report, 'report witness value '+value)
            for phrase in ('post hoc', 'not a typical-case estimate', 'not call all network interaction an economic effect',
                           'not proof that the circuit has no explanation', 'not established global Bellman optimality',
                           'numeric placeholders are zero'):
                check(phrase in report, 'report qualification '+phrase)
    verification = saved['verification']
    check(total_labels == verification['labeled_hidden_unit_count'] == 672, 'all 672 hidden units')
    check(verification['model_count'] == 9 and verification['all_frozen_weights_preserved'] is True, 'saved verification counts')
    check(max_residual < verification['attribution_tolerance'] == 1e-12, 'full-grid accounting tolerance')
    close(verification['maximum_absolute_attribution_residual'], max_residual, 'maximum accounting residual')
    check(f'{max_residual:.3g}' in report, 'report maximum accounting residual')
    return dict(status='PASS', models=9, labeled_hidden_units=total_labels,
                undefined_correlations=undefined, maximum_parameters_per_model=max_parameters,
                capital_points_per_corner=301, shock_corners=4,
                full_model_grid_forward_states=9*301*4,
                maximum_absolute_attribution_residual=max_residual,
                interaction_fidelity=interaction_fidelity,
                scope='Saved artifacts independently reconstructed. Teacher policy recovery reused. No training, teacher solve, or writes.')


if __name__ == '__main__':
    print(json.dumps(verify(), indent=2))
