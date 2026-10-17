"""Independent numerical and response-validation checks for saved DSGE tasks.
Run with Python and NumPy; does not generate tasks or modify study artifacts.
"""
import copy
import hashlib
import json
from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

ROOT = Path(__file__).resolve().parent


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(value).hexdigest()


def independent_forward(model, raw, edit=None):
    """Matrix implementation, independent of the participant simulator."""
    n = model['normalization']
    w = {key: np.asarray(value, dtype=float).copy() for key, value in model['network'].items()}
    x = (np.asarray(raw) - n['input_mean']) / n['input_std']
    e = edit or {}
    layer = e.get('layer')
    if layer in ('W1', 'W2'):
        w[layer][e['unit'], e['input_index']] += e['delta']
    p1 = w['W1'] @ x + w['b1']
    h1 = np.maximum(p1, 0)
    if layer == 'H1':
        h1[e['unit']] *= 1 - e['fraction']
    p2 = w['W2'] @ h1 + w['b2']
    h2 = np.maximum(p2, 0)
    if layer == 'H2':
        h2[e['unit']] *= 1 - e['fraction']
    y = h2 @ w['W3'] + w['b3']
    return dict(normalized_input=x, p1=p1, h1=h1, p2=p2, h2=h2,
                normalized_output=y, physical_output=y * n['output_std'] + n['output_mean'])


def independent_jacobian(model, task):
    w = {key: np.asarray(value) for key, value in model['network'].items()}
    b = independent_forward(model, task['raw_input'])
    e = task['intervention']
    unit, out = e['unit'], task['output_index']
    if e['layer'] == 'H2':
        return -e['fraction'] * b['h2'][unit] * w['W3'][unit, out]
    dh1 = np.zeros(16)
    dp2 = np.zeros(16)
    if e['layer'] == 'H1':
        dh1[unit] = -e['fraction'] * b['h1'][unit]
    elif e['layer'] == 'W1':
        dh1[unit] = e['delta'] * b['normalized_input'][e['input_index']] * (b['p1'][unit] > 0)
    else:
        dp2[unit] = e['delta'] * b['h1'][e['input_index']]
    return ((w['W2'] @ dh1 + dp2) * (b['p2'] > 0)) @ w['W3'][:, out]


def main():
    from dsge_prediction_materials import forward
    from dsge_prediction_experiment import jacobian
    from score_dsge_predictions import score, summarize
    tasks_path = ROOT / 'dsge_prediction_tasks.json'
    protocol_path = ROOT / 'dsge_prediction_protocol.json'
    tasks = json.loads(tasks_path.read_bytes())
    protocol = json.loads(protocol_path.read_bytes())
    baselines = json.loads((ROOT / 'dsge_prediction_baselines.json').read_bytes())
    model = tasks['model']
    assert sum(np.asarray(v).size for v in model['network'].values()) == protocol['parameter_count'] == 387
    hashes = dict(protocol_sha256=digest(protocol_path.read_bytes()), tasks_sha256=digest(tasks_path.read_bytes()),
                  model_sha256=digest(canonical(model)), labels_sha256=digest(canonical(tasks['labels'])))
    for key, value in hashes.items():
        assert baselines['metadata'][key] == value, key
        if key != 'tasks_sha256':
            assert tasks['metadata'][key] == value, key
    for filename, expected in tasks['metadata']['source_sha256'].items():
        assert digest((ROOT / filename).read_bytes()) == expected, filename
    assert baselines['metadata']['source_sha256'] == tasks['metadata']['source_sha256']
    richer = json.loads((ROOT / 'richer_results.json').read_bytes())
    system = json.loads((ROOT / 'system_analysis.json').read_bytes())
    saved = next(m for m in richer['models'] if m['width'] == 16 and m['seed'] == 113)
    assert model['network'] == saved['network']
    assert model['normalization'] == richer['provenance']['normalization']
    assert tasks['labels'] == next(m for m in system['models'] if m['id'] == model['id'])['labels']
    assert model['inputs'] == system['inputs'] and model['outputs'] == system['outputs']
    partitions = np.load(ROOT / 'richer_partitions.npz')
    seen = {(float(k).hex(), int(s)) for name in partitions.files if name.endswith('_capital')
            for k, s in zip(partitions[name], partitions[name.replace('_capital', '_state')])}
    ids = set()
    max_error = 0.
    truth_by_id = {}
    for bank, rows in tasks['banks'].items():
        assert len(rows) == protocol['banks'][bank]
        if bank != 'practice':
            assert Counter((t['family'], t['output_index']) for t in rows) == Counter({(f, o):len(rows)//12 for f in protocol['families'] for o in range(3)})
        for task in rows:
            assert task['task_id'] not in ids
            ids.add(task['task_id'])
            assert task['bank'] == bank
            key = (float(task['raw_input'][0]).hex(), task['state'])
            assert key not in seen, key
            seen.add(key)
            sampling = tasks['teacher_state_sampling']
            assert .5 * sampling['kss'] <= task['raw_input'][0] <= 1.8 * sampling['kss']
            np.testing.assert_array_equal(task['raw_input'][1:], sampling['state_features'][task['state']])
            baseline = independent_forward(model, task['raw_input'])
            edited = independent_forward(model, task['raw_input'], task['intervention'])
            assert set(task['baseline']) == set(baseline) - {'normalized_input'}
            assert task['output_name'] == model['outputs'][task['output_index']]
            for key, expected in baseline.items():
                actual = task['normalized_input'] if key == 'normalized_input' else task['baseline'][key]
                np.testing.assert_allclose(actual, expected, atol=2e-13, rtol=2e-13)
            for key, expected in edited.items():
                actual = forward(model, task['raw_input'], task['intervention'])[key]
                max_error = max(max_error, float(np.max(np.abs(np.asarray(actual)-expected))))
                np.testing.assert_allclose(actual, expected, atol=2e-13, rtol=2e-13)
            np.testing.assert_allclose(jacobian(model, task), independent_jacobian(model, task), atol=2e-13, rtol=2e-13)
            e = task['intervention']
            assert baseline['h2' if e['layer'] in ('H2', 'W2') else 'h1'][e['unit']] > 0
            if e['layer'] == 'W2':
                assert baseline['h1'][e['input_index']] > 0
            if task['family'] in ('h1_attenuation', 'h2_attenuation'):
                assert e['layer'] == ('H1' if task['family'] == 'h1_attenuation' else 'H2')
                assert e['fraction'] in protocol['interventions']['attenuation_fractions']
            elif task['family'] == 'incoming_weight':
                assert e['layer'] in ('W1', 'W2')
                assert e['delta'] in protocol['interventions']['incoming_weight_delta']
            if task['family'] == 'sham':
                assert e['layer'] == 'H1' and e['fraction'] == 0
            out = task['output_index']
            truth_by_id[task['task_id']] = float(edited['normalized_output'][out] - baseline['normalized_output'][out])
            if bank == 'practice':
                np.testing.assert_allclose(task['answer_sd'], truth_by_id[task['task_id']], atol=2e-13)
                for key, value in edited.items():
                    np.testing.assert_allclose(task['edited_trace'][key], value, atol=2e-13, rtol=2e-13)
            else:
                assert set(task) == {'task_id','bank','family','output_index','output_name','raw_input','normalized_input','state','baseline','intervention'}
        for family, reported in baselines['automated_baselines'][bank].items():
            subset = [t for t in rows if family == 'all' or t['family'] == family]
            for name, metric in reported.items():
                selected = [t for t in subset if name != 'h2_direct_readout' or t['family'] == 'h2_attenuation']
                y = np.array([truth_by_id[t['task_id']] for t in selected])
                p = np.array([0 if name == 'zero_change' else independent_jacobian(model,t) if name in ('frozen_gate_jacobian','h2_direct_readout') else truth_by_id[t['task_id']] for t in selected])
                direction = lambda a: np.where(abs(a)<=.002,0,np.sign(a))
                expected = dict(n=len(y),mae_sd=float(abs(p-y).mean()),numeric_accuracy=float((abs(p-y)<=.02).mean()),direction_accuracy=float((direction(p)==direction(y)).mean()))
                for key, value in expected.items():
                    np.testing.assert_allclose(metric[key], value, atol=2e-12, rtol=2e-12)
    cells = [(c,f) for c in protocol['conditions'] for f in ('A','B')]
    assert [a['slot'] for a in tasks['allocations']] == list(range(24))
    for start in range(0,24,4):
        assert Counter((a['condition'],a['form']) for a in tasks['allocations'][start:start+4]) == Counter(cells)
    a = tasks['allocations'][0]
    rows = tasks['banks'][a['form']]
    export = dict(schema_version=1,participant_kind='human',participant_id='verification',slot=0,condition=a['condition'],form=a['form'],task_order=[t['task_id'] for t in rows],responses=[dict(task_id=t['task_id'],prediction_sd=truth_by_id[t['task_id']],response_ms=1.,tool_use='independent matrix calculation') for t in rows],materials_ms=3.,total_ms=15.,metadata=hashes)
    result = score(export)
    assert result['overall']['mae_sd'] < 2e-13 and result['overall']['numeric_accuracy'] == 1 and result['overall']['direction_accuracy'] == 1
    wrong = copy.deepcopy(export)
    for r in wrong['responses']: r['prediction_sd'] += 1
    assert score(wrong)['overall']['mae_sd'] > .99
    cases = [lambda e:e['metadata'].update(tasks_sha256='wrong'),lambda e:e.update(participant_kind='invalid'),lambda e:e.update(slot=True),lambda e:e.update(form='B' if a['form']=='A' else 'A'),lambda e:e['responses'].pop(),lambda e:e['task_order'].__setitem__(0,e['task_order'][1]),lambda e:e['responses'][0].update(prediction_sd=float('nan')),lambda e:e['responses'][0].update(prediction_sd=True),lambda e:e['responses'][0].update(response_ms=-1),lambda e:e.update(total_ms=0),lambda e:e['responses'][0].update(tool_use=None)]
    for mutate in cases:
        bad = copy.deepcopy(export); mutate(bad)
        try: score(bad)
        except (ValueError,KeyError,TypeError): pass
        else: raise AssertionError('malformed export accepted')
    with TemporaryDirectory() as directory:
        paths=[]
        for i, (kind, ident) in enumerate([('human','verification'),('human','verification'),('human','other-human'),('demo','demo'),('agent_smoke','smoke')]):
            e=copy.deepcopy(export);e.update(participant_kind=kind,participant_id=ident)
            path=Path(directory)/f'{i}.json';path.write_text(json.dumps(e));paths.append(path)
        pooled=summarize(paths)
        assert len(pooled['exclusions']) == 2
        assert len(pooled['included_participants']) == 3
        assert pooled['groups']['human'][a['condition']+'/'+a['form']]['n_participants'] == 1
    print(json.dumps(dict(status='PASS',tasks=len(ids),parameter_count=387,max_forward_difference=max_error,invalid_export_cases=len(cases),human_completed=baselines['human_completed']),indent=2))


if __name__ == '__main__':
    main()
