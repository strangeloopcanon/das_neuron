"""Standalone pure-Python model simulator. No test answers or scoring shortcuts.
Load dsge_prediction_tasks.json; call forward(tasks['model'], raw_input, edit).
Attenuation scales a post-ReLU activation by (1-fraction); incoming edits add
 delta to one standardized-network weight, before evaluating all gates afresh.
"""
import json
from pathlib import Path


def forward(model, raw_input, edit=None):
    n = model['normalization']; w = model['network']
    x = [(v-m)/s for v,m,s in zip(raw_input,n['input_mean'],n['input_std'])]
    w1 = [row[:] for row in w['W1']]; w2 = [row[:] for row in w['W2']]
    edit = edit or {}
    if edit.get('layer') in ('W1','W2'):
        (w1 if edit['layer']=='W1' else w2)[edit['unit']][edit['input_index']] += edit['delta']
    p1 = [sum(a*b for a,b in zip(row,x))+b for row,b in zip(w1,w['b1'])]
    h1 = [max(v,0.) for v in p1]
    if edit.get('layer') == 'H1': h1[edit['unit']] *= 1-edit['fraction']
    p2 = [sum(a*b for a,b in zip(row,h1))+b for row,b in zip(w2,w['b2'])]
    h2 = [max(v,0.) for v in p2]
    if edit.get('layer') == 'H2': h2[edit['unit']] *= 1-edit['fraction']
    yn = [sum(h2[j]*w['W3'][j][o] for j in range(len(h2)))+w['b3'][o] for o in range(3)]
    return dict(normalized_input=x,p1=p1,h1=h1,p2=p2,h2=h2,normalized_output=yn,
                physical_output=[v*s+m for v,s,m in zip(yn,n['output_std'],n['output_mean'])])


if __name__ == '__main__':
    import argparse
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('raw_input',help='JSON array with 3 physical inputs');p.add_argument('--edit',default='{}');p.add_argument('--tasks',default=str(Path(__file__).with_name('dsge_prediction_tasks.json')))
    a=p.parse_args(); print(json.dumps(forward(json.loads(Path(a.tasks).read_text())['model'],json.loads(a.raw_input),json.loads(a.edit)),indent=2))
