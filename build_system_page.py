"""Package frozen labels and full weights with a live system-composition trace."""
import json
from pathlib import Path
import numpy as np
from richer_teacher import load_teacher

ROOT=Path(__file__).resolve().parent


def main():
    analysis=json.loads((ROOT/'system_analysis.json').read_text())
    teacher=load_teacher();capital=np.asarray(analysis['capital_grid_over_kss'])*analysis['kss']
    policies=[]
    for state in range(len(teacher.P)):
        p=teacher.policy(capital,state)
        policies.append(np.column_stack([p[k] for k in ('c','labor','investment')]).tolist())
    model_keys=('id','width','seed','network','labels','fit','top5_route_overlap_low_high')
    models=[{k:m[k] for k in model_keys} for m in analysis['models']]
    economic_interaction=np.asarray(analysis['teacher']['interaction'])
    teacher_rms=np.sqrt(np.mean(economic_interaction**2,axis=0))
    for packaged,original in zip(models,analysis['models']):
        neural=np.asarray(original['grid']['interaction']['network'])
        error_rms=np.sqrt(np.mean((neural-economic_interaction)**2,axis=0))
        packaged['interaction_fidelity']=dict(error_rms_over_teacher_rms=(error_rms/teacher_rms).tolist(),
            sign_agreement=np.mean(np.sign(neural)==np.sign(economic_interaction),axis=0).tolist(),
            scope='Fixed high-productivity/high-work-cost contrast across all 301 capital points; descriptive diagnostic, not checkpoint selection.')
    data=dict(models=models,
        default_model_id=analysis['default_model_id'],normalization=analysis['provenance']['normalization'],
        capital=capital.tolist(),kss=analysis['kss'],features=teacher.state_features.tolist(),
        teacher=policies,provenance=analysis['provenance'])
    serialized=json.dumps(data,separators=(',',':'),allow_nan=False).replace('</','<\\/')
    page=(ROOT/'system_template.html').read_text()
    page=page.replace('/*BASE_STYLE*/',(ROOT/'style.css').read_text())
    page=page.replace('/*SYSTEM_DATA*/','window.SYSTEM_DATA='+serialized+';')
    page=page.replace('/*SYSTEM_SCRIPT*/',(ROOT/'system.js').read_text())
    assert '/*SYSTEM_' not in page
    (ROOT/'system.html').write_text(page)
    print(f'Built system.html ({len(page):,} characters), all nine frozen networks; no training.')


if __name__=='__main__':
    main()
