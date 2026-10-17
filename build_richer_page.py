"""Build a standalone larger-economy explorer and a measured research readout."""
import json
from pathlib import Path
import numpy as np
from richer_teacher import load_teacher

ROOT=Path(__file__).resolve().parent


def main():
    results=json.loads((ROOT/'richer_results.json').read_text())
    teacher_report=json.loads((ROOT/'richer_teacher.json').read_text())
    teacher=load_teacher();models=results['models'];norm=results['provenance']['normalization']
    capital=np.linspace(.5,1.8,301)*teacher.parameters['kss']
    policies=[]
    for s in range(len(teacher.P)):
        policy=teacher.policy(capital,s)
        policies.append(np.column_stack([policy[key] for key in ('c','labor','investment')]).tolist())
    eligible=sum(m['training']['eligible'] for m in models)
    endpoints={}
    for seed in (113,229,347):
        paired=[m for m in models if m['seed']==seed]
        endpoints[seed]=[m['interventions']['first_layer'][2]['normalized_response_rmse'] for m in paired]
    increases=sum(v[-1]>v[0] for v in endpoints.values())
    conclusion=(f'{eligible}/9 networks passed the three-output accuracy gate. '
        f'The local removal-prediction error rose from the narrowest to widest network in {increases}/3 seeds. '
        'The width results are descriptive for this one model and explanation method. '
        'They do not prove a general increase in understanding difficulty.')
    teacher_summary=(f'The numerical solution uses 960 capital points and 21 joint shock states. '
        f"Across 12,000 fresh states, the maximum relative Euler residual is {teacher_report['fine_diagnostics']['euler_relative']['maximum']:.3g}. "
        'Resource and labour residuals, grid refinement, and 10,000 simulated transitions pass the declared checks. '
        'These checks support a numerical first-order solution; they do not certify global Bellman optimality.')
    data=dict(models=models,norm=norm,features=teacher.state_features.tolist(),
        capital=capital.tolist(),teacher=policies,kss=teacher.parameters['kss'],
        baselines=results['baselines'],protocol=results['protocol'],
        membership=results['provenance']['partitions'],conclusion=conclusion,teacher_summary=teacher_summary)
    serialized=json.dumps(data,separators=(',',':')).replace('</','<\\/')
    page=(ROOT/'richer_template.html').read_text().replace('/*BASE_STYLE*/',(ROOT/'style.css').read_text())
    page=page.replace('/*RICHER_DATA*/','window.RICHER_DATA='+serialized+';').replace('/*RICHER_SCRIPT*/',(ROOT/'richer.js').read_text())
    assert '/*RICHER_' not in page
    (ROOT/'richer.html').write_text(page)
    pct=lambda v:f'{100*v:.2f}%'
    rows=[]
    for m in models:
        errors=' | '.join(pct(v) for v in m['fit']['TEST']['normalized_rmse'])
        rows.append(f"| {m['width']} | {m['seed']} | {m['training']['parameter_count']} | {'Pass' if m['training']['eligible'] else 'FAIL'} | {errors} | {pct(m['interventions']['first_layer'][2]['normalized_response_rmse'])} |")
    baseline_rows=[]
    for name,b in results['baselines'].items():
        baseline_rows.append('| '+name+' | '+' | '.join(pct(v) for v in b['normalized_rmse'])+' |')
    diagnostics_rows=[]
    for m in models:
        d=m['economic_residuals']
        diagnostics_rows.append(f"| {m['width']} / {m['seed']} | {pct(d['resource_relative_rms'])} | {pct(d['labor_foc_relative_rms'])} | {pct(d.get('euler_relative_rms',float('nan')))} |")
    teacher_sd=np.asarray(norm['output_std'])
    refinement=np.array([teacher_report['grid_refinement'][key]['absolute']['maximum'] for key in ('c','labor','investment')])/teacher_sd
    dense=np.array([m['fit']['dense_21_state_grid']['normalized_max_error'] for m in models])
    labels=[]
    for m in models:
        l=m['labels'];r=l['responses']
        labels.append(f"| {m['width']} / {m['seed']} | {l['unit']} | {l['selected_correlations'][1]:.3f} | {pct(r[1]['retained_response_norm_ratio'])} | {pct(r[0]['response_distortion_ratio'])} | {pct(r[2]['response_distortion_ratio'])} |")
    report=f'''# A larger nonlinear economic circuit

The economy now includes endogenous labour, capital installation costs, and a second shock. Nine networks learn three policy outputs: consumption, labour, and investment. Open `richer.html` to inspect their decisions and actual finite interventions.

{conclusion}

## The added economics

The previous model fixed labour at one and installed capital without adjustment costs. This model lets the household choose positive work effort and pay convex installation costs. Work has no unit time cap. Investment may be negative. These conventions make the reported equations interior first-order conditions rather than constrained alternatives.

```
maximize E sum_t beta^t [c_t^(1-sigma)/(1-sigma) - chi exp(eta_t) n_t^(1+phi)/(1+phi)]
y = z k^alpha n^(1-alpha)
Phi(k,i) = (kappa/2) k (i/k-delta)^2
c + i + Phi(k,i) = y
k_next = (1-delta) k + i
chi exp(eta) n^phi = c^(-sigma) (1-alpha)y/n
q = 1 + kappa (i/k-delta)
c^(-sigma) q = beta E[c_next^(-sigma) (alpha y_next/k_next + (1-delta)q_next - Phi_k_next)]
Phi_k holding i fixed = (kappa/2) [delta^2-(i/k)^2]
```

Competitive firms pay wage `(1-alpha)y/n` and rental rate `alpha y/k`. Households own capital and pay installation costs. Their budget and factor income reproduce resource clearing. The independent equation review confirmed both cost derivatives and the Euler equation. [Carroll's investment notes](https://www.econ2.jhu.edu/people/ccarroll/public/lecturenotes/Investment/qModel/) provide a related adjustment-cost and envelope derivation; the equations above specify this model.

Calibration: beta=.96, sigma=2, alpha=.36, delta=.08, phi=1, kappa=2. The deterministic reference labour is .33. It implies k=1.7974464, c=.4636746, chi=16.6054455. This reference is not the stochastic mean. Independent 7-state productivity and 3-state labour-cost Rouwenhorst chains give 21 joint states. The finite transition matrix is the benchmark, not an exact continuous-shock solution.

## Numerical solution

{teacher_summary}

The solver uses endogenous-grid policy time iteration. It integrates future marginal values over all 21 joint states. Each policy query interpolates next capital, then solves the resource and labour equations. Capital is solved on [.08,4] reference capital; study states use [.5,1.8]. No extrapolation is allowed. The fresh core checks remain strictly inside the numerical capital boundaries.

Coarse-to-fine maximum differences, scaled by each output's TRAIN SD: consumption {pct(refinement[0])}, labour {pct(refinement[1])}, investment {pct(refinement[2])}. These are refinement diagnostics, not rigorous error bounds. The Euler residual is not a forecast or welfare score.

## Frozen neural sweep

Architecture: 3 inputs → H ReLU → H ReLU → 3 outputs. Every weight and bias is trained. Parameter counts are H²+8H+3: 387, 1,283, and 4,611. Seeds are 113, 229, and 347. No units are duplicated and no output weights are withheld.

The frozen `richer_protocol.json` precedes training. A 1,000-step pilot measured cost before nine 30,000-step local CPU fits. Saved total execution time including analysis was {results['seconds']:.2f} seconds. Checkpoint selection uses the first scheduled VALIDATION checkpoint where every output's RMSE is at most 2% of its TRAIN SD. Failed models would remain visible. Default display is fixed at width32/seed113.

| Width | Seed | Parameters | Fit gate | Consumption TEST | Labour TEST | Investment TEST | Removal prediction error |
|---:|---:|---:|---|---:|---:|---:|---:|
{chr(10).join(rows)}

Policy errors use each output's TRAIN standard deviation. They are not percentages of consumption, labour, or investment levels. The final column uses RMS actual intervention response as its denominator.

| Baseline | Consumption TEST | Labour TEST | Investment TEST |
|---|---:|---:|---:|
{chr(10).join(baseline_rows)}

The quadratic baseline is a TRAIN-fitted degree-two polynomial with ten terms per output. These simple baselines matter: the economic policy can remain compact even if its neural implementation has many weights. Random-initialization TEST errors are retained for every network.

Sampled dense checks cover 301 capital values at EACH joint state. Maximum errors across models range from {pct(dense[:,0].min())} to {pct(dense[:,0].max())} for consumption, {pct(dense[:,1].min())} to {pct(dense[:,1].max())} for labour, and {pct(dense[:,2].min())} to {pct(dense[:,2].max())} for investment. These are sampled maxima, not continuous-domain certificates.

## Interventions with all weights available

For each first hidden unit, reduce activation by 1%, 10%, or 100% on 1,024 independent states. The local prediction uses all signed weights and the baseline second-layer ReLU gates. The reference actually changes the first-layer activation and reruns all downstream gates. Output responses are divided by each TRAIN output SD before pooling.

Width endpoint removal-prediction NRMSE by paired seed:

{chr(10).join('- Seed '+str(seed)+': '+' → '.join(pct(v) for v in values)+'.' for seed,values in endpoints.items())}

Final-hidden-unit removals supply a positive control: the signed readout prediction matches the actual edited forward pass to numerical precision. Full forward evaluation also resolves first-layer edits. Any failure is a failure of the specified local explanation. The network is fully available throughout.

## Labels and collateral effects

Each model's chosen final neuron has the strongest absolute productivity correlation on DISCOVERY. Remove it, then measure all three input-response vectors on independent INTERVENTION_TEST backgrounds. A response ratio greater than one indicates that removal strengthened the response norm. It need not increase every state's signed response.

| Width / seed | Unit | Productivity correlation | Productivity response retained | Capital response distortion | Labour-cost response distortion |
|---|---:|---:|---:|---:|---:|
{chr(10).join(labels)}

The unit selection sees no intervention-test results. These inputs are known economic coordinates. Correlation and collateral effects show a label's limits; they do not establish learned superposition or a complete semantic inventory.

## Economic consistency of neural outputs

The unconstrained network predicts all three outputs separately. Their joint first-order equations are checked on the first 1,024 TEST states, including the expected future neural policy at every shock state.

| Width / seed | Resource relative RMS | Labour FOC relative RMS | Euler relative RMS |
|---|---:|---:|---:|
{chr(10).join(diagnostics_rows)}

A low policy-fit error does not automatically impose the economic equations. These residuals are distinct from the numerical teacher's residuals. They remain visible in `richer_results.json` and this report.

## Evidence and limits

TRAIN=8,192/seed112001; VALIDATION=2,048/112002; DISCOVERY=2,048/112003; TEST=8,192/112004; INTERVENTION_TEST=1,024/112005. Capital is uniform on the core interval; shock states use the stationary product-chain distribution. Dataset hashes and raw-state disjointness are recorded. No real-world data, forecast test, or human understanding experiment is involved.

This is a larger, genuinely nonlinear DSGE with interacting choices. It gives a stronger testbed for incomplete neuron labels and local explanations. It does not prove intrinsic incomprehensibility or a general scaling law. The earlier hidden-readout theorem remains a separate information result.

## Reproduce

```
python3 -B verify_richer.py
node verify_richer_ui.js
python3 -B build_richer_page.py
```

Training and teacher generation overwrite their own result files; use separate copies to preserve this run before repeating them. Verification does not retrain. Browser-rendered visual QA remains incomplete. The saved runs used local CPU computation and no paid training.
'''
    (ROOT/'richer_report.md').write_text(report)
    print(f'Built richer.html ({len(page):,} characters), richer_report.md; {eligible}/9 fit eligible.')


if __name__=='__main__':
    main()
