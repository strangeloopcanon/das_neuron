# A larger nonlinear economic circuit

The economy now includes endogenous labour, capital installation costs, and a second shock. Nine networks learn three policy outputs: consumption, labour, and investment. Open `richer.html` to inspect their decisions and actual finite interventions.

9/9 networks passed the three-output accuracy gate. The local removal-prediction error rose from the narrowest to widest network in 0/3 seeds. The width results are descriptive for this one model and explanation method. They do not prove a general increase in understanding difficulty.

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

The numerical solution uses 960 capital points and 21 joint shock states. Across 12,000 fresh states, the maximum relative Euler residual is 1.85e-06. Resource and labour residuals, grid refinement, and 10,000 simulated transitions pass the declared checks. These checks support a numerical first-order solution; they do not certify global Bellman optimality.

The solver uses endogenous-grid policy time iteration. It integrates future marginal values over all 21 joint states. Each policy query interpolates next capital, then solves the resource and labour equations. Capital is solved on [.08,4] reference capital; study states use [.5,1.8]. No extrapolation is allowed. The fresh core checks remain strictly inside the numerical capital boundaries.

Coarse-to-fine maximum differences, scaled by each output's TRAIN SD: consumption 0.02%, labour 0.04%, investment 0.10%. These are refinement diagnostics, not rigorous error bounds. The Euler residual is not a forecast or welfare score.

## Frozen neural sweep

Architecture: 3 inputs → H ReLU → H ReLU → 3 outputs. Every weight and bias is trained. Parameter counts are H²+8H+3: 387, 1,283, and 4,611. Seeds are 113, 229, and 347. No units are duplicated and no output weights are withheld.

The frozen `richer_protocol.json` precedes training. A 1,000-step pilot measured cost before nine 30,000-step local CPU fits. Saved total execution time including analysis was 31.46 seconds. Checkpoint selection uses the first scheduled VALIDATION checkpoint where every output's RMSE is at most 2% of its TRAIN SD. Failed models would remain visible. Default display is fixed at width32/seed113.

| Width | Seed | Parameters | Fit gate | Consumption TEST | Labour TEST | Investment TEST | Removal prediction error |
|---:|---:|---:|---|---:|---:|---:|---:|
| 16 | 113 | 387 | Pass | 1.16% | 1.64% | 1.85% | 39.29% |
| 16 | 229 | 387 | Pass | 1.14% | 1.45% | 1.93% | 38.84% |
| 16 | 347 | 387 | Pass | 1.08% | 1.45% | 1.84% | 40.95% |
| 32 | 113 | 1283 | Pass | 1.39% | 1.83% | 1.94% | 60.05% |
| 32 | 229 | 1283 | Pass | 1.49% | 1.80% | 1.87% | 50.20% |
| 32 | 347 | 1283 | Pass | 1.27% | 1.56% | 1.90% | 39.28% |
| 64 | 113 | 4611 | Pass | 1.42% | 1.74% | 1.80% | 38.68% |
| 64 | 229 | 4611 | Pass | 1.15% | 1.47% | 1.70% | 32.69% |
| 64 | 347 | 4611 | Pass | 1.08% | 1.46% | 1.90% | 34.59% |

Policy errors use each output's TRAIN standard deviation. They are not percentages of consumption, labour, or investment levels. The final column uses RMS actual intervention response as its denominator.

| Baseline | Consumption TEST | Labour TEST | Investment TEST |
|---|---:|---:|---:|
| affine | 8.32% | 14.61% | 23.91% |
| quadratic | 1.44% | 2.85% | 3.52% |

The quadratic baseline is a TRAIN-fitted degree-two polynomial with ten terms per output. These simple baselines matter: the economic policy can remain compact even if its neural implementation has many weights. Random-initialization TEST errors are retained for every network.

Sampled dense checks cover 301 capital values at EACH joint state. Maximum errors across models range from 6.97% to 16.95% for consumption, 10.82% to 26.21% for labour, and 16.89% to 30.06% for investment. These are sampled maxima, not continuous-domain certificates.

## Interventions with all weights available

For each first hidden unit, reduce activation by 1%, 10%, or 100% on 1,024 independent states. The local prediction uses all signed weights and the baseline second-layer ReLU gates. The reference actually changes the first-layer activation and reruns all downstream gates. Output responses are divided by each TRAIN output SD before pooling.

Width endpoint removal-prediction NRMSE by paired seed:

- Seed 113: 39.29% → 60.05% → 38.68%.
- Seed 229: 38.84% → 50.20% → 32.69%.
- Seed 347: 40.95% → 39.28% → 34.59%.

Final-hidden-unit removals supply a positive control: the signed readout prediction matches the actual edited forward pass to numerical precision. Full forward evaluation also resolves first-layer edits. Any failure is a failure of the specified local explanation. The network is fully available throughout.

## Labels and collateral effects

Each model's chosen final neuron has the strongest absolute productivity correlation on DISCOVERY. Remove it, then measure all three input-response vectors on independent INTERVENTION_TEST backgrounds. A response ratio greater than one indicates that removal strengthened the response norm. It need not increase every state's signed response.

| Width / seed | Unit | Productivity correlation | Productivity response retained | Capital response distortion | Labour-cost response distortion |
|---|---:|---:|---:|---:|---:|
| 16 / 113 | 6 | 0.570 | 93.11% | 8.48% | 20.62% |
| 16 / 229 | 8 | 0.652 | 76.85% | 23.91% | 23.57% |
| 16 / 347 | 9 | 0.748 | 90.93% | 35.46% | 32.93% |
| 32 / 113 | 20 | 0.716 | 72.94% | 7.80% | 15.58% |
| 32 / 229 | 21 | 0.666 | 85.46% | 6.47% | 16.30% |
| 32 / 347 | 2 | 0.850 | 88.21% | 1.93% | 4.75% |
| 64 / 113 | 20 | -0.829 | 110.48% | 11.14% | 26.21% |
| 64 / 229 | 17 | -0.810 | 90.43% | 8.71% | 11.16% |
| 64 / 347 | 20 | 0.866 | 92.89% | 4.90% | 9.69% |

The unit selection sees no intervention-test results. These inputs are known economic coordinates. Correlation and collateral effects show a label's limits; they do not establish learned superposition or a complete semantic inventory.

## Economic consistency of neural outputs

The unconstrained network predicts all three outputs separately. Their joint first-order equations are checked on the first 1,024 TEST states, including the expected future neural policy at every shock state.

| Width / seed | Resource relative RMS | Labour FOC relative RMS | Euler relative RMS |
|---|---:|---:|---:|
| 16 / 113 | 0.21% | 0.29% | 0.17% |
| 16 / 229 | 0.19% | 0.31% | 0.16% |
| 16 / 347 | 0.19% | 0.26% | 0.13% |
| 32 / 113 | 0.24% | 0.35% | 0.22% |
| 32 / 229 | 0.24% | 0.38% | 0.21% |
| 32 / 347 | 0.20% | 0.35% | 0.20% |
| 64 / 113 | 0.24% | 0.32% | 0.21% |
| 64 / 229 | 0.19% | 0.28% | 0.18% |
| 64 / 347 | 0.18% | 0.29% | 0.19% |

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
