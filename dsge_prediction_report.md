# Frozen 387-parameter intervention-prediction pilot

Human participants completed: **0**. The planned pilot has 24 completed participants, six in each condition/form cell. No recruitment occurred. Automated calculations below measure the supplied algorithms, not human comprehension or difficulty.

The task asks for a selected output change after an edit to the frozen H16 seed 113 neural network. Answers use the TRAIN output standard deviation. They do not use the economic teacher's policy. Both conditions expose every weight, TRAIN normalization, DISCOVERY association label, and a generic forward simulator. Each case includes baseline inputs, outputs, and both layers' preactivations and activations. This information is mathematically sufficient to compute every answer. Full simulation is allowed in both conditions.

The mechanism guide adds the gate-update/readout explanation and two practice examples. Neither condition receives test answers or a dedicated test-answer button. Participants predict fresh intervention cases, not stored training or test rows. A/B forms match family and output only. No selection used edited effects. The practice cases are one H1 removal and one H2 removal.

The protocol was written before the generator ran. The new seeds are 88100–88104. Exact raw input pairs were checked against all old richer-study partitions and all new banks. Targets are sampled uniformly from baseline-active units. Incoming W2 edits also use baseline-active upstream H1 units. Attenuation removes 25%, 50%, or 100% of one post-ReLU activation. Weight edits add ±0.5 to a standardized-network weight. A sham removes 0%.

## Automatic reference results

The automation bank contains 1,200 cases: 300 per family, with 100 per selected output in each family. Every case remains in the calculations, including null effects and shams.

| Family | Zero-change MAE | Frozen-gate Jacobian MAE | Jacobian numeric accuracy |
|---|---:|---:|---:|
| H1 attenuation | 0.122442 | 0.024526 | 77.33% |
| Incoming weight | 0.106362 | 0.018502 | 86.00% |
| H2 attenuation | 0.205062 | <1e-15 | 100% |
| Sham | 0 | 0 | 100% |
| All | 0.108467 | 0.010757 | 90.83% |

Exact edited forward evaluation scores 100% on all cases. Direct H2 readout is exact to floating-point precision on the 300 H2 cases only. The frozen-gate Jacobian holds the baseline ReLU gates fixed. It can fail after H1 or incoming-weight edits change gates. Its overall direction accuracy is 98.25%. Direction uses a ±0.002 near-zero band. Numeric accuracy requires absolute error at most 0.02 SD. These results do not establish human difficulty or a guide effect.

## Blinded computational smoke check

A fresh agent received only the task packet, frozen protocol, and generic simulator. It completed all twelve form-B cases with exact simulation: mean absolute error 0 SD and numeric accuracy 100%. It did not read the generator, scorer, aggregate results, or saved answers. One scripted agent is not a human participant or a condition comparison. Its timing measures computation, not reading or comprehension. The response export stays local and is excluded from the release, since it contains form predictions.

## Verification

An independent verifier checks all 1,226 cases with separate NumPy equations. Maximum forward difference is 3.55×10⁻¹⁵. It checks unseen-state membership, parameter count, source hashes, allocations, balanced forms, practice-only answers, metric recomputation, valid scoring, eleven malformed-export rejections, duplicate exclusions, and participant-kind separation.

## Scoring and provenance

`dsge_prediction_tasks.json` contains complete canonical numeric weights and normalization, DISCOVERY labels, sampling state features, transition matrix, stationary probabilities, kss, blocked allocations, and all task inputs. Only the two practice rows contain answers and edited traces. Test and automation rows contain no edited answers. Aggregate automatic metrics are stored separately in `dsge_prediction_baselines.json`.

The response schema is frozen in `dsge_prediction_protocol.json`. Each export records participant kind, ID, allocation slot, condition, form, task order, responses in SD units, response times, tool use, materials time, total time, and protocol/task/model/label hashes. Use `python score_dsge_predictions.py participant.json --output score.json`. Use `python score_dsge_predictions.py --summarize participant*.json --output summary.json` for participant-unit cell means and family metrics. Human, demo, and agent-smoke groups remain separate. The scorer rejects mismatched hashes, allocations, incomplete or duplicate IDs, invalid numbers, or inconsistent times. Summary output lists every exclusion and rejects duplicate participant IDs and duplicate human slots.

The scorer recomputes edited forward values from the supplied model. It does not read a saved test answer key. Protocol, source, model, label, and task hashes support verification. All files remain local pending publication approval. Previous studies are preserved. No model was trained or refit.
