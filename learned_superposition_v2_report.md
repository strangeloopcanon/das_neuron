# Learned economic response superposition, version 2

Full-width TEST controls pass: 5/5. Compressed TEST models pass: 0/5. Fixed study criterion: False.

The six-coordinate model has 42 trainable parameters. Its projected two-coordinate model has 18. Encoder and decoder share weights. Compressed models start from a random orthogonal projection of the trained full encoder, with decoder biases reset to zero, then fine-tune on TRAIN. No hexagon or sector directions are built in.

## Frozen protocol and data

Six known consumption-response features come from separate log-utility growth economies. The features reconstruct economic solution outputs, not raw states or policy functions. TRAIN/VALIDATION/TEST contain balanced singleton responses with amplitudes .8–1.2 and repeated zero baselines. All positive rows are disjoint. Pair and sparse tests are distribution shifts. No TEST results affect checkpoint, projection, optimizer, support or gates.

The compressed 25% active-error and leakage targets are declared operational criteria. Each feature must also have positive amplitude R² on its active examples. Constant feature classification is insufficient. Full-width gates are 2% active error/leakage, amplitude R²≥.95 and zero-output bound. All five VALIDATION controls must pass before compression. All outcomes are retained.

## All fixed seeds

| Stage | Seed | Parameters | TEST response RMSE | Largest active error | Largest leakage | Lowest active amplitude R² | TEST gate | Pair RMSE |
|---|---:|---:|---:|---:|---:|---:|---|---:|
| full | 1103 | 42 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | True | 0.2501 |
| full | 2207 | 42 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | True | 0.2848 |
| full | 3301 | 42 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | True | 0.2730 |
| full | 4409 | 42 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | True | 0.2594 |
| full | 5519 | 42 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | True | 0.3471 |
| compressed | 1103 | 18 | 0.3793 | 0.8551 | 0.1524 | -51.1587 | False | 0.7411 |
| compressed | 2207 | 18 | 0.3793 | 0.8494 | 0.1565 | -50.4696 | False | 0.7395 |
| compressed | 3301 | 18 | 0.3796 | 0.8506 | 0.1536 | -54.0784 | False | 0.7408 |
| compressed | 4409 | 18 | 0.3814 | 0.8494 | 0.1592 | -54.8129 | False | 0.7399 |
| compressed | 5519 | 18 | 0.3820 | 0.8537 | 0.1520 | -54.2099 | False | 0.7417 |

The default seed is 1103, fixed before training. Every seed is shown, including failures. Relative RMSE is sqrt(sum squared reconstruction error / sum squared actual response), not percentage error in economic consumption levels. Leakage uses the combined squared error of all five inactive outputs, divided by active-response squared amplitude. Active amplitude R² uses only the feature's active examples.

## Geometry, controls, and interventions

Six encoder columns represent the six independently controlled features. Selective reconstruction of all six from rank-two coordinates, with distinct signed directions and nonzero amplitudes, is the operational evidence for learned superposition. Antipodal directions are allowed. Raw neuron correlations alone are not the criterion.

The PCA and two-coordinate-retention references are fitted on TRAIN. Warm-start reconstructions are measured before fine-tuning. Each final model is also tested by adding and removing its feature direction in latent space, decoding and measuring target error and changes to unaffected outputs. Linear encoding equalities are implementation checks; decoded selectivity carries the measured content. Coordinate removals expose how individual latent coordinates participate in multiple features.

The economic teacher stays exactly additive. Pair errors are outside singleton training support, and the full-width controls can also err there. Compare the matched control before attributing a pair effect to compression. This is a test of learned response compression, not a measured human understanding gap or a proof of intrinsic difficulty.

## Execution and reproduction

Measured CPU training time: 5.754s. Pilot projection: 7.064s. No paid computation. No retries or outcome-guided hyperparameter changes.

```sh
python learned_superposition_v2.py --pilot
python learned_superposition_v2.py --run
```

Training writes only the version-2 result files; preserve the existing frozen evidence before rerunning. Independent verification is supplied separately.
