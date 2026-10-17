# Economic-response compression

The learned runs do not demonstrate six-feature compression. All three compressed seeds fail the frozen gates. The six-dimensional controls also fail. Several output units never recover their assigned feature, with 100% active-value error. This training failure prevents a clean conclusion about compression capacity.

Six separate log-utility growth economies use full capital depreciation. Each has alpha=0.36 and beta=0.96. Their exact policy gives c=(1-alpha beta)zk^alpha and next capital=alpha beta zk^alpha. At k=1, normalized consumption excess is x=z-1. There are no transfers or economic interactions.

The autoencoder compresses economic response vectors, not economic states or policies. TRAIN contains zeros and single-feature responses. Active responses range from 0.8 to 1.2 relative to baseline consumption. Held-out pairs test a distribution shift. Sparse tests use independent activity probability 0.05. All seeds and weights are retained. No retries or parameter search occurred.

| Width | Seed | Singleton relative RMSE | Pair relative RMSE | Sparse relative RMSE | Gates |
|---|---|---|---|---|---|
| 2 | 11 | 0.9137 | 0.9176 | 0.9215 | False |
| 2 | 23 | 0.7113 | 0.7399 | 0.7141 | False |
| 2 | 47 | 0.8167 | 0.8861 | 0.8377 | False |
| 6 | 11 | 0.7064 | 0.7037 | 0.6874 | False |
| 6 | 23 | 0.5730 | 0.6273 | 0.5762 | False |
| 6 | 47 | 0.8207 | 0.8463 | 0.8215 | False |

The default display uses seed 11. It is not selected by performance. Feature gates measure active-value error and combined inactive-output leakage. Both must be below 10% for every feature. The compressed model must reduce singleton MSE by at least 50% against both TRAIN-fit references. Geometry also requires rank two, nonzero columns, and distinct directions.

Latent additions and removals are decoded and checked against actual input edits. Their encoding equality follows from the linear encoder. It does not establish discovery of an independent mechanism. Decoded errors and unaffected-output changes show whether feature edits remain selective in context. Pair errors measure reconstruction failure on combined inputs; the economic teacher itself is additive.

This bounded experiment tests representation of a narrow, declared feature distribution. It provides no DSGE solution acceleration, general policy-learning result, or hardness theorem. The six-dimensional control and all sparse tests appear above.

## Per-seed diagnostic

Width 2, seed 11: 1/6 features meet both error gates; encoder rank 2; maximum absolute off-diagonal cosine 0.986846; pilot 0.028s, full 0.332s.

Width 2, seed 23: 3/6 features meet both error gates; encoder rank 2; maximum absolute off-diagonal cosine 1.000000; pilot 0.029s, full 0.333s.

Width 2, seed 47: 2/6 features meet both error gates; encoder rank 2; maximum absolute off-diagonal cosine 0.998548; pilot 0.028s, full 0.334s.

Width 6, seed 11: 3/6 features meet both error gates; encoder rank 6; maximum absolute off-diagonal cosine 0.534481; pilot 0.030s, full 0.348s.

Width 6, seed 23: 4/6 features meet both error gates; encoder rank 6; maximum absolute off-diagonal cosine 0.909350; pilot 0.029s, full 0.354s.

Width 6, seed 47: 2/6 features meet both error gates; encoder rank 6; maximum absolute off-diagonal cosine 0.846333; pilot 0.029s, full 0.355s.

The declared random initialization and ReLU decoder can leave output units inactive. Their gradients are then zero on the training samples. This is consistent with the observed dead features. These results cannot distinguish a poor training setup from insufficient representational capacity. No additional training was run after observing this failure.
