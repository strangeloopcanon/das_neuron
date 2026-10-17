# Joint computation in the frozen economic networks

Every hidden unit has a qualified association tag and an exact weighted-input definition. All saved weights are retained. The tags do not specify the computation by themselves. Exact forward evaluation does.

No training, refitting, teacher solution, or publication occurred. The default is H16 seed 113, fixed before these diagnostic calculations.

The four shock corners have indices 10 (center), 19 (high productivity), 11 (high labor disutility), and 20 (both). Capital is held fixed. Labor disutility increases the utility cost of work; it is not a wage.

The analysis evaluates each of nine networks at 301 capital points in [0.5, 1.8] times steady-state capital. Three teaching examples use the nearest points to 0.6, 1.0, and 1.6. These are new descriptive diagnostics, not a held-out model-selection test.

## Exact accounting

`delta output[o] = TRAIN_sd[o] * sum_j W3[j,o] * (h2_target[j] - h2_base[j])`.

The largest absolute residual across every model, grid point, contrast, and interaction is 2.08e-16. Contributions can have opposite signs. Their gross sum and their net sum answer different questions.

Open path counts describe nonzero connections with open gates. They do not measure causal importance. A final-hidden contribution identifies a readout unit and aggregates its upstream paths.

## Fixed default teaching cases

| Capital/kss | Output | Joint net | Positive terms | Negative terms | NN interaction | Teacher interaction | Interaction error |
|---:|---|---:|---:|---:|---:|---:|---:|
| 0.5997 | consumption | +0.0379542 | +0.0940115 | -0.0560573 | +9.16124e-05 | -0.000644702 | +0.000736314 |
| 0.5997 | labor | -0.0221811 | +0.0305027 | -0.0526838 | +0.00074911 | +0.000470994 | +0.000278116 |
| 0.5997 | investment | +0.032474 | +0.0723715 | -0.0398975 | -0.00168583 | -0.000722673 | -0.000963157 |
| 0.9983 | consumption | +0.0403357 | +0.10048 | -0.0601443 | +8.96311e-05 | -0.000704896 | +0.000794527 |
| 0.9983 | labor | -0.0179798 | +0.0451937 | -0.0631735 | +0.000835384 | +0.000357602 | +0.000477782 |
| 0.9983 | investment | +0.0455243 | +0.092219 | -0.0466947 | -0.00147503 | -0.00107351 | -0.00040151 |
| 1.6007 | consumption | +0.044075 | +0.0703773 | -0.0263024 | +0.000676728 | -0.000762939 | +0.00143967 |
| 1.6007 | labor | -0.0142333 | +0.0268309 | -0.0410641 | -0.000962904 | +0.00023656 | -0.00119946 |
| 1.6007 | investment | +0.0584896 | +0.0747417 | -0.0162521 | -0.00186092 | -0.00154836 | -0.000312558 |

**Interaction fidelity fails:** exact neural accounting succeeds, but these networks do not accurately recover the teacher's small cross-shock interaction. Across the 301 capital points, default-model interaction error RMS / economic interaction RMS is consumption 1.84x, labor 2.76x, investment 1.06x. Sign agreement is consumption 35.9%, labor 57.1%, investment 81.1%. Every saved model has consumption and labor interaction error RMS greater than the corresponding economic interaction RMS. These are descriptive diagnostics of already selected checkpoints.


Interaction means `both - productivity - cost + baseline`. The teacher can have genuine interaction. The network can also have interaction approximation error. Report both; do not call all network interaction an economic effect.

## Route changes across capital

The top five routes rank final-hidden units by absolute contribution. This table compares the fixed low and high capital examples. Ties use ascending unit index.

| Shock | Consumption overlap | Labor overlap | Investment overlap |
|---|---:|---:|---:|
| productivity | 4/5 | 3/5 | 4/5 |
| cost | 1/5 | 1/5 | 1/5 |
| joint | 2/5 | 3/5 | 3/5 |

## Exploratory witness

A separate post hoc search finds productivity → labor at capital/kss=1.8: positive terms +0.0256479, negative terms -0.0260193, net -0.000371368. Gross/absolute-net=139.1. This witness was chosen for cancellation and is not a typical-case estimate.

## What these labels establish

Labels use all 2048 DISCOVERY examples (seed 112003). Each tag reports the largest absolute Pearson correlation with the three known inputs. Every correlation and activation frequency is available. Inactive or constant units have no defined Pearson correlation; their numeric placeholders are zero.

The same fixed tags accompany different gate patterns, signed magnitudes and route rankings across states. A tag therefore omits information needed for a finite shock calculation. This is a limitation of the tag, not proof that the circuit has no explanation.

The complete equations and weights reproduce every response. No human comprehension test, general hardness claim, size-scaling claim, or inference of polysemanticity follows from these measurements. The numerical teacher has checked first-order equations, not established global Bellman optimality.

## Reproduction and evidence

Run `python3 system_analysis.py` for local forward-only recomputation. The JSON includes SHA-256 hashes of the saved networks, protocol, teacher artifacts, and source files. It includes exact partition memberships and TRAIN normalization. No pre-existing evidence file is written.
