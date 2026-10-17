# Predicting edits with every weight available

The main model has 387 parameters: 352 weights and 35 biases. It uses two hidden layers with 16 rectified linear units each. A rectified linear unit (ReLU) outputs zero for a negative input and passes a positive input unchanged.

The model learns consumption, work, and investment from a nonlinear dynamic stochastic general equilibrium (DSGE) solution. Productivity and work-discomfort shocks affect decisions. Convex capital installation costs connect current choices to future capital. The neural circuit approximates those decision rules. Its edited outputs are the targets of this study; an edited circuit need not solve the economic equations.

## Run a session

1. Open `dsge_prediction.html`. No server or account is needed.
2. Use demo mode to inspect the task. Demo exports never count as human observations.
3. For a study session, the coordinator assigns one unused slot from 1 to 24. Use a pseudonymous ID.
4. Read the complete model packet and complete two practice tasks. Practice provides feedback.
5. Predict twelve signed output changes. A positive number means an increase. Use training-output standard deviation units, not percentages.
6. Record the tools used. A calculator, code, and full simulation are allowed in both conditions.
7. Submit each prediction once. Scored questions provide no feedback during the session.
8. Download the final JSON export. The page sends nothing to a server. Give the file to the coordinator through an agreed channel.

The two conditions receive identical weights, labels, normalization, baseline activations, and generic simulation code. One condition also receives a guide to gate changes and signed output contributions. Conditions use two balanced forms. Labels report measured input correlations on separate DISCOVERY states. They are association labels, not exclusive economic meanings.

## Coordinate and score

Do not show participants aggregate results, another participant's export, the scorer, or the report before completion. The full model remains sufficient to compute every answer. Blinding withholds feedback, not mathematical information.

The frozen allocation has six randomized blocks of four condition/form cells. Use each human slot once. Replace incomplete sessions with the same unfilled slot. Do not select participants or exclude sessions based on their scores. Keep anonymous exports outside the public release. No recruitment has occurred.

```sh
python score_dsge_predictions.py participant.json --output score.json
python score_dsge_predictions.py --summarize participant-*.json --output summary.json
```

Keep the task packet, protocol, and export together when scoring. The scorer checks hashes, allocation, completeness, timing, and finite numbers. It computes edited network outputs directly. Pooled scoring separates human, demo, and agent smoke sessions. It flags duplicate participant IDs and human slots.

Primary outcome: each participant's mean absolute error in training-output standard deviations. Report direction accuracy, numeric accuracy, time, tool use, and all four edit families. Numeric accuracy allows an absolute error of 0.02 standard deviations. Direction uses a ±0.002 near-zero band. The planned 24-person pilot is descriptive. Human participants are the analysis units.

## What a result would mean

Low accuracy with the complete packet would show difficulty using these materials under these study conditions. Better accuracy or lower time with the guide would support the value of that guide. Neither result would prove that system understanding is impossible or that labels have complete semantic meaning.

High accuracy would be informative too. Participants may reconstruct each response with code. The full model already determines every answer. Exact simulation is the reference result, not a prohibited shortcut.

The automated bank contains 1,200 new cases. It is separate from the 24 scored form questions and two practice tasks. All source networks, teacher data, original training results, and earlier superposition studies remain preserved. No network was trained or enlarged for this experiment.
