# Original test-set evaluation - 25 September 2026

At the user's explicit request, the four development-selected primary models were scored on the original full test. No coefficients, spline knots, scales, penalties, history settings or splits were refitted or selected using these test outcomes.

Coverage: 378 images, 4137 selected trials, 8134 modeled choices and 10 participants. Fit/dev and previously evaluated pilot groups do not overlap these test images.

| Shared response | Shared T/D | Additional personal T/D |
| --- | ---: | ---: |
| Existing T, D, D^2 | -7.773718 | -7.769448 |
| Fixed spline response | -7.771750 | -7.767575 |

Scores are mean log2 probability per choice; higher is better. **All four models already have personal spatial controls.** Personal T/D adds only target/distance offsets.

| Contrast | Gain (bits/choice) | Exploratory 95% interval |
| --- | ---: | --- |
| Personal T/D under existing response | 0.004270 | [-0.000596, 0.009504] |
| Personal T/D under spline response | 0.004175 | [-0.000550, 0.009026] |
| Spline gain without personal T/D | 0.001968 | [-0.003846, 0.008197] |
| Spline gain with personal T/D | 0.001873 | [-0.003785, 0.007885] |
| Change in personal gain after adding splines | -0.000094 | [-0.000551, 0.000321] |

The gains are small, and every interval includes zero. The four main gain intervals also straddle the pre-existing 0.005 practical margin. Neither extra personal T/D nor shared splines show a reliable advantage here. This is not evidence that the models are equivalent or that people lack individual differences. Shared nonlinearity again does not visibly absorb the personal gain.

**Evidence boundary:** these images were held out from coefficient fitting and dev-based selection in this run, but the same test was already examined in the earlier COCO study. This is reused test evidence, not a fresh independent confirmation. The crossed subject/image bootstrap uses 2,000 replicates and conditions on the fixed selected models; it does not include training uncertainty.

**Verification:** the evaluation-only path reproduced all 40,052 saved primary dev predictions exactly before test scoring. Saved test arrays and all five contrast summaries were reloaded and checked; source and model-result hashes match. No training was run.

Reproduce only when explicitly intended: `audit_data.py test` refuses to overwrite an existing result. The default fit/dev loaders remain unchanged.

[Machine-readable result](test-evaluation.json) ; Per-choice prediction arrays are retained locally and excluded from the publication snapshot.
