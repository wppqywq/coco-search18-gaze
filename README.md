# COCO-Search18 gaze modeling

A conditional next-fixation model for visual search. The study asks which shared gaze regularities help prediction and whether observers need different factor weights.

## Results

### 1. Shared factor weights

The Bayesian shared-factor model estimates common target, saliency, distance and history weights while allowing five spatial controls to vary across observers. Values are standardized posterior means with 95% credible intervals.

| Factor | Weight [95% credible interval] | Interpretation |
| --- | --- | --- |
| Target resemblance (T) | 1.258 [1.227, 1.290] | Locations resembling the target category are more likely to be selected. |
| Saliency (S) | 0.520 [0.506, 0.533] | Visual saliency has a positive association after controlling for other features. |
| Distance (D) | -1.099 [-1.152, -1.044] | Movement distance affects selection; interpret jointly with the squared term. |
| Distance squared | 0.554 [0.491, 0.617] | The distance response is curved rather than a constant penalty per unit distance. |
| Recent history (H) | -0.0245 [-0.0355, -0.0138] | A weak conditional tendency to avoid recently viewed regions. |

The full shared fit passed convergence checks (maximum R-hat 1.006; minimum bulk ESS 1,170). Its development score is -7.74453 bits/choice. The A-block shared fit also passed. The full personal T/S/D/H fit failed (R-hat 1.275; bulk ESS 11.15), so individual factor weights and population variation remain unresolved. These coefficients are associations, not causal effects or contribution percentages. [Saved estimates](reports/hierarchical/full-shared/evaluation.json) and [personal-fit diagnostics](reports/hierarchical/full-personal/run.json).

### 2. Term ablation

This is a separate experiment: a fully shared spline model with no personal parameters. Each row deletes one control and refits the remaining model. Positive prediction loss favors retaining the term.

| Deleted term | Dev loss, bits/choice [95% bootstrap interval] | Conclusion |
| --- | --- | --- |
| Horizontal preference | 0.04529 [0.02648, 0.06698] | Clear evidence for retaining this control. |
| Directional momentum | 0.02165 [0.00975, 0.03284] | Clear evidence for retaining this control. |
| First-distance correction | 0.00904 [-0.00130, 0.01844] | Potential benefit, but uncertain. |
| Vertical position (y) | 0.00324 [-0.00521, 0.01164] | No clear additional benefit. |
| Horizontal position (x) | 0.00160 [-0.00136, 0.00456] | No clear additional benefit. |
| First-target correction | 0.00048 [-0.00125, 0.00225] | No clear additional benefit. |
| Horizontal position squared (x^2) | -0.00013 [-0.00189, 0.00158] | Near-zero estimated benefit. |
| Vertical position squared (y^2) | -0.00008 [-0.00417, 0.00410] | Near-zero estimated benefit. |
| Position interaction (xy) | -0.00004 [-0.00078, 0.00071] | Near-zero estimated benefit. |

Intervals are exploratory and not adjusted for nine comparisons. Weak single-term effects do not establish that deleting several terms together is harmless. T/S/D/H were not individually removed in this ablation. [All ablation results](reports/shared-ablation.json).

An earlier 72-fit comparison found a small personal T/D gain on the reused test set: 0.00427 bits/choice [-0.00060, 0.00950]. The interval spans zero and the predeclared 0.005 practical threshold; practical benefit remains inconclusive. This penalized-model result does not resolve the failed Bayesian personal fit. [Comparison](reports/comparison.json) and [test evaluation](reports/test-evaluation.json).

## Data and preprocessing

The analysis uses the target-present (TP) subset of COCO-Search18: ten observers searching for 18 object categories in natural images. After transition filtering, the custom partitions contain 29,434 fitting, 10,013 development and 8,134 test choices.

- Read the public train/validation release using `split1`. Keep the supplied image/near-duplicate groups in [split.json](reports/frozen/split.json). Our test is a custom partition, not the official hidden benchmark test.
- Condition on the first recorded fixation. Stop a trial at an invalid or out-of-display sample without joining across the gap. Include the first fixation entering the target, then stop.
- Quantize the next fixation onto 640 equal-area cells (32 by 20) in the 1680 by 1050 display. Express geometry in image-width units.
- Build target templates from fitting-image crops only, excluding the current image for fitting maps. Center features within each candidate set and RMS-scale using fitting observations only.

Test outcomes were already inspected in earlier work and are not fresh confirmation. The two experiments above use fitting/development data only. Predictions condition on real fixation histories; they do not validate autonomous scanpaths or generalization to new observers.

## Model and features

For observer i, candidate a and feature j:

```text
score(i, a) = sum_j weight(i, j) * feature(a, j)
P(next fixation = a) = exp(score(i, a)) / sum_b exp(score(i, b))
```

The original model has 14 features and no intercept:

- **Target:** cosine resemblance to a category crop template using frozen ImageNet ResNet18 layer4 features; 448 by 280 input, interpolated to the candidate grid.
- **Saliency:** cached GBVS maps from 420 by 263 input, reduced to the candidate grid. The implementation is a locally corrected Python port.
- **Distance:** Euclidean distance from the current fixation, plus its square.
- **History:** Gaussian traces of earlier fixations, spatial width 0.06 image widths and exponential decay constant 0.5 seconds. Time uses cumulative fixation durations and excludes saccade time.
- **Controls:** x, y, x^2, y^2, xy; alignment with the previous movement; horizontal-versus-vertical movement preference; distance and target corrections on the first transition.

## Algorithms and experiment settings

**Bayesian hierarchy.** Personal slopes use partial pooling:

```text
mu[j] ~ Normal(0, 1.5)
sigma[k] ~ HalfNormal(0.5)
z[i, k] ~ Normal(0, 1)
weight[i, k] = mu[k] + sigma[k] * z[i, k]
```

The shared-factor baseline applies random slopes to the five spatial controls; the personal model adds T/S/D/H. Other weights equal mu. Each random slope has its own inferred scale, without a fitted between-factor covariance matrix. PyMC NUTS uses an exact Numba likelihood and analytic gradients, float64 accumulation, and a Fisher-initialized adaptive full mass matrix. Settings: four sequential chains, 1,000 warmup plus 1,000 retained draws per chain, target acceptance 0.95, maximum tree depth 10. Required diagnostics: R-hat <= 1.01, bulk/tail ESS >= 400, no divergences or maximum-depth hits, finite draws and BFMI >= 0.3. Prediction averages probabilities over retained draws. The planned eight-fit study is incomplete; A/B personal stability and prior-sensitivity conclusions are unavailable.

**Shared spline ablation.** Replace the shared target and distance responses with natural cubic bases at fit-only 5%, 35%, 65% and 95% quantiles, with explicit linear tails. Fit the full model and nine single-control deletions using SciPy L-BFGS-B, analytic gradients, a fixed shared L2 penalty of 0.01 and at most 500 iterations. All ten fits converged; there is no personal parameter or tuning search.

**Earlier penalized comparison.** Cross two response forms, two personalization levels, three personal penalties (1, 10, 100), three fitting subsets (full, A, B), and history on/off: 72 fits. Select penalties on development data. Every variant retains personal spatial controls. A/B assignments are deterministic image-group splits.

Predictive contrasts use mean log2 probability (higher is better) and 2,000 crossed observer/image-group bootstrap replicates. Intervals condition on fitted models and omit fitting/selection uncertainty. Detailed plans: [comparison protocol](protocol.md) and [hierarchical protocol](reports/hierarchical-protocol.md).

## Data access and reproduction

Download from the [official COCO-Search18 page](https://sites.google.com/view/cocosearch/home):

1. [TP train/validation fixations](https://vision.cs.stonybrook.edu/~cvlab_download/COCOSearch18-fixations-TP.zip): save as `data/COCOSearch18-fixations-TP.zip`, without extracting. The loader reads `coco_search18_fixations_TP_train_split1.json` and `coco_search18_fixations_TP_validation_split1.json` inside it.
2. [TP images](https://vision.cs.stonybrook.edu/~cvlab_download/COCOSearch18-images-TP.zip): needed for visual feature generation. Cite [Chen et al. (2021)](https://www.nature.com/articles/s41598-021-87715-9). The official links were checked, but archive downloads were not successfully verified.

**A complete fresh-machine rerun is not yet supported.** It also requires `data/processed/factor-maps-full.npz` and the exact locally corrected GBVS source at `vendor/gbvs-master/saliency_models/`. The repository supplies [feature checksums](reports/frozen/feature-audit.json), but no cache download or active feature-generation pipeline. GBVS source is excluded because the local snapshot has no license file; see [provenance](vendor/README.md). Historical arrays and exact source versions are additionally needed for full saved-result audits.

With those inputs restored, `infer_hierarchical.py prepare` builds the hierarchical cache and `validate` checks its numerical implementation. Fitting entry points are `fit_factors.py compare`, `ablate_shared.py` and `infer_hierarchical.py sample`; existing results are protected against overwrite. `requirements.txt` records the Python 3.11 numerical environment, not a clean-install lockfile. Local verification passed 15 regression checks in Python 3.10 and the hierarchical prediction check separately in Python 3.11.

## Code layout

- `coco_data.py`: frozen input checks and fit/dev loader.
- `fit_factors.py`, `ablate_shared.py`: feature design, penalized comparison and ablation.
- `infer_hierarchical.py`: hierarchical inference, diagnostics and evaluation.
- `analyze_factors.py`, `audit_data.py`, `test_factors.py`: data diagnostics and verification.
- `reports/`: numerical results, protocols and frozen manifests.
