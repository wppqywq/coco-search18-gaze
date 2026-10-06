# Hierarchical observer-specific factor weights, v2

25 September 2026. Authorized exploratory extension of the completed COCO study. This document supersedes the old protocol's no-MCMC boundary only for this separate analysis. Previously inspected development and test results remain known. Neither existing test set is accessed or described as fresh confirmation.

## Question and model

Which of target resemblance (T), saliency (S), movement distance (D), and decaying visit history (H) show stable observer-specific conditional weighting? Reuse the original 640-candidate conditional softmax, frozen maps, transition rules, fit/dev image groups, and 14 existing features. Retain shared D-squared, momentum, horizontal, first-distance and first-target controls. Retain five personal spatial controls. No spline, new feature, covariance matrix, PCA, clustering, duration model, new dataset, or freely generated scanpath claim.

For candidate a and known observer i:

    P(a | i, image, task, observed history) = softmax_a(sum_j w_ij X_j(a))
    w_ik = mu_k + sigma_k z_ik, z_ik ~ Normal(0, 1)
    k = T, S, D, H, x, y, x_squared, y_squared, xy
    w_ij = mu_j for the other five control terms
    mu_j ~ Normal(0, 1.5)
    sigma_k ~ HalfNormal(0.5)

All means and all nine scale parameters are inferred jointly. The four scientific scales have independent priors; there is no fitted random-effect covariance matrix. Spatial scales are nuisance parameters. Individuals are conditionally independent draws from the population distribution, without a sample sum-to-zero constraint. Thus population mu and the mean of these ten observers are different estimands. Noncentered or equivalent centered coordinates may be selected for numerical efficiency without changing this distribution. There is no softmax intercept because it is unidentifiable.

Features are centered within choice sets and RMS-scaled using the full fit set only, exactly as in the retained original feature builder. The same axes/scales and frozen image templates are used in all block fits to permit direct coefficient comparisons. A/B fits therefore test conditional stability across image groups, not independent end-to-end feature construction or cross-session reliability. Only ten people inform population variation. Weights are conditional sensitivities, not contribution percentages or causal mechanisms.

## Comparisons fixed before v2 fits

1. Full fit: four-factor hierarchy versus a matched shared-factor model. The latter retains the same five hierarchical spatial controls and all 14 population means, but has no personal T/S/D/H deviations.
2. The same two models fitted separately to the existing deterministic A/B fit-image blocks. Evaluate both on the same dev set, report dev A/B blocks, and score A-fit on B-fit and B-fit on A-fit as held-out-image checks. Those cross-block checks share the frozen full-fit feature preprocessing and are secondary.
3. Full four-factor hierarchy with scientific HalfNormal scales 0.25 and 1.0. Keep spatial scale priors at 0.5. These are prior sensitivities, not development-selected winners.

Eight main posterior fits in total. No test-driven choice of terms, priors, or results. Numerical pilot fits do not count as completed posterior results.

## Inference and verification

Use installed PyMC NUTS with exact full-data categorical likelihood and analytic gradients, float64 accumulation, no likelihood subsampling, variational approximation, or empirical-Bayes plug-in scales. A bounded-memory custom likelihood is allowed after independent likelihood/gradient checks. Start with real-data numerical checks and short multi-chain smoke/benchmark runs. Validate saved posterior reload. Engineering fixtures and simulated outcomes must remain explicitly separate from human observations.

The numerical implementation retains the original float32 feature values and computes every logit, probability, sum and gradient in float64. Installed Numba compiles the loops with `fastmath=False`; it does not change the likelihood. Fit-data Fisher information may initialize an adaptive NUTS mass matrix. This is sampler preconditioning, not a Gaussian approximation to the fitted posterior. Previous fitted weights supply dispersed numerical starting points only, never priors or fixed estimates in the new model.

Main default: four independently seeded chains, 1,000 warmup and 1,000 retained draws per chain, target acceptance 0.95, maximum tree depth 10. Adaptation and equivalent coordinates may be repaired based only on numerical diagnostics, never predictive gains. Require all population means, scale parameters and personal weights to have rank-normalized R-hat <= 1.01, bulk and tail ESS >= 400, zero post-warmup divergences, zero maximum-depth hits, and each chain's BFMI >= 0.3. Finite draws and a successful launch do not establish convergence. A failed gate stops substantive interpretation and dependent main runs until diagnosed; do not blindly extend failed chains.

## Deliverables and interpretation

- Posterior mean/median and 95% intervals for population means, all four scientific sigmas, and each observer's four weights and deviations. Preserve full posterior arrays and sampler diagnostics.
- Report posterior Pr(sigma > 0.05) as a descriptive, predeclared standardized-scale threshold, together with prior probabilities and sensitivity. A continuous positive prior makes Pr(sigma > 0) uninformative; a positive interval cannot establish a discrete nonzero effect. The 0.05 threshold is a project convention, not a psychological boundary.
- A/B posterior-mean correlations across ten observers and uncertainty from independent posterior draws, plus cross-block predictive performance. These are descriptive with a small observer sample.
- Posterior predictive dev probabilities average probabilities across draws, not coefficients or log probabilities. Compare models in bits/choice with the existing 2,000-replicate subject-by-image-group crossed bootstrap. Report posterior variation in the aggregate contrast separately from sampling uncertainty over observed data. The inherited 0.005 bits/choice margin is descriptive; imprecision is not equivalence.
- Conditional posterior predictive checks of distance, center tendency and revisit probability under observed histories, including person-level discrepancies. Do not claim validation of full scanpaths.
- A concise English report and coefficient/scale figure. Clearly separate numerical validation, synthetic recovery diagnostics if performed, human-data estimates, and unresolved limitations. No broad mechanism or person-type claims from prediction alone.

Implementation reference: [PyMC custom likelihood and gradient example](https://www.pymc.io/projects/examples/en/latest/howto/blackbox_external_likelihood_numpy.html). This supplies an interface pattern, not scientific validation of the model.

## Reproduction and current evidence

Use `<local-path>` for the new hierarchy. The older `../phd_proposal/.venv/bin/python` remains the existing regression-check runtime; it does not contain PyMC. Installed hierarchy versions are recorded in each run.json. No package installation is required.

```sh
python infer_hierarchical.py prepare  # Once only; refuses an existing cache.
python infer_hierarchical.py validate
python infer_hierarchical.py sample --name smoke-02 --max-per-subject 10 --tune 30 --draws 20 --chains 2
python infer_hierarchical.py sample --name benchmark-01 --tune 50 --draws 20 --chains 2
python -u infer_hierarchical.py run
```

The runner writes atomic progress and per-chain partial traces, then the complete posterior and diagnostics. It refuses to overwrite existing runs and stops at a failed diagnostic gate. `run` skips only completed, provenance-matched fits. `evaluate --name NAME` and `summarize` are available for explicit postprocessing. A partial trace, a smoke test, or a running process is not a completed scientific result. Full completion requires all eight gated fits, their predictions, stability/prior checks, and the final report.
