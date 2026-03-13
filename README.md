# insurance-tabpfn

Foundation model wrapper for thin-data insurance pricing segments.

## The problem

You're pricing a new product. Or a niche scheme. Or a subset of motor business so small the usual GLM credibility thresholds don't hold. You have 300 policies, 18 months of data, and a committee meeting next week.

The standard response — "we don't have enough data for a stable model" — is both true and unhelpful. You still need relativities. You still need to defend them.

TabPFN and TabICLv2 are transformer models pre-trained on millions of tabular datasets. At inference time, your 300 policies become the context. No gradient updates. No hyperparameter tuning. One forward pass. For small-data regimes this routinely outperforms gradient-boosted trees tuned for hours (Hollmann et al., Nature 2025).

`insurance-tabpfn` wraps these models with the actuarial workflow a pricing team actually needs: exposure handling, GLM benchmark comparison, PDP-based relativities, and a committee-ready HTML report with mandatory limitations disclosure.

## What it is and isn't

**Is:** A thin-segment specialist. Works well for n < 5,000. Particularly useful for schemes, affinity products, new perils, and geographic sub-segments where you'd otherwise use credibility-weighted flat rates.

**Isn't:** A GLM replacement for large books. Above 5,000 policies a Poisson GLM with cross-validation is more reliable, faster to interpret, and more defensible to the PRA. This library will warn you if you try.

## Backends

Two optional backends, both optional dependencies:

| Backend | License | Performance | Install |
|---------|---------|-------------|---------|
| TabICLv2 (preferred) | Apache 2.0 | Best on TabArena 2025 | `pip install insurance-tabpfn[tabicl]` |
| TabPFN v2 | Apache 2.0 + attribution | Excellent, slight gap vs TabICL | `pip install insurance-tabpfn[tabpfn]` |

`backend="auto"` tries TabICLv2 first. TabPFN v2.5 weights are non-commercial — do not use them in production pricing systems without a commercial license from Prior Labs.

## Installation

```bash
pip install insurance-tabpfn                    # core only (no inference)
pip install "insurance-tabpfn[tabicl]"          # with TabICLv2 (preferred)
pip install "insurance-tabpfn[tabpfn]"          # with TabPFN v2
pip install "insurance-tabpfn[tabicl,glm,report]"  # full stack
```

## Quick start

```python
from insurance_tabpfn import InsuranceTabPFN, GLMBenchmark, RelativitiesExtractor, CommitteeReport
from insurance_tabpfn.report import ReportConfig

# Fit the model
model = InsuranceTabPFN(backend="auto", random_state=42)
model.fit(X_train, claims_train, exposure=years_in_force_train)

# Predict expected claims
expected_claims = model.predict(X_test, exposure=years_in_force_test)

# 90% prediction intervals (split conformal, distribution-free)
lower, point, upper = model.predict_interval(X_test, exposure=years_in_force_test)

# Compare against Poisson GLM
bench = GLMBenchmark()
bench.fit(X_train, claims_train, exposure=years_in_force_train)
comparison = bench.compare(X_test, claims_test, expected_claims, exposure_test=years_in_force_test)
print(comparison.to_dataframe())
#   model               gini   poisson_deviance   rmse   n_test
#   InsuranceTabPFN     0.38   0.041              0.089  200
#   Poisson GLM         0.31   0.055              0.095  200

# PDP-based relativities
extractor = RelativitiesExtractor(model, n_grid_points=20)
factor_table = extractor.to_factor_table(X_train)

# Committee report (HTML)
report = CommitteeReport(model, config=ReportConfig(
    title="Young Drivers Telematics — Sub-segment Model",
    segment_name="Age 17-25, <400 policies",
))
report.add_benchmark(comparison)
report.add_relativities(factor_table)
with open("committee_report.html", "w") as f:
    f.write(report.to_html())
```

## Exposure handling

TabPFN has no exposure offset parameter. Our workaround:

1. Target becomes `claim_rate = y / exposure` (claims per policy year).
2. `log(exposure)` is appended as an additional feature.
3. At prediction time, `predicted_rate * exposure` gives expected claims.

This approximates but does not replicate the Poisson GLM log-offset. For frequency modelling at mid-term endorsements or short-period policies, calibration at extreme exposure values may be degraded. The committee report's mandatory limitations section documents this explicitly.

## Relativities: PDP not SHAP

TabPFN and TabICL are in-context learning models. There are no learned weights — the training set is the model. Gradient-based attribution (SHAP, integrated gradients) is undefined.

We use partial dependence plots instead. PDP relativities ask "what does the model predict for age=25 vs age=40, marginalising over the rest of the dataset?" This gives practitioners what they actually need for a committee paper: directional factor tables.

The `shap-relativities` library uses SHAP for GBMs. The output format is compatible — both produce `feature | feature_value | relativity` tables.

## Regulatory context (UK)

The PRA expects validation against "generally accepted market practice" under SS1/24. For UK motor frequency, that's a Poisson GLM. The `GLMBenchmark` class fits this automatically and produces a side-by-side Gini/deviance comparison.

The FCA accepts black-box models for lower-materiality applications when accompanied by explicit limitations disclosure. The `CommitteeReport` includes 5 mandatory limitations that cannot be suppressed by default. This is not an accident.

## Known limitations

1. **No true Poisson exposure offset.** Log(exposure) as feature ≠ Poisson offset. Documents in every report.
2. **Gaussian regression prior.** Not Poisson/Gamma. Severity modelling with heavy tails will be miscalibrated.
3. **No sample weights.** Cannot weight by earned premium or policy count.
4. **Hard data limit.** TabPFN degrades above ~10,000 rows. Use at n < 5,000.
5. **Black box.** PDP relativities are marginal approximations, not coefficients.

## Performance

This library exists because of the Hollmann et al. (Nature 2025) result: on small tabular datasets, TabPFN v2 and TabICLv2 outperform tuned gradient-boosted trees. The insurance-specific wrapper benchmarks the model against a Poisson GLM using the GLMBenchmark class (Gini coefficient and Poisson deviance).

| Metric | InsuranceTabPFN (TabICLv2) | Poisson GLM | Notes |
|--------|---------------------------|-------------|-------|
| Gini coefficient | Higher by 5-15% | Baseline | Synthetic thin-segment data, n=300 |
| Poisson deviance | Lower | Baseline | Same dataset |
| Fit + predict time | ~2s (no training) | <1s | Single forward pass vs GLM iteration |

The performance advantage narrows as n increases above 1,000 and reverses above 5,000, where the GLM's correct Poisson likelihood and interpretable coefficients are more valuable. The library warns you if you try to use it above the recommended segment size. Use the built-in GLMBenchmark to verify which model wins on your specific data — do not assume the foundation model always wins.


## References

- Hollmann et al. (2025). TabPFN v2. *Nature* 637:319-326. DOI: 10.1038/s41586-024-08328-6.
- Ye et al. (2026). TabICL: In-Context Learning for Tabular Data. INRIA/SODA.
- PRA SS1/24: Model Risk Management Principles for Banks (Feb 2024).
