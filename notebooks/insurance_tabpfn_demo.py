# Databricks notebook source
# MAGIC %md
# MAGIC # insurance-tabpfn: Foundation Models for Thin Insurance Segments
# MAGIC
# MAGIC **What this notebook shows:**
# MAGIC 1. Generate realistic thin-segment motor data (~400 policies)
# MAGIC 2. Fit `InsuranceTabPFN` with exposure handling
# MAGIC 3. Compare against Poisson GLM benchmark (Gini, deviance, double-lift)
# MAGIC 4. Extract PDP-based relativities
# MAGIC 5. Generate a committee-ready HTML report
# MAGIC
# MAGIC **When to use this library:** Segments with < 5,000 policies where you
# MAGIC need defensible relativities but don't have enough data for a stable GLM.

# COMMAND ----------

# MAGIC %pip install insurance-tabpfn[tabpfn,glm,report] statsmodels jinja2

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Synthetic thin-segment data
# MAGIC
# MAGIC Simulates a small motor scheme: ~400 young drivers, 18-25, with telematics
# MAGIC score and vehicle age. True DGP is Poisson frequency with known relativities.

# COMMAND ----------

import numpy as np
import pandas as pd

rng = np.random.default_rng(2024)

N = 400

# Features
age = rng.uniform(18, 25, size=N)
vehicle_age = rng.integers(0, 10, size=N).astype(float)
telematics_score = rng.uniform(30, 100, size=N)  # higher = safer
region = rng.choice(["London", "South", "Midlands", "North"], size=N, p=[0.35, 0.25, 0.25, 0.15])
exposure = rng.uniform(0.2, 1.0, size=N)  # policy years in force

# True relativities (what we want the model to learn)
base_rate = 0.12  # 12% annual claim frequency
age_rel = 1.0 + 0.03 * (25 - age)          # younger = higher risk
telem_rel = 1.2 - 0.005 * telematics_score  # better score = lower risk
veh_rel = 1.0 + 0.04 * vehicle_age         # older vehicle = higher risk
region_rel = {"London": 1.3, "South": 1.0, "Midlands": 0.95, "North": 0.85}
region_factor = np.array([region_rel[r] for r in region])

true_rate = base_rate * age_rel * telem_rel * veh_rel * region_factor
true_rate = np.clip(true_rate, 0.01, 0.5)

# Claim counts (Poisson)
claims = rng.poisson(true_rate * exposure)

X = pd.DataFrame({
    "age": age,
    "vehicle_age": vehicle_age,
    "telematics_score": telematics_score,
    "region": region,
})

print(f"Dataset: {N} policies, {claims.sum()} total claims")
print(f"Mean exposure: {exposure.mean():.2f} years")
print(f"Observed frequency: {claims.sum() / exposure.sum():.3f}")
print(f"\nFeatures:\n{X.describe()}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Train/test split

# COMMAND ----------

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test, exp_train, exp_test = train_test_split(
    X, claims.astype(float), exposure,
    test_size=0.25, random_state=42
)

print(f"Train: {len(X_train)} policies, {y_train.sum():.0f} claims")
print(f"Test:  {len(X_test)} policies, {y_test.sum():.0f} claims")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Fit InsuranceTabPFN

# COMMAND ----------

from insurance_tabpfn import InsuranceTabPFN

# backend="auto" will use TabICLv2 if available, otherwise TabPFN v2
# For this demo we explicitly use tabpfn (installed above)
model = InsuranceTabPFN(
    backend="tabpfn",
    device="cpu",
    conformal_coverage=0.9,
    conformal_test_size=0.2,
    random_state=42,
)

model.fit(X_train, y_train, exposure=exp_train)
print(f"Fitted backend: {model._backend.name}")
print(f"Training features: {model._n_features_in} (incl. log-exposure)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Predict and inspect

# COMMAND ----------

# Point predictions (expected claims)
pred_claims = model.predict(X_test, exposure=exp_test)

# 90% prediction intervals
lower, point, upper = model.predict_interval(X_test, exposure=exp_test, alpha=0.1)

# Quick calibration check
in_interval = (y_test >= lower) & (y_test <= upper)
coverage = in_interval.mean() * 100

print(f"Empirical coverage (90% target): {coverage:.1f}%")
print(f"Mean predicted claims: {pred_claims.mean():.4f}")
print(f"Mean actual claims:    {y_test.mean():.4f}")
print(f"Mean interval width:   {(upper - lower).mean():.4f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. GLM benchmark comparison

# COMMAND ----------

from insurance_tabpfn import GLMBenchmark

bench = GLMBenchmark()
bench.fit(X_train, y_train, exposure=exp_train)

comparison = bench.compare(
    X_test, y_test,
    tabpfn_predictions=pred_claims,
    exposure_test=exp_test,
    n_deciles=10,
)

print("=== Model Comparison ===")
display(comparison.to_dataframe())
print(f"\nHigher Gini: {comparison.winner()}")

# COMMAND ----------

# Double-lift chart
display(comparison.tabpfn.double_lift)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. PDP-based relativities

# COMMAND ----------

from insurance_tabpfn import RelativitiesExtractor

extractor = RelativitiesExtractor(
    model,
    n_grid_points=15,
    n_sample_rows=200,
    random_state=42,
)

# Extract relativities for each feature
factor_table = extractor.to_factor_table(X_train, exposure=exp_train)
print(f"Factor table: {len(factor_table)} rows across {factor_table['feature'].nunique()} features")
display(factor_table.head(20))

# COMMAND ----------

# Compare telematics score relativity vs true relativity
telem_relat = extractor.extract(X_train, "telematics_score", exposure=exp_train)

import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(telem_relat["feature_value"], telem_relat["relativity"], "b-o", label="PDP relativity")
# True relativity (normalised to 1 at median)
ts_grid = telem_relat["feature_value"].values
true_rel = (1.2 - 0.005 * ts_grid)
true_rel_norm = true_rel / true_rel.mean()
ax.plot(ts_grid, true_rel_norm, "r--", label="True relativity")
ax.set_xlabel("Telematics score")
ax.set_ylabel("Relativity")
ax.set_title("Telematics score: PDP vs true relativity")
ax.legend()
plt.tight_layout()
plt.savefig("/tmp/telematics_relativity.png", dpi=120)
plt.show()
print("Saved to /tmp/telematics_relativity.png")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Committee report

# COMMAND ----------

from insurance_tabpfn import CommitteeReport
from insurance_tabpfn.report import ReportConfig

config = ReportConfig(
    title="Young Drivers Telematics — Thin Segment Model",
    segment_name="Age 18-25, telematics scheme (<400 policies)",
    analyst="Pricing Team",
    notes="Proof of concept on synthetic data. Requires real data validation before committee submission.",
)

report = CommitteeReport(model, config=config)
report.add_benchmark(comparison)
report.add_relativities(factor_table)
report.add_coverage(lower, point, upper, y_test)

# Save HTML
html = report.to_html()
with open("/tmp/committee_report.html", "w") as f:
    f.write(html)

print(f"Report saved: /tmp/committee_report.html ({len(html):,} chars)")
print("\nJSON summary (first 500 chars):")
import json
data = json.loads(report.to_json())
print(json.dumps({k: v for k, v in data.items() if k != "limitations"}, indent=2)[:500])

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Key findings from this demo
# MAGIC
# MAGIC | Metric | TabPFN | Poisson GLM |
# MAGIC |--------|--------|-------------|
# MAGIC | Gini | See output | See output |
# MAGIC | Poisson deviance | See output | See output |
# MAGIC
# MAGIC **Limitations disclosed in committee report:**
# MAGIC 1. No true Poisson exposure offset
# MAGIC 2. Gaussian regression prior (not Poisson)
# MAGIC 3. Black-box ICL — PDP relativities are marginal approximations
# MAGIC 4. 5,000-policy thin-segment scope
# MAGIC 5. 10,000-row hard data limit
# MAGIC
# MAGIC **When TabPFN wins vs the GLM:**
# MAGIC - Very thin data (< 200 policies) where GLM coefficients are unstable
# MAGIC - Complex non-linear interactions (telematics features, geospatial)
# MAGIC - Sparse categorical features with many levels
# MAGIC
# MAGIC **When GLM wins:**
# MAGIC - Medium-large books (> 2,000 policies)
# MAGIC - Regulatory submissions requiring coefficient tables
# MAGIC - When actuary needs to manual-adjust specific factors
