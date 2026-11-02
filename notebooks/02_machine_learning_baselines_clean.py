# Databricks notebook source
# MAGIC %md
# MAGIC # 02 — Machine Learning Baselines
# MAGIC ### Respiratory Health Intelligence | Version 1 historical reference
# MAGIC
# MAGIC **Purpose.** Reproduce and explain the **initial Version 1** model comparison on a frozen country-week dataset. This notebook is a concise, presentation-ready baseline chapter, not the production model, final model-selection study, or a request to repeat the historical experiments.
# MAGIC
# MAGIC **Important project decision (30 October 2026).** The project's selected final model is the **THREE_HEAD disease-aware Random Forest**, with separate influenza, RSV and COVID-19 heads and the maximum available disease-head score for weekly prioritisation. The Logistic Regression in this notebook remains an **earlier benchmark**. The final comparison and its specification belong to Notebooks 04–05.
# MAGIC
# MAGIC **Safety and reproducibility rules:**
# MAGIC - Source snapshot stays frozen at Delta version `0`; this file never writes, overwrites, registers or changes tables.
# MAGIC - Load only the `train` and `validation` splits, **never** the Version 1 `retrospective_check` split (479 rows).
# MAGIC - Median imputation, standardisation, and the single historical baseline fit use training data only.
# MAGIC - No hyperparameter search, new feature search, validation-based model selection or final holdout evaluation.
# MAGIC - Running the model cell below **will fit the original Logistic Regression baseline in memory**. Preparation did not execute Databricks or fit any model on the project dataset; only a local synthetic-data smoke test was used for code QA.
# MAGIC
# MAGIC **Source of record:** the untouched original `02_machine_learning_baselines (2).py` contains the full experiment history, historical source reconstructions, issue audits, one-time evaluation procedure and Delta writes. Do not run that original notebook end-to-end merely to reproduce this cleaned presentation.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Frozen Version 1 dataset and temporal split
# MAGIC
# MAGIC Snapshot `20261004_132520_0c36ef` contains **2,935 country-weeks**, **40 countries**, and **35 primary predictors**, using influenza, RSV and COVID surveillance.
# MAGIC
# MAGIC | Original split | Country-weeks | Use in this notebook |
# MAGIC |---|---:|---|
# MAGIC | Training | 2,127 | Fit imputation, scale factors, heuristic thresholds and Logistic Regression |
# MAGIC | Validation | 329 | Compare the already specified models without retuning |
# MAGIC | Later reserved evaluation | 479 | **Do not load or rerun** |
# MAGIC
# MAGIC Target `high_risk_next_4w`: future **elevated respiratory activity** within a 28-day outcome window (not an official outbreak declaration). The forecast origin is the Monday after the latest completed surveillance week. This is a retrospective study; historical publication dates are not proven by reference-time ordering alone.

# COMMAND ----------

import json
from importlib.metadata import version, PackageNotFoundError

import numpy as np
import pandas as pd
from pyspark.sql import functions as F

from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

SNAPSHOT_ID = "20261004_132520_0c36ef"
DELTA_VERSION = 0
TARGET = "high_risk_next_4w"
BASE_TABLE = f"humanitarian.model_ready.health_risk_v1_1_dev_{SNAPSHOT_ID}"

META = [
    "iso3", "week_start", "prediction_date", "outcome_start", "outcome_end", "split"
]
EXPECTED_SPLITS = {"train": 2127, "validation": 329, "retrospective_check": 479}

def read_frozen(table_name):
    """Read historical source by pinned Delta version, without writing to it."""
    return spark.read.option("versionAsOf", DELTA_VERSION).table(table_name)

manifest_rows = (
    read_frozen(BASE_TABLE + "_manifest")
    .select("manifest_json")
    .limit(2)
    .collect()
)
if len(manifest_rows) != 1:
    raise ValueError("Expected exactly one frozen Version 1 manifest.")

manifest = json.loads(manifest_rows[0]["manifest_json"])
expected = {
    "snapshot_id": SNAPSHOT_ID,
    "dataset_table": BASE_TABLE,
    "rows": 2935,
    "countries": 40,
    "predictors": 35,
    "target": TARGET,
    "split_rows": EXPECTED_SPLITS,
}
for field, value in expected.items():
    if manifest.get(field) != value:
        raise ValueError(f"Frozen manifest mismatch: {field}: {manifest.get(field)!r}")

FEATURES = list(manifest["primary_features"])
if len(FEATURES) != 35 or len(set(FEATURES)) != 35:
    raise ValueError("Expected exactly 35 distinct predictors.")
if set(FEATURES) & set(META + [TARGET]):
    raise ValueError("A metadata field or target was included as a predictor.")

print("VERSION 1 MANIFEST: PASS")
print("Snapshot:", SNAPSHOT_ID, "| Delta version:", DELTA_VERSION)
print("Predictors:", len(FEATURES))
for pkg in ("numpy", "pandas", "scikit-learn"):
    try:
        print(f"{pkg}: {version(pkg)}")
    except PackageNotFoundError:
        print(f"{pkg}: not available — resolve the environment before proceeding")

# COMMAND ----------

# The split filter is pushed down to Spark BEFORE any pandas collection.
# The 479 reserved observations are never collected.
development = (
    read_frozen(BASE_TABLE)
    .filter(F.col("split").isin("train", "validation"))
    .select(*(META + [TARGET] + FEATURES))
    .toPandas()
)

counts = development["split"].value_counts().to_dict()
if counts != {"train": 2127, "validation": 329}:
    raise ValueError(f"Unexpected development split counts: {counts}")

for column in ("week_start", "prediction_date", "outcome_start", "outcome_end"):
    development[column] = pd.to_datetime(development[column], errors="raise")

if development[META].isna().any().any():
    raise ValueError("Missing country/date/split metadata.")
if development.duplicated(["iso3", "prediction_date"]).any():
    raise ValueError("Duplicate country-forecast-date key.")
if not development[TARGET].isin([0, 1]).all():
    raise ValueError("Expected observed binary development labels.")

for feature in FEATURES:
    development[feature] = pd.to_numeric(development[feature], errors="raise")
if np.isinf(development[FEATURES].to_numpy(dtype=float)).any():
    raise ValueError("Infinite feature values are not permitted.")

if not development["week_start"].dt.dayofweek.eq(0).all():
    raise ValueError("Surveillance weeks must start on Monday.")
if not development["prediction_date"].eq(
    development["week_start"] + pd.Timedelta(days=7)
).all():
    raise ValueError("Unexpected forecast-origin convention.")
if not development["outcome_start"].eq(development["prediction_date"]).all():
    raise ValueError("Unexpected outcome start.")
if not development["outcome_end"].eq(
    development["prediction_date"] + pd.Timedelta(days=27)
).all():
    raise ValueError("Outcome windows must cover 28 calendar days.")

split_boundary = development["split"].map({
    "train": pd.Timestamp("2025-01-06"),
    "validation": pd.Timestamp("2025-07-07"),
})
if not development["outcome_end"].lt(split_boundary).all():
    raise ValueError("An outcome window crosses its split boundary.")

development = development.sort_values(["prediction_date", "iso3"]).reset_index(drop=True)
train = development.loc[development["split"].eq("train")].copy()
validation = development.loc[development["split"].eq("validation")].copy()

if train["iso3"].nunique() != 40:
    raise ValueError("Training cohort should contain 40 countries.")
if int(train[TARGET].sum()) != 629 or int(validation[TARGET].sum()) != 113:
    raise ValueError("Original training/validation label counts did not reproduce.")

X_train, y_train = train[FEATURES], train[TARGET].astype(int)
X_val, y_val = validation[FEATURES], validation[TARGET].astype(int)

print("DEVELOPMENT DATA QA: PASS")
print(development.groupby("split").agg(
    rows=("iso3", "size"), countries=("iso3", "nunique"),
    first_origin=("prediction_date", "min"), last_origin=("prediction_date", "max")
).to_string())
print("Training positives:", int(y_train.sum()))
print("Validation positives:", int(y_val.sum()))
print("Reserved evaluation rows were not loaded.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Activity target: meaning and limitations
# MAGIC
# MAGIC The original pipeline constructed country/stream-specific historical **80th-percentile activity thresholds** from a 2022–2024 calibration reference period and required sufficient source coverage to qualify streams. Influenza and RSV are measured through laboratory positivity; COVID is represented by `log1p` of reported weekly cases because the available files did not establish a compatible SARS-CoV-2 tested denominator.
# MAGIC
# MAGIC Version 1 defines a future positive label when at least one **locally evaluable eligible stream** becomes elevated in the following four weeks while the evaluable streams are not currently elevated. This is a surveillance-activity transition, not a clinical or WHO outbreak designation.
# MAGIC
# MAGIC **Archived QA, not newly rerun here:** the original notebook records 83 reproduced activity thresholds, plus three training-row target-semantics edge cases involving an elevated but locally omitted eligible stream. The historical issue review found these exceptions did not change the corrected four-fold ranking comparison. The frozen V1 snapshot was deliberately not relabelled.
# MAGIC
# MAGIC **Timing caveat:** the archived point-in-time register distinguishes temporal reference order from verified historical publication availability. Retrospective source revisions and uneven influenza/RSV/COVID reporting remain limitations.

# COMMAND ----------

# Small read-only consistency check; does NOT recalibrate thresholds.
thresholds = (
    read_frozen(BASE_TABLE + "_thresholds")
    .select("iso3", "stream", "threshold")
    .toPandas()
)
if len(thresholds) != 83:
    raise ValueError(f"Expected 83 saved V1 thresholds, found {len(thresholds)}")
if thresholds.duplicated(["iso3", "stream"]).any():
    raise ValueError("Duplicate saved country-stream thresholds.")
if thresholds["threshold"].isna().any():
    raise ValueError("Missing saved V1 threshold.")
if not np.isfinite(pd.to_numeric(thresholds["threshold"], errors="raise")).all():
    raise ValueError("Nonfinite saved V1 threshold.")
print("FROZEN TARGET THRESHOLD MANIFEST: PASS —", len(thresholds), "rows")
print("No threshold was estimated, overwritten or recalibrated.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Simple non-ML benchmark — unusually sharp recent rise
# MAGIC
# MAGIC For each disease stream, take the **75th percentile of strictly positive one-week activity changes in the training period**. A country-week is flagged when at least one current stream change exceeds its own training-derived cutoff. The ranking score is the maximum change-to-cutoff ratio across available streams. Missing observations are not converted into observed growth.
# MAGIC
# MAGIC This benchmark was specified without target labels and is kept fixed when applied to validation. It tests whether a linear model provides additional ranking value beyond obvious one-week increases.

# COMMAND ----------

CHANGE_FEATURES = {
    "flu": "flu_change_1w",
    "rsv": "rsv_change_1w",
    "covid": "covid_change_1w",
}
SHARP_RISE_QUANTILE = 0.75
sharp_rise_cutoffs = {}

for stream, feature in CHANGE_FEATURES.items():
    positives = X_train.loc[X_train[feature].notna() & X_train[feature].gt(0), feature]
    if positives.empty:
        raise ValueError(f"No positive training changes for {stream}.")
    cutoff = float(positives.quantile(SHARP_RISE_QUANTILE, interpolation="linear"))
    if not np.isfinite(cutoff) or cutoff <= 0:
        raise ValueError(f"Invalid sharp-rise training cutoff for {stream}.")
    sharp_rise_cutoffs[stream] = cutoff

def sharp_rise_predictions(X):
    relative = pd.DataFrame(index=X.index)
    for stream, feature in CHANGE_FEATURES.items():
        relative[stream] = X[feature] / sharp_rise_cutoffs[stream]
    score = relative.max(axis=1, skipna=True).fillna(0.0)
    prediction = score.gt(1.0).astype(int)
    return prediction.to_numpy(), score.to_numpy()

rule_train_pred, rule_train_score = sharp_rise_predictions(X_train)
rule_val_pred, rule_val_score = sharp_rise_predictions(X_val)
print("FIXED TRAINING-DERIVED SHARP-RISE CUTOFFS")
for stream, cutoff in sharp_rise_cutoffs.items():
    print(f"  {stream}: {cutoff:.6f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. First machine-learning baseline — Logistic Regression
# MAGIC
# MAGIC **Historical fixed specification:** 35 inputs; median imputation → standardisation → L2 Logistic Regression (`C=1.0`, `lbfgs`, `max_iter=5000`, no class weighting). Scikit-learn's preprocessing pipeline is fitted exclusively on the 2,127 training rows. The classification cutoff of `0.50` is **a reporting reference**, not a recommended operational alert policy. Probability ranking is evaluated separately through ROC-AUC and average precision.
# MAGIC
# MAGIC This recreates one **existing baseline**, without a new search or tuning procedure.

# COMMAND ----------

logistic_v1 = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(
        penalty="l2", C=1.0, solver="lbfgs", max_iter=5000
    )),
])

if list(X_train.columns) != FEATURES or list(X_val.columns) != FEATURES:
    raise ValueError("Predictor order differs from frozen manifest.")
if X_train.isna().all().any():
    raise ValueError("At least one original predictor is all-missing in training.")

# This fits the original benchmark IN MEMORY; it does not save or register a model.
logistic_v1.fit(X_train, y_train)
train_lr_prob = logistic_v1.predict_proba(X_train)[:, 1]
val_lr_prob = logistic_v1.predict_proba(X_val)[:, 1]

if not np.isfinite(val_lr_prob).all() or not np.all((val_lr_prob >= 0) & (val_lr_prob <= 1)):
    raise ValueError("Invalid validation Logistic Regression probabilities.")

train_lr_pred = (train_lr_prob >= 0.50).astype(int)
val_lr_pred = (val_lr_prob >= 0.50).astype(int)

print("ORIGINAL V1 LOGISTIC REGRESSION FIT: COMPLETE")
print("Fit rows:", len(X_train), "| Validation rows:", len(X_val))
print("Default classification threshold: 0.50 (diagnostic only)")
print("No hyperparameters were searched; no model artifact was written.")

# COMMAND ----------

def evaluate_binary_rule(y, pred, score, *, is_probability=False):
    y = np.asarray(y, dtype=int)
    pred = np.asarray(pred, dtype=int)
    score = np.asarray(score, dtype=float)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "rows": len(y), "positives": int(y.sum()), "alerts": int(pred.sum()),
        "accuracy": accuracy_score(y, pred),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
        "roc_auc": roc_auc_score(y, score),
        "average_precision": average_precision_score(y, score),
        "brier": brier_score_loss(y, score) if is_probability else np.nan,
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }

v1_validation_comparison = pd.DataFrame([
    {"model": "Sharp-rise heuristic", **evaluate_binary_rule(
        y_val, rule_val_pred, rule_val_score
    )},
    {"model": "Logistic Regression @ 0.50", **evaluate_binary_rule(
        y_val, val_lr_pred, val_lr_prob, is_probability=True
    )},
])

print("V1 VALIDATION — FIXED HISTORICAL BASELINES")
display(v1_validation_comparison.round(4))
print("Always-negative reference accuracy:", round(float(1 - y_val.mean()), 4))
print("Validation outcomes were used for evaluation, not to refit or tune.")

# COMMAND ----------

# Descriptive error analysis of a FIXED model; no threshold optimisation.
validation_diagnostics = validation[["iso3", "prediction_date", TARGET]].copy()
validation_diagnostics["probability"] = val_lr_prob
validation_diagnostics["prediction_at_050"] = val_lr_pred
validation_diagnostics["error_type"] = np.select(
    [
        (validation_diagnostics[TARGET].eq(1) & validation_diagnostics["prediction_at_050"].eq(1)),
        (validation_diagnostics[TARGET].eq(0) & validation_diagnostics["prediction_at_050"].eq(1)),
        (validation_diagnostics[TARGET].eq(1) & validation_diagnostics["prediction_at_050"].eq(0)),
    ],
    ["true_positive", "false_positive", "false_negative"],
    default="true_negative",
)
validation_diagnostics["month"] = validation_diagnostics["prediction_date"].dt.to_period("M").astype(str)

by_month = validation_diagnostics.groupby("month", as_index=False).agg(
    rows=(TARGET, "size"),
    observed_positive_pct=(TARGET, lambda x: 100 * x.mean()),
    mean_predicted_score=("probability", "mean"),
    positives_detected=("error_type", lambda x: int(x.eq("true_positive").sum())),
    positives_missed=("error_type", lambda x: int(x.eq("false_negative").sum())),
)

print("DEFAULT 0.50 THRESHOLD — VALIDATION ERROR COUNTS")
print(validation_diagnostics["error_type"].value_counts().to_string())
print("\nTEMPORAL DRIFT SNAPSHOT (descriptive, not a tuning grid)")
display(by_month.round(4))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Historical Version 1 results: what the experiments established
# MAGIC
# MAGIC These values are **previously recorded in the original notebook**, not results newly executed or independently verified during the cleanup. Re-run the clean validation cells above in Databricks to compare against the recorded benchmark before presenting regenerated figures.
# MAGIC
# MAGIC **Original 2025 validation:** the sharp-rise rule identified 29/113 positives (recall ≈25.7%, precision ≈39.2%, F1 ≈31.0%). Logistic Regression's ranking reached ROC-AUC ≈`0.748`, average precision ≈`0.535`, but its `0.50` cutoff identified only 14/113 positives. A retrospective validation-only F1 diagnostic suggested a cutoff near `0.17`; this was **not** a final alert threshold and must not be presented as independent generalisation performance.
# MAGIC
# MAGIC **Corrected fold-aware historical Logistic Regression** (eligibility and q80 reference reconstructed separately before each fold):
# MAGIC
# MAGIC | Historical fold | ROC-AUC | Average precision |
# MAGIC |---|---:|---:|
# MAGIC | 2023 H1 | 0.6203 | 0.3071 |
# MAGIC | 2023 H2 | 0.6630 | 0.4590 |
# MAGIC | 2024 H1 | 0.6541 | 0.2520 |
# MAGIC | 2024 H2 | 0.6908 | 0.3197 |
# MAGIC | **Unweighted fold mean** | **≈0.6570** | **≈0.3344** |
# MAGIC
# MAGIC This corrected reconstruction matters because an earlier rolling-origin analysis had **reused full-2022–2024 calibration/eligibility definitions** within older folds. That earlier analysis was only a conditional retrospective check and should not be substituted for the corrected fold-aware one.
# MAGIC
# MAGIC **Historical Version 1 nonlinear challenge:** Logistic Regression was retained over one fixed HistGradientBoosting challenger (mean ROC-AUC `0.6570` versus `0.6221`; mean AP `0.3344` versus `0.3111`). This was **only a Version 1 decision** and does not supersede the later disease-aware Random Forest selection.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Operational ranking and methodological safeguards — recorded audit conclusions
# MAGIC
# MAGIC The original issue-driven analyses established why a fixed-score country ranking was more defensible than presenting unstable probability cutoffs as calibrated binary outbreak alarms.
# MAGIC
# MAGIC | Audit | Recorded result | Interpretation |
# MAGIC |---|---|---|
# MAGIC | **PT-01** reporting delay | One-week mean ROC-AUC Δ `+0.0035`, mean AP Δ `+0.0039` versus the zero-delay simulation | Passed the predefined one-week sensitivity criterion; publication timestamps still unavailable |
# MAGIC | **EV-01** scoreability | 4,540 candidate country-weeks → 2,405 scoreable now → 1,846 onset-queue opportunities → 1,412 later evaluable (76.49% outcome coverage) | Future completeness must **not** decide who enters the forecast queue; unknown is not negative |
# MAGIC | **OP-01** weekly Top-5 | LR: 525 reviews, 400 known outcomes, 132 future elevations; heuristic: 525 reviews, 382 known, 93 elevations | LR improved observed precision (33.0% vs 24.3%) and future-positive capture (45.4% vs 32.0%) at equal workload |
# MAGIC | **MS-01** missingness | Stable flu+COVID / RSV+COVID cohorts audited; sparse three-stream cases had too little evidence | Report outcome coverage and stream composition; avoid unsupported subgroup claims |
# MAGIC | **CM-01** source comparability | No compatible SARS-CoV-2 positivity denominator was verified in accessible source files | Preserve the transparent COVID case-based definition; do not invent an inferred positivity ratio |
# MAGIC | **TG-01** target semantics | Three frozen training edge cases documented; no changes to the corrected replay sample from strict exclusion | Maintain provenance; no silent relabelling of frozen source |
# MAGIC | **HI-01** context | Humanitarian vulnerability metrics were developed separately | Do not conflate predicted respiratory elevation with humanitarian severity |
# MAGIC
# MAGIC **Operational conclusion:** retrospective prioritisation can support analyst review, but it is not a validated live early-warning system. Source timeliness, partial outcome verification, epidemiological differences, and lead-time limitations must be reported.
# MAGIC
# MAGIC The full computations and saved issue-register updates remain in the **untouched historical original**. They are not re-executed here, especially the large source replays, `saveAsTable(...)` operations, and one-time Version 1 reserved-set evaluation.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Baseline-to-final model handoff
# MAGIC
# MAGIC **This notebook proves the development history, not final model performance.** The Version 1 baselines motivated time-aware evaluation and analyst Top-K ranking. Later Version 2 and Version 3 research compared Random Forest and other challengers on different evaluation designs. Those evaluations must not be blended with the figures here.
# MAGIC
# MAGIC **Final selected model (recorded 30 October 2026):** `THREE_HEAD` disease-aware Random Forest (influenza, RSV and COVID heads; maximum available head score), preferred to RF59. The corrected later historical comparison recorded mean ROC-AUC `0.7187` for THREE_HEAD against `0.6917` for RF59 (difference `+0.0270`). That finding belongs in the **final model-selection / corrected-temporal-modelling notebook**, alongside the appropriate comparison protocol and limitations, not as a result produced by this Version 1 notebook.
# MAGIC
# MAGIC **Version 1 reserved data:** the original manuscript has a final-evaluation/freeze section for the 479 later observations. Its one-time execution must **not** be repeated while cleaning. The later V2/H1-2026 locked holdout is a separate evaluation boundary and should not be confused with those 479 V1 rows.
# MAGIC
# MAGIC **Next:** clean `03_v2_data_enrichment` to establish source-enrichment, engineering and handoff contracts; then clean Notebooks 04 and 05 to present selected final model, corrected comparisons, episode detection, MLflow/provenance and final limitations. Keep each original `.py` unchanged.

# COMMAND ----------

# In-memory reproduction summary for the clean Version 1 baseline chapter.
# No table writes; nothing here evaluates the reserved 479 rows.
reproduction_summary = pd.DataFrame([{
    "v1_snapshot": SNAPSHOT_ID,
    "frozen_delta_version": DELTA_VERSION,
    "training_rows": len(train),
    "validation_rows": len(validation),
    "features": len(FEATURES),
    "train_positives": int(y_train.sum()),
    "validation_positives": int(y_val.sum()),
    "logistic_fit": "training-only / original fixed baseline",
    "validation_only": True,
    "reserved_rows_collected": 0,
    "tables_written": 0,
    "project_final_model": "THREE_HEAD disease-aware Random Forest (separate later study)",
}])

print("NOTEBOOK 02 — CLEAN REPRODUCTION SUMMARY")
display(reproduction_summary)
print("Frozen Version 1 snapshot and all persisted Delta tables unchanged by this notebook.")
print("Do not rerun the original one-time reserved evaluation to tidy up the project.")
