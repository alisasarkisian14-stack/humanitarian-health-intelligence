# Databricks notebook source
# MAGIC %md
# MAGIC # 03 — Version 2 Data Enrichment and Feature Engineering
# MAGIC
# MAGIC **Respiratory Health Intelligence | Archived Version 2 enrichment evidence | Clean, read-only presentation copy**
# MAGIC
# MAGIC **Purpose.** Explain and verify the accepted source-quality decisions, past-only respiratory dynamics, historical evaluation contracts and persisted enrichment closeout. This is a reproducible **read-only evidence chapter**, not the original table-building pipeline or the final selected model.
# MAGIC
# MAGIC **Project sequence.** `01_exploratory_data_analysis_CLEAN` → `02_machine_learning_baselines_CLEAN` → **this notebook** → `04_final_model_selection_CLEAN` → `05_corrected_temporal_modelling_CLEAN`.
# MAGIC
# MAGIC **Critical version distinction.** On 13 October 2026, the enrichment stage carried forward **RF59_28DAY_REGION_LAG2_MEDIAN** as a **development candidate**, explicitly *not* a frozen final model. Later corrected model selection (30 October 2026) selected **THREE_HEAD**, a disease-aware Random Forest with influenza/RSV/COVID heads. This notebook must not restore the earlier RF59 candidate as the final winner.
# MAGIC
# MAGIC **Safety contract**
# MAGIC - Read existing, versioned development tables only; do not recompute or overwrite saved data, issue registers, models or scores.
# MAGIC - Do not load H1-2026 holdout observations, labels or predictions. Read **only** the single governance lock's `status` field.
# MAGIC - Do not retrain, retune, promote or rerun historical model selections to clean the project.
# MAGIC - Preserve the original `03_data_enrichment.py` as the full experimental provenance, including source reconstructions, spatial experiments, attempted enrichments and one-time Delta writes.
# MAGIC - The numeric historical findings below are *archived* results until the corresponding read-only checks pass in Databricks.

# COMMAND ----------

import json
import numpy as np
import pandas as pd

SCHEMA = "humanitarian.model_ready"
FEATURE_TABLE = SCHEMA + ".health_risk_v2_dynamic_features_20261005_174033"
FEATURE_MANIFEST = FEATURE_TABLE + "_manifest"
REGISTER_TABLE = SCHEMA + ".health_risk_v2_research_register"
CLOSEOUT_TABLE = SCHEMA + ".health_risk_v2_data_enrichment_closeout_20261013_134936"
LOCK_TABLE = SCHEMA + ".health_risk_v2_holdout_lock"
REGION_FOLD_METRICS = SCHEMA + ".health_risk_v2_region_delay_20261012_135524_fold_metrics"
HORIZON_COMPARISON = SCHEMA + ".health_risk_v2_rolling_regionlag2_20261012_140742_horizon_comparison"
SCORED_QUEUE = SCHEMA + ".health_risk_v2_diag01_20261012_151124_scored_queue"
KNOWN_PREDICTIONS = SCHEMA + ".health_risk_v2_diag04_28day_vs_hmax_20261012_160846_matched_predictions"
TOP5_REVIEWS = SCHEMA + ".health_risk_v2_diag05_28d_warning_burden_20261012_161651_top5_reviews"
EPISODE_DETAIL = SCHEMA + ".health_risk_v2_diag06_fold_failure_20261012_163909_episode_detail"
FOLDS = ["2023_H1", "2023_H2", "2024_H1", "2024_H2"]

# Reads from the protected holdout are forbidden; only its status lock is checked.
def read_development_table(table_name):
    if "holdout" in table_name.lower() or "2026_h1" in table_name.lower():
        raise ValueError("Refusing to read a holdout data table.")
    return spark.table(table_name)

lock = spark.table(LOCK_TABLE).select("status").limit(2).collect()
if len(lock) != 1 or lock[0]["status"] != "LOCKED_UNTOUCHED":
    raise RuntimeError("H1-2026 lock status is not LOCKED_UNTOUCHED; stop and investigate.")

closeout = read_development_table(CLOSEOUT_TABLE).limit(2).toPandas()
if len(closeout) != 1 or closeout.loc[0, "stage"] != "DATA_ENRICHMENT_COMPLETE":
    raise RuntimeError("Missing or inconsistent archived enrichment closeout.")

print("NOTEBOOK 03 PRE-FLIGHT: PASS")
print("Completed enrichment stage:", CLOSEOUT_TABLE)
print("H1-2026 governance status: LOCKED_UNTOUCHED")
print("No holdout observations, labels or predictions loaded.")
print("No model fitting or table writes.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Data-source quality and eligibility decisions
# MAGIC
# MAGIC The original Version 2 audit considered extending combined respiratory modelling to **2020–2021**. This was **rejected** as routine additional training history because influenza/RSV surveillance was insufficiently complete. Older history was investigated for separate seasonal references instead; a **2015–2019 pre-pandemic seasonal bridge was not approved** as a robust general feature.
# MAGIC
# MAGIC The project continued to use:
# MAGIC
# MAGIC | Stream | Recorded activity definition | Important cleaning constraint |
# MAGIC |---|---|---|
# MAGIC | Influenza | WHO FluNet *sentinel* laboratory positivity | No invented influenza zeros from blank counts; require valid positive and processed totals |
# MAGIC | RSV | WHO FluNet sentinel RSV positivity | Nonzero tested denominator; positives cannot exceed tested |
# MAGIC | COVID-19 | `log1p` of nonnegative weekly WHO reported cases | Negative reporting corrections remain unavailable; no invented SARS-CoV-2 positivity denominator |
# MAGIC
# MAGIC These sources measure **different surveillance quantities**, which is a limitation, not an invitation to create unsupported conversions. `HUMAN_CORONA` is not SARS-CoV-2. FluID ILI/SARI data and influenza subtype composition were audited but **not retained as primary predictors** because of coverage/operational-performance limitations.
# MAGIC
# MAGIC The original source-processing code remains archived. The executable cells in this clean chapter inspect **already saved development evidence**, rather than downloading or overwriting surveillance extracts.

# COMMAND ----------

research = read_development_table(REGISTER_TABLE).toPandas()
required = {"task_id", "status", "task"}
if not required.issubset(research.columns):
    raise ValueError(f"Research-register columns missing: {sorted(required - set(research.columns))}")
if research["task_id"].duplicated().any():
    raise ValueError("Duplicate research task IDs in saved register.")
required_tasks = {"V2-DATA-01", "V2-FEAT-01", "V2-FEAT-02", "V2-SPATIAL-01"}
missing = sorted(required_tasks - set(research["task_id"]))
if missing:
    raise ValueError("Missing expected enrichment tasks: " + ", ".join(missing))

research_topics = [
    "V2-DATA-01", "V2-FEAT-01", "V2-FEAT-02", "V2-SPATIAL-01",
    "V2-FLU-01", "V2-COVID-01", "V2-SYNDROMIC-01",
]
research_review = research.loc[
    research["task_id"].isin(research_topics),
    [c for c in ["task_id", "task", "status", "objective", "closure_criterion"]
     if c in research.columns]
].sort_values("task_id")
print("ARCHIVED ENRICHMENT DECISIONS — READ ONLY")
display(research_review)
print("Research register was not modified.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Past-only dynamic features: exact construction contract
# MAGIC
# MAGIC The approved feature-engineering approach builds each country's activity history in **chronological order** and computes reference statistics from **strictly earlier non-missing observations**. A minimum of **26 earlier observations** is required for the historical 80th-percentile reference and percentile rank. The current observation is inserted into the historical reference **only after** its own features have been calculated.
# MAGIC
# MAGIC For each influenza, RSV and COVID-19 stream, the original code constructed **nine candidates**:
# MAGIC
# MAGIC | Candidate | Definition / interpretation |
# MAGIC |---|---|
# MAGIC | `threshold_distance` | Current activity minus past-only q80 |
# MAGIC | `threshold_ratio` | Current activity divided by past-only q80, only when the denominator is positive |
# MAGIC | `historical_percentile` | Fraction of earlier non-missing observations at or below the current value |
# MAGIC | `slope_4w` | Four-week least-squares trend over the current and previous three values |
# MAGIC | `acceleration_1w` | Current change minus previous change |
# MAGIC | `consecutive_rising_weeks` | Number of immediately consecutive rising week-to-week transitions, at most four |
# MAGIC | `max_4w`, `min_4w` | Complete current-plus-three-previous-week extrema |
# MAGIC | `volatility_4w` | Population standard deviation over those four completed weeks |
# MAGIC
# MAGIC **27 candidates = 9 × 3 streams.** The sparse `threshold_ratio` family (three columns) was rejected. **24 robust past-only features** were retained as a candidate addition to the **35 original predictors**, giving **RF59 = 35 + 24**. This is an earlier *feature-family* decision, not the later THREE_HEAD final model specification.
# MAGIC
# MAGIC The archived construction passed a future-mutation/invariance check; this clean notebook verifies the persisted snapshot and its numeric constraints without rebuilding it.

# COMMAND ----------

STREAMS = ["flu", "rsv", "covid"]
ROBUST_SUFFIXES = [
    "threshold_distance", "historical_percentile", "slope_4w",
    "acceleration_1w", "consecutive_rising_weeks",
    "max_4w", "min_4w", "volatility_4w"
]
ROBUST24 = [f"{s}_{suffix}" for s in STREAMS for suffix in ROBUST_SUFFIXES]
RATIO3 = [f"{s}_threshold_ratio" for s in STREAMS]
DYNAMIC27 = [f"{s}_{suffix}" for s in STREAMS
             for suffix in ["threshold_distance", "threshold_ratio"] + ROBUST_SUFFIXES[1:]]
assert len(ROBUST24) == 24 and len(DYNAMIC27) == 27
assert len(set(DYNAMIC27)) == 27 and set(DYNAMIC27) == set(ROBUST24 + RATIO3)

manifest_rows = read_development_table(FEATURE_MANIFEST).select("manifest_json").limit(2).collect()
if len(manifest_rows) != 1:
    raise ValueError("Expected exactly one V2 dynamic-feature manifest.")
feature_manifest = json.loads(manifest_rows[0]["manifest_json"])
if feature_manifest.get("candidate_feature_count") != 27:
    raise ValueError("Manifest no longer states 27 candidate dynamic features.")
if feature_manifest.get("countries") != 40:
    raise ValueError("Manifest country scope is not the archived 40-country cohort.")
if feature_manifest.get("minimum_past_reference_observations") != 26:
    raise ValueError("Past-only reference-support rule has changed.")
if set(feature_manifest.get("features", [])) != set(DYNAMIC27):
    raise ValueError("Candidate feature names do not match the archived manifest.")

feature_columns = (
    ["iso3", "week_start"]
    + [f"{s}_activity" for s in STREAMS]
    + [f"{s}_past_reference_n" for s in STREAMS]
    + [f"{s}_past_q80" for s in STREAMS]
    + DYNAMIC27
)
feature_panel = read_development_table(FEATURE_TABLE).select(*feature_columns).toPandas()
feature_panel["week_start"] = pd.to_datetime(feature_panel["week_start"], errors="raise")
if feature_panel.empty or feature_panel.duplicated(["iso3", "week_start"]).any():
    raise ValueError("Empty or duplicate-key dynamic feature panel.")
if feature_panel["iso3"].nunique() != 40:
    raise ValueError("Dynamic feature panel does not cover 40 countries.")
if not feature_panel["week_start"].dt.dayofweek.eq(0).all():
    raise ValueError("Feature weeks must start on Monday.")
if not feature_panel["week_start"].between("2022-01-03", "2025-12-29").all():
    raise ValueError("Unexpected V2 feature reference-period dates.")

numeric = feature_panel.drop(columns=["iso3", "week_start"]).apply(pd.to_numeric, errors="raise")
if np.isinf(numeric.to_numpy(dtype=float)).any():
    raise ValueError("Persisted feature panel contains infinity.")
for stream in STREAMS:
    pct = numeric[f"{stream}_historical_percentile"].dropna()
    rising = numeric[f"{stream}_consecutive_rising_weeks"].dropna()
    support = numeric[f"{stream}_past_reference_n"]
    if not pct.between(0, 1).all() or not rising.between(0, 4).all():
        raise ValueError(f"Dynamic range check failed for {stream}.")
    if support.lt(0).any():
        raise ValueError(f"Negative historical observation support for {stream}.")
    if numeric.loc[support.lt(26), f"{stream}_historical_percentile"].notna().any():
        raise ValueError(f"Past-reference support rule violated for {stream}.")

feature_support = pd.DataFrame([
    {"stream": s,
     "rows": len(feature_panel),
     "activity_available_pct": 100 * numeric[f"{s}_activity"].notna().mean(),
     "historical_percentile_available_pct": 100 * numeric[f"{s}_historical_percentile"].notna().mean(),
     "robust8_mean_available_pct": 100 * numeric[[f"{s}_{suffix}" for suffix in ROBUST_SUFFIXES]].notna().mean().mean()}
    for s in STREAMS
])
print("V2 DYNAMIC-FEATURE SNAPSHOT: PASS")
print("Rows:", len(feature_panel), "| Countries:", feature_panel["iso3"].nunique())
print("Candidates:", len(DYNAMIC27), "| Retained robust features:", len(ROBUST24))
print("Reference observations required:", feature_manifest["minimum_past_reference_observations"])
display(feature_support.round(2))
print("No feature values recalculated or saved.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Historical fold design and the operational evaluation queue
# MAGIC
# MAGIC The Version 2 temporal comparison uses expanding historical training sets and four later **development folds** (`2023_H1`, `2023_H2`, `2024_H1`, `2024_H2`). Eligibility and the elevated-activity q80 reference are reconstructed before each fold; training outcome windows must end before that fold begins. The prediction target is an **elevation in at least one locally usable respiratory stream during the next four weeks**, not a WHO-certified outbreak.
# MAGIC
# MAGIC **Scoreable now vs evaluable later:** ranking eligibility comes only from current and past observations. Whether future surveillance is complete determines *later evaluation coverage*, never who can enter the forecast queue. Missing future labels are **unknown**, not zero.
# MAGIC
# MAGIC The archived operational queue has 1,612 unique country-weeks across 105 forecast dates and four exact horizons (1–4 weeks), producing 6,448 horizon rows. The fixed fold-size contract is: 444 / 333 / 332 / 503. This is **not** the 2,935-row Version 1 dataset or the 479-row V1 reserved evaluation.

# COMMAND ----------

# Read identifiers/horizons, not 2026 holdout data and not model scores.
queue = read_development_table(SCORED_QUEUE).select(
    "fold", "iso3", "prediction_date", "horizon_weeks"
).toPandas()
queue["prediction_date"] = pd.to_datetime(queue["prediction_date"], errors="raise")
keys = ["fold", "iso3", "prediction_date"]
queue_cw = queue[keys].drop_duplicates()
expected = {"2023_H1": 444, "2023_H2": 333, "2024_H1": 332, "2024_H2": 503}
fold_counts = queue_cw.groupby("fold").size().to_dict()
if fold_counts != expected:
    raise ValueError(f"Unexpected development fold sizes: {fold_counts}")
if len(queue) != 6448 or len(queue_cw) != 1612:
    raise ValueError("The full-queue / horizon counts changed.")
if queue_cw["prediction_date"].nunique() != 105:
    raise ValueError("Unexpected number of forecast weeks.")
if queue["prediction_date"].ge(pd.Timestamp("2026-01-05")).any():
    raise ValueError("Protected 2026 holdout period appeared in development queue.")
if queue.duplicated(keys + ["horizon_weeks"]).any():
    raise ValueError("Duplicate country-week-horizon record.")
if set(queue["horizon_weeks"].dropna().unique()) != {1, 2, 3, 4}:
    raise ValueError("Missing or unexpected forecast horizon.")
coverage = queue.groupby(keys)["horizon_weeks"].agg(["size", "nunique"])
if not coverage["size"].eq(4).all() or not coverage["nunique"].eq(4).all():
    raise ValueError("Some country-weeks lack one of the four horizons.")

print("V2 TEMPORAL DEVELOPMENT QUEUE: PASS")
display(pd.DataFrame([{"fold": f, "country_weeks": fold_counts[f]} for f in FOLDS]))
print("Unique country-weeks:", len(queue_cw))
print("Distinct forecast dates:", queue_cw["prediction_date"].nunique())
print("Exact-horizon rows:", len(queue))
print("No holdout-period rows in development queue.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Additional enrichments and conservative regional timing
# MAGIC
# MAGIC Controlled historical experiments made the following decisions:
# MAGIC
# MAGIC - **24 past-only dynamics** were a promising V2 addition (archived mean ROC-AUC change +0.0219 and mean AP change +0.0161 over the 35-feature reference); the original research explicitly *did not* establish a robust Top-5 improvement from this feature family alone.
# MAGIC - **12 spatial neighbourhood features** were retained during an intermediate Logistic Regression development stage, giving a 71-feature representation. Later Random Forest studies did **not** justify retaining those additional spatial variables over the 59-feature comparator.
# MAGIC - **Influenza subtype6/subtype8**, **all-age ILI/SARI**, and an inferred COVID-19 lab-positivity denominator were **not** adopted as primary predictors. The unapproved enrichment candidates remain visible in the archived original notebook.
# MAGIC - **WHO-region reporting delay:** exact historic publication dates were unavailable. The development-stage RF59 comparison examined current, 1-week-lagged and **2-week-lagged** regional information against RF47 (region removed). The 2-week lag was carried forward as a conservative timing assumption, **not proof of what was published at that time**.
# MAGIC
# MAGIC The enrichment closeout's RF59 lag-2 candidate was **intermediate**; the later THREE_HEAD selection is not a result of this notebook.

# COMMAND ----------

region = read_development_table(REGION_FOLD_METRICS).select(
    "fold", "variant", "auc", "ap"
).toPandas()
if region.duplicated(["fold", "variant"]).any():
    raise ValueError("Duplicate regional-delay fold/variant rows.")
required_variants = {
    "rf59_current_region", "rf59_region_delay1",
    "rf59_region_delay2", "rf47_without_region"
}
if not required_variants.issubset(set(region["variant"])):
    raise ValueError("Regional comparison variants are missing.")
if set(region["fold"]) != set(FOLDS):
    raise ValueError("Regional comparison folds changed.")

region_summary = region.groupby("variant", as_index=False).agg(
    mean_roc_auc=("auc", "mean"),
    mean_average_precision=("ap", "mean"),
    minimum_fold_auc=("auc", "min"),
    folds=("fold", "nunique")
)
if not region_summary["folds"].eq(4).all():
    raise ValueError("A regional variant is missing historical folds.")
rs = region_summary.set_index("variant")
for variant, auc, ap in [
    ("rf59_region_delay2", 0.6710, 0.3653),
    ("rf47_without_region", 0.6592, 0.3461),
]:
    if not np.isclose(rs.loc[variant, "mean_roc_auc"], auc, atol=0.0005, rtol=0):
        raise ValueError(f"Unexpected historical ROC-AUC for {variant}.")
    if not np.isclose(rs.loc[variant, "mean_average_precision"], ap, atol=0.0005, rtol=0):
        raise ValueError(f"Unexpected historical average precision for {variant}.")

horizons = read_development_table(HORIZON_COMPARISON).toPandas()
if "horizon_weeks" not in horizons.columns or set(horizons["horizon_weeks"]) != {1, 2, 3, 4}:
    raise ValueError("Missing archived 1–4-week horizon comparison.")
if len(horizons) != 4:
    raise ValueError("Expected four archived horizon-summary rows.")
print("V2 REGIONAL DELAY / EXACT-HORIZON EVIDENCE: PASS")
display(region_summary.round(4))
display(horizons[[c for c in ["horizon_weeks", "mean_auc", "mean_ap", "mean_prevalence", "minimum_auc"]
                  if c in horizons.columns]].sort_values("horizon_weeks").round(4))
print("These are archived development results, not the final THREE_HEAD comparison.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Weekly Top-5 and episode evaluation: archived RF59-era evidence
# MAGIC
# MAGIC The persisted **RF59-era** operational evaluation scored the complete weekly queue and ranked scoreable countries **before** excluding outcomes that were unavailable later. Recorded evidence:
# MAGIC
# MAGIC | Historical evaluation item | Archived value |
# MAGIC |---|---:|
# MAGIC | Development country-week opportunities | 1,612 |
# MAGIC | Exact-horizon scoring rows | 6,448 |
# MAGIC | Matched predictions with observed outcomes | 1,254 |
# MAGIC | Top-5 selections across 105 forecast weeks | 525 |
# MAGIC | Top-5 outcomes known later | 402 |
# MAGIC | Top-5 outcomes unknown later | 123 |
# MAGIC | Evaluable episodes | 129 |
# MAGIC | Top-5 episodes detected | 87 |
# MAGIC | Top-5 episodes detected ≥1 / ≥2 / ≥3 weeks early | 57 / 39 / 28 |
# MAGIC | Top-10 episodes detected | 112 |
# MAGIC
# MAGIC The weakest archived fold was `2023_H1` (42 evaluable episodes, 18 detected at Top-5 and 29 at Top-10). RSV reporting availability and country/surveillance-system generalisation were important limitations. **These figures must not be labelled final THREE_HEAD episode results** without an explicit matched artefact/protocol comparison in Notebooks 04–05.

# COMMAND ----------

# Only archived DEVELOPMENT evidence. No protected 2026 tables.
known_count = read_development_table(KNOWN_PREDICTIONS).count()
if known_count != 1254:
    raise ValueError(f"Matched development prediction rows changed: {known_count}")

top5 = read_development_table(TOP5_REVIEWS).select(
    "fold", "iso3", "prediction_date", "future_elevation_observed"
).toPandas()
if len(top5) != 525 or top5.duplicated(["fold", "iso3", "prediction_date"]).any():
    raise ValueError("Top-5 review membership/count changed.")
known_top5 = int(top5["future_elevation_observed"].notna().sum())
unknown_top5 = int(top5["future_elevation_observed"].isna().sum())
if (known_top5, unknown_top5) != (402, 123):
    raise ValueError("Unknown-outcome coverage changed in saved Top-5 reviews.")

episodes = read_development_table(EPISODE_DETAIL).select(
    "fold", "top5_detected", "top5_early1", "top5_early2", "top5_early3", "top10_detected"
).toPandas()
if len(episodes) != 129:
    raise ValueError("Episode-audit denominator changed.")
flag_expectations = {
    "top5_detected": 87, "top5_early1": 57,
    "top5_early2": 39, "top5_early3": 28, "top10_detected": 112
}
for col, count in flag_expectations.items():
    if episodes[col].isna().any() or int(episodes[col].sum()) != count:
        raise ValueError(f"Saved episode event count changed: {col}")

print("RF59-ERA HISTORICAL OPERATIONAL EVIDENCE: PASS")
print("Matched known predictions:", known_count)
print("Top-5 reviews:", len(top5), "| Known:", known_top5, "| Unknown:", unknown_top5)
print("Episode denominator:", len(episodes))
display(pd.DataFrame([{"measure": key, "episodes": value} for key, value in flag_expectations.items()]))
print("Unknown outcomes have not been recoded as negative.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Enrichment closeout and the handoff to final modelling
# MAGIC
# MAGIC The completed enrichment stage's **versioned closeout record** is authoritative for what was decided **at that time**. It records `DATA_ENRICHMENT_COMPLETE`, `high_risk_next_4w_foldaware`, 59 RF features, median imputation, 2-week regional lag, the four 2023–2024 folds, the 1,612-country-week operational queue, and an **unfrozen** RF59 development candidate.
# MAGIC
# MAGIC **Important:** Enrichment closure (13 October) is **not** the final project model freeze. The later corrected 30 October head-to-head chose **THREE_HEAD disease-aware Random Forest** over RF59 (recorded mean ROC-AUC 0.7187 versus 0.6917, difference +0.0270). That conclusion and its exact evaluation specification belong in Notebooks 04 and 05, **not** in an overwritten enrichment closeout.
# MAGIC
# MAGIC **Still material:** missing surveillance, unequal lab and case-count measurements, retrospective publication timing, non-calibrated scores, short-horizon versus long-horizon decay, and limited country-level generalisation. Retain them in the GitHub README and results pack.
# MAGIC
# MAGIC The original `03_data_enrichment.py` remains the detailed source-building and research archive. This clean chapter intentionally does not call `.write`, `.saveAsTable`, MLflow registry actions or model `.fit`.

# COMMAND ----------

c = closeout.iloc[0]
expected_closeout = {
    "stage": "DATA_ENRICHMENT_COMPLETE",
    "development_candidate": "RF59_28DAY_REGION_LAG2_MEDIAN",
    "primary_target": "high_risk_next_4w_foldaware",
    "regional_reporting_lag_weeks": 2,
    "development_queue_country_weeks": 1612,
    "development_forecast_weeks": 105,
    "exact_horizon_rows": 6448,
    "h1_2026_status": "LOCKED_UNTOUCHED",
}
for field, expected in expected_closeout.items():
    if field not in closeout.columns or c[field] != expected:
        raise ValueError(f"Enrichment closeout mismatch: {field}")
if bool(c["final_model_frozen"]):
    raise ValueError("Enrichment closeout must not claim the final model was frozen.")

notebook03_summary = pd.DataFrame([{
    "enrichment_stage": c["stage"],
    "feature_snapshot": FEATURE_TABLE.rsplit(".", 1)[-1],
    "feature_candidates": len(DYNAMIC27),
    "robust_features_retained": len(ROBUST24),
    "intermediate_rf59_development_candidate": c["development_candidate"],
    "development_queue_rows": len(queue_cw),
    "four_horizon_rows": len(queue),
    "top5_reviews": len(top5),
    "known_top5_outcomes": known_top5,
    "unknown_top5_outcomes": unknown_top5,
    "evaluable_episodes": len(episodes),
    "h1_2026_lock_status": lock[0]["status"],
    "final_model_selection_recorded_elsewhere": "THREE_HEAD (30 October 2026)",
}])

print("NOTEBOOK 03 — READ-ONLY ENRICHMENT REPRODUCTION SUMMARY")
display(notebook03_summary)
print("All checks above use existing development artifacts.")
print("No original raw sources, saved features, models, issue registers or Delta tables changed.")
print("No H1-2026 holdout observations/outcomes/scoring accessed.")
print("Next: 04_final_model_selection_CLEAN and 05_corrected_temporal_modelling_CLEAN.")