# Databricks notebook source
# MAGIC %md
# MAGIC # 04 — Model Selection and Specification Freeze | Clean Research Audit
# MAGIC
# MAGIC **Humanitarian Respiratory Early Warning · Databricks · Read-only presentation notebook**
# MAGIC
# MAGIC This notebook verifies **existing** records from the historical model-selection work without executing the 68,535-line original experiment notebook. It is a compact *audit and presentation version*, not a replacement for the archived computational source.
# MAGIC
# MAGIC **Research sequence (distinct historical decisions):**
# MAGIC
# MAGIC 1. **Version 2 (13 October 2026):** the frozen development selection chose `RF59_28DAY_REGION_LAG2_MEDIAN` (59 features, 2-week regional lag) before the one-time H1-2026 evaluation. We read the original selection, feature manifest and predeclared evaluation protocol as **historical records**.
# MAGIC 2. **Version 3, frozen D2 experiment:** seven historical outer periods (including 2026_H1) were used to compare a prespecified 12-candidate matrix with a matched RF59-D2 reference; the formal **Gate-4 promotion result was NO PROMOTION**. The registered RF59-D2 reference was retained for **shadow use pending prospective validation**.
# MAGIC 3. **Later corrected analysis (Notebook 05):** `CORRECTED-10A` compares the saved `THREE_HEAD` disease-aware score and RF59 across seven historical periods. That later research comparison is **not the same as an approved Gate-4 promotion or a registered THREE_HEAD model**.
# MAGIC
# MAGIC **Scientific protection:** The 2026_H1 period was consumed in the later V3 historical development study. A preholdout V2 record stating `LOCKED_UNTOUCHED` describes its status **at the time it was frozen**; it does **not** establish an untouched 2026 holdout as of this audit. Historical source-publication timing is not fully proven by these results.
# MAGIC
# MAGIC **Safe to run:** Only allowlisted existing Delta research summaries, configuration manifests, metadata and evaluation protocols are read. No raw/row-level evaluation scores, 2026 labels, model fitting, predictions, MLflow writes, registry changes or Delta writes. Never run the full original archived notebook to tidy it up.
# MAGIC
# MAGIC **Preserve original:** `04_final_model_selection (3).py` is the unchanged experiment-history archive. Next notebook: `05_corrected_temporal_modelling_CLEAN`.
# MAGIC

# COMMAND ----------

import json
import hashlib
import pandas as pd

SCHEMA = "humanitarian.model_ready"
NAMES = {
    "v2_selection": "health_risk_v2_finalsel02_model_selection_decision_20261013_142926",
    "v2_feature_manifest": "health_risk_v2_finalsel01e_rf59_feature_manifest_20261013_142603",
    "v2_full_spec": "health_risk_v2_freeze03_final_specification_20261013_143333",
    "v2_protocol": "health_risk_v2_freeze03_holdout_protocol_20261013_143333",
    "study": "health_risk_v3_study_contract_v1",
    "timing": "health_risk_v3_g3_03_manifest_v1",
    "representations": "health_risk_v3_g3_03_representations_v1",
    "matrix": "health_risk_v3_g4_01_model_matrix_v1",
    "matrix_manifest": "health_risk_v3_g4_01_experiment_manifest_v1",
    "nested_panels_manifest": "health_risk_v3_g4_02a_manifest_v1",
    "inner_manifest": "health_risk_v3_g4_02b1_manifest_v1",
    "inner_selection": "health_risk_v3_g4_02b1_family_selections_v1",
    "outer_manifest": "health_risk_v3_g4_02b2_manifest_v1",
    "outer_discrimination": "health_risk_v3_g4_02b2_discrimination_v1",
    "outer_operational": "health_risk_v3_g4_02b2_operational_metrics_v1",
    "outer_summary": "health_risk_v3_g4_02b2_model_summary_v1",
    "promotion_manifest": "health_risk_v3_g4_03_manifest_v1",
    "promotion_audit": "health_risk_v3_g4_03_promotion_audit_v1",
    "gate4_closeout": "health_risk_v3_gate4_closeout_v1",
    "binding_limitations": "health_risk_v3_gate4_binding_limitations_v1",
    "inference_contract": "health_risk_v3_g5_02a_inference_contract_v1",
}

# Explicit allowlist prevents reading raw evaluation/holdout tables accidentally.
def read_aggregate(label):
    if label not in NAMES:
        raise ValueError("Unapproved research-aggregate identifier: " + str(label))
    return spark.table(f"{SCHEMA}.{NAMES[label]}")

def one_record(label):
    frame = read_aggregate(label).limit(2).toPandas()
    if len(frame) != 1:
        raise ValueError(f"Expected exactly one saved {label} record; got {len(frame)}")
    return frame.iloc[0]

study = one_record("study")
timing = one_record("timing")
closeout = one_record("gate4_closeout")

if study["status"] != "FROZEN":
    raise RuntimeError("Original V3 study contract is not frozen.")
if timing["status"] != "V3_POINT_IN_TIME_DESIGN_FROZEN":
    raise RuntimeError("Historical timing study not frozen.")
if timing["primary_timing_scenario"] != "D2_PRIMARY":
    raise RuntimeError("Original primary D2 timing scenario changed.")
if closeout["status"] != "GATE4_CLOSED_NO_PROMOTION":
    raise RuntimeError("Original Gate-4 no-promotion closeout changed.")
if closeout["retained_reference"] != "RF59_D2_MATCHED_REFERENCE":
    raise RuntimeError("Retained registered reference identity changed.")
if closeout["promoted_model"] != "NONE":
    raise RuntimeError("A challenger is unexpectedly marked as promoted.")

print("NOTEBOOK 04 PRE-FLIGHT: PASS")
print("Frozen study contract: VERIFIED")
print("Primary historical timing scenario: D2_PRIMARY")
print("Original formal Gate-4 decision: NO PROMOTION")
print("Retained reference: RF59_D2_MATCHED_REFERENCE")
print("No model fits, table writes, or raw evaluation rows accessed.")


# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Frozen Version 2 model selection — historical preholdout record
# MAGIC
# MAGIC Before the later V3 study, the Version 2 development process selected a **Random Forest model with 59 predictors** (35 original features plus 24 robust dynamics), two-calendar-week delayed regional features, median imputation and 500 trees. The V2 study also specified **four primary H1-2026 evaluation criteria before opening that reserved period**.
# MAGIC
# MAGIC This cell checks that the selection decision, exact feature-list fingerprint, frozen model specification and success criteria still exist in their persisted historical form. **It does not reopen or score the 2026 dataset**. Its saved status `LOCKED_UNTOUCHED` is interpreted at the historical *freeze date* only; the subsequent V3 study explicitly used 2026_H1 observations as development evidence.
# MAGIC
# MAGIC The older V2 RF59 development results and their operational Top-5 episode counts are documented separately in Notebook 03. The original V2 frozen decision must not be rewritten because later corrected exploratory research favoured THREE_HEAD on an AUC comparison.
# MAGIC

# COMMAND ----------

# Archived V2 selection/freeze: read metadata and 59-row feature manifest only.
V2_RF59_SHA256 = "41c3dfef6e40d0e6413b86052d7827fafbec50212106e789b8cb5c8c15a850f3"
selection = one_record("v2_selection")
spec = one_record("v2_full_spec")
features = read_aggregate("v2_feature_manifest").toPandas()
protocol = read_aggregate("v2_protocol").toPandas()

if selection["decision"] != "SELECT_RANDOM_FOREST":
    raise RuntimeError("V2 preholdout model-selection decision differs from the archive.")
if selection["selected_model_spec"] != "RF59_28DAY_REGION_LAG2_MEDIAN":
    raise RuntimeError("V2 selected model identity changed.")
if bool(selection["post_holdout_model_switching_allowed"]):
    raise RuntimeError("Original V2 model-switching safeguard changed.")
if bool(selection["holdout_used_for_selection"]):
    raise RuntimeError("V2 selection was marked as using its reserved holdout.")
if spec["freeze_status"] != "FULL_SPECIFICATION_FROZEN":
    raise RuntimeError("V2 specification-freeze status is not intact.")
if spec["selected_model"] != selection["selected_model_spec"]:
    raise RuntimeError("V2 selection and specification disagree.")
if int(spec["feature_count"]) != 59 or str(spec["rf59_sha256"]) != V2_RF59_SHA256:
    raise RuntimeError("V2 final model feature fingerprint changed.")
if len(features) != 59 or features["feature_name"].duplicated().any():
    raise RuntimeError("V2 RF59 manifest does not contain 59 unique features.")
if int(features["is_regional"].sum()) != 12:
    raise RuntimeError("Expected 12 historical regional features.")
if set(features["rf59_sha256"].dropna().astype(str)) != {V2_RF59_SHA256}:
    raise RuntimeError("V2 feature-manifest hash mismatch.")

primary = protocol.loc[protocol["role"].eq("PRIMARY")].copy()
expected_thresholds = {
    "roc_auc": 0.60,
    "average_precision_lift_vs_prevalence": 1.30,
    "top5_episode_detection_rate": 0.50,
    "top5_episode_detection_1plus_week_early": 0.33,
}
if len(primary) != 4 or primary["metric"].duplicated().any():
    raise RuntimeError("Original V2 primary evaluation protocol is incomplete.")
actual_thresholds = dict(zip(primary["metric"], primary["threshold"].astype(float)))
if actual_thresholds != expected_thresholds:
    raise RuntimeError(f"Original V2 evaluation floors changed: {actual_thresholds}")

print("V2 PREHOLDOUT MODEL SELECTION AND FREEZE: PASS")
print("Archived selected model:", selection["selected_model_spec"])
print("Frozen predictors: 59 | Regional predictors: 12")
print("Four prespecified holdout-evaluation criteria: VERIFIED")
display(primary[["metric", "direction", "threshold"]].sort_values("metric"))
print("Historical V2 preholdout status:", selection["h1_2026_status"])
print("This does not assert a currently untouched holdout or a THREE_HEAD promotion.")
print("No training, scoring or Delta writes.")


# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Prespecified, bounded experiment matrix
# MAGIC
# MAGIC The study froze **12** configurations before evaluation: 3 regularized Logistic Regression, 4 Random Forest, 2 Histogram Gradient Boosting, 2 disease-aware stream-expert configurations and 1 fixed-weight LR+RF ensemble. Four predefined feature representations had 59, 70, 68 and 79 predictors.
# MAGIC
# MAGIC The primary timing scenario was **D2**, requiring a two-week reporting delay for relevant local and regional signals in the experimental representation. **This is an imposed availability assumption**, not proof of WHO publication timestamps. A historical score cannot automatically be interpreted as a genuinely real-time forecast.
# MAGIC
# MAGIC The disease-aware Random Forest configuration already existed among the prespecified challengers; this cell **does not train any of them**.
# MAGIC

# COMMAND ----------

matrix_manifest = one_record("matrix_manifest")
if matrix_manifest["status"] != "V3_MODEL_MATRIX_FROZEN":
    raise RuntimeError("Predeclared V3 model matrix was not frozen.")

matrix = read_aggregate("matrix").toPandas()
if len(matrix) != 12 or matrix["config_id"].duplicated().any():
    raise RuntimeError("V3 candidate matrix must contain 12 unique IDs.")

expected_families = {
    "LOGISTIC": 3,
    "RANDOM_FOREST": 4,
    "HIST_GRADIENT_BOOSTING": 2,
    "DISEASE_AWARE": 2,
    "ENSEMBLE": 1,
}
counts = matrix["family"].value_counts().to_dict()
if counts != expected_families:
    raise RuntimeError(f"Frozen candidate-family counts changed: {counts}")

reps = read_aggregate("representations").toPandas()
expected_repr = {
    "R59_REFERENCE": 59,
    "R70_RF59_Q11": 70,
    "R68_RF59_T9": 68,
    "R79_RF59_Q11_T9": 79,
}
observed_repr = dict(zip(reps["representation"], reps["feature_count"].astype(int)))
if observed_repr != expected_repr:
    raise RuntimeError(f"Frozen representation manifest changed: {observed_repr}")

print("V3 FROZEN EXPERIMENT MATRIX: PASS")
print("Prespecified configurations:", len(matrix))
print("Prespecified feature representations:", len(reps))
display(matrix[[c for c in ["config_id", "family", "representation", "estimator_kind", "derived_candidate"] if c in matrix.columns]].sort_values("family"))
display(reps[["representation", "feature_count"]].sort_values("feature_count"))
print("No configuration added, searched or fitted.")


# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Nested chronological selection and outcome protection
# MAGIC
# MAGIC The V3 protocol selected each fitted model family **within earlier chronological development windows**, using **mean AP lift** as the primary criterion, then ROC-AUC, then lexicographic configuration ID. The design included label purging so an inner training example's 28-day outcome window could not overlap its validation period.
# MAGIC
# MAGIC The archived selection attempted **14 chronological inner splits** across **seven outer periods**; **210 estimator fits** were already completed historically. This cleaned notebook reads only the saved **selection summaries**, not the underlying features or labels. Disease-specific heads were separately fitted in the original historical research, not here.
# MAGIC
# MAGIC **Seven historical periods**, rather than Notebook 03's earlier four, were used in this later V3 study. These are different evaluation protocols and their AUC/AP figures must not be pooled without qualification.
# MAGIC

# COMMAND ----------

inner_panel_manifest = one_record("nested_panels_manifest")
inner_manifest = one_record("inner_manifest")
if inner_panel_manifest["status"] != "V3_NESTED_PANELS_MATERIALIZED":
    raise RuntimeError("Historical nested-panel source not finalised.")
if inner_manifest["status"] != "V3_INNER_MODEL_SELECTION_COMPLETE":
    raise RuntimeError("Historical inner model-selection not complete.")
if int(inner_manifest["actual_estimator_fits"]) != 210:
    raise RuntimeError("Expected 210 saved historical inner estimator fits.")
if bool(inner_manifest["outer_performance_used_for_selection"]):
    raise RuntimeError("Inner selection unexpectedly used outer evaluation.")

selected = read_aggregate("inner_selection").toPandas()
expected_folds = {"2023_H1", "2023_H2", "2024_H1", "2024_H2", "2025_H1", "2025_H2", "2026_H1"}
expected_inner_families = {"LOGISTIC", "RANDOM_FOREST", "HIST_GRADIENT_BOOSTING", "DISEASE_AWARE"}
if len(selected) != 28 or selected.duplicated(["outer_fold", "family"]).any():
    raise RuntimeError("Expected 28 unique historical family selections.")
if set(selected["outer_fold"]) != expected_folds or set(selected["family"]) != expected_inner_families:
    raise RuntimeError("Unexpected historical selection periods or families.")
if not set(selected["selected_config_id"]).issubset(set(matrix["config_id"])):
    raise RuntimeError("Selected configuration is absent from the frozen matrix.")
if "outer_performance_used" in selected and selected["outer_performance_used"].astype(bool).any():
    raise RuntimeError("Saved selection table indicates outer-performance selection.")

print("V3 NESTED CHRONOLOGICAL SELECTION: PASS")
print("Historical inner estimator fits:", int(inner_manifest["actual_estimator_fits"]))
print("Historical family selections:", len(selected), "across 7 outer periods")
display(selected[[c for c in ["outer_fold", "family", "selected_config_id", "selected_mean_ap_lift", "selected_mean_roc_auc"] if c in selected.columns]].sort_values(["outer_fold", "family"]))
print("No historical model fitting or selection was rerun.")


# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Outer evaluation, operational ranking, and denominators
# MAGIC
# MAGIC The historical Gate-4 outer study evaluated **five model families** (Logistic Regression, Random Forest, Histogram Gradient Boosting, Disease-aware experts, and a derived ensemble) over **seven** periods, yielding **35** fold-level discrimination records. Its operational review used **175** saved model/fold/review-budget records.
# MAGIC
# MAGIC For the five-family cross-fold summary, report ROC-AUC alongside **AP lift** and fixed analyst review budgets, particularly **Top-3** early episode utility. Missing later outcomes were not assumed negative. Model scores were **not certified calibrated probabilities**. The later 2026_H1 period was *consumed* in this V3 analysis and is not an untouched test here.
# MAGIC

# COMMAND ----------

outer_manifest = one_record("outer_manifest")
if outer_manifest["status"] != "V3_OUTER_EVALUATION_COMPLETE":
    raise RuntimeError("Historical outer evaluation not completed.")
for key, expected in [("inner_estimator_fits", 210), ("outer_estimator_fits", 42),
                      ("cumulative_estimator_fits", 252), ("outer_folds", 7)]:
    if int(outer_manifest[key]) != expected:
        raise RuntimeError(f"Historical outer-evaluation manifest changed: {key}")

disc = read_aggregate("outer_discrimination").toPandas()
ops = read_aggregate("outer_operational").toPandas()
summary = read_aggregate("outer_summary").toPandas()
expected_outer_families = expected_inner_families | {"ENSEMBLE"}
if len(disc) != 35 or len(ops) != 175 or len(summary) != 5:
    raise RuntimeError("Saved 7-fold outer-evaluation result counts changed.")
if set(disc["fold"]) != expected_folds or set(disc["model_family"]) != expected_outer_families:
    raise RuntimeError("Outer-evaluation fold/family identities changed.")
if disc.duplicated(["fold", "model_family"]).any():
    raise RuntimeError("Duplicated outer discrimination rows.")
if set(summary["model_family"]) != expected_outer_families:
    raise RuntimeError("Outer summary contains unexpected model families.")

summary_cols = ["model_family", "mean_roc_auc", "min_roc_auc", "mean_ap_lift",
                "k3_precision_known", "k3_episode_detection",
                "k3_early_episode_yield_per_100_reviews"]
print("V3 OUTER EVALUATION: PASS")
print("Saved discrimination rows:", len(disc), "| Operational-budget rows:", len(ops))
print("Saved estimator-fit budget:", int(outer_manifest["cumulative_estimator_fits"]), "/ 300")
display(summary[[c for c in summary_cols if c in summary.columns]].sort_values("mean_roc_auc", ascending=False).round(4))
print("These are archived V3 Gate-4 study metrics, not CORRECTED-10A metrics.")


# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Formal Gate-4 promotion decision
# MAGIC
# MAGIC The original prespecified V3 evaluation compared every challenger with a **matched RF59-D2 reference** under the same frozen study conditions. The promotion standard also included **operational utility, Top-K behaviour, uncertainty, random-baseline advantage, and fold robustness**, not just mean ROC-AUC.
# MAGIC
# MAGIC **Recorded official conclusion:** `NO_V3_CHALLENGER_PASSES_FROZEN_PROMOTION_STANDARD`. The formal Gate-4 closeout is `GATE4_CLOSED_NO_PROMOTION`; `promoted_model = NONE`. The RF59-D2 reference was retained without post-hoc retuning. **This remains a valid description of that specific historical protocol**, even though later research explored different corrected comparisons.
# MAGIC
# MAGIC Do **not** interpret a classifier's improved AUC as automatically satisfying a separately specified promotion gate.
# MAGIC

# COMMAND ----------

promotion_manifest = one_record("promotion_manifest")
if promotion_manifest["status"] != "V3_PROMOTION_GATE_AUDIT_COMPLETE":
    raise RuntimeError("Historical promotion-gate audit not complete.")
if promotion_manifest["promotion_decision"] != "NO_V3_CHALLENGER_PASSES_FROZEN_PROMOTION_STANDARD":
    raise RuntimeError("Unexpected formal promotion-gate decision.")
if int(promotion_manifest["cumulative_estimator_fits"]) != 266:
    raise RuntimeError("Frozen promotion experiment budget changed.")

prom = read_aggregate("promotion_audit").toPandas()
if len(prom) != 5 or prom["all_core_promotion_gates_pass"].astype(bool).any():
    raise RuntimeError("Historical promotion-audit outcome changed.")
if closeout["status"] != "GATE4_CLOSED_NO_PROMOTION" or closeout["promoted_model"] != "NONE":
    raise RuntimeError("Saved Gate-4 closeout is inconsistent.")
if closeout["post_outer_retuning"] != "PROHIBITED":
    raise RuntimeError("Gate-4 post-outer retuning was not prohibited.")
if int(closeout["estimator_fits_used"]) != 266:
    raise RuntimeError("Gate-4 study used a different fit budget.")
limits = read_aggregate("binding_limitations").toPandas()
if limits.empty or limits["limitation_id"].duplicated().any():
    raise RuntimeError("Historical binding-limitations register is missing or malformed.")

print("FORMAL V3 GATE-4 PROMOTION AUDIT: PASS")
print("Promotion decision:", promotion_manifest["promotion_decision"])
print("Promoted challenger:", closeout["promoted_model"])
print("Retained registered reference:", closeout["retained_reference"])
print("Total historical estimator fits:", int(closeout["estimator_fits_used"]))
display(prom[[c for c in ["candidate", "candidate_primary_utility", "primary_relative_gain",
                          "all_core_promotion_gates_pass", "failed_gates"] if c in prom.columns]].round(4))
display(limits[[c for c in ["limitation_id", "category", "implication", "correct_next_response"] if c in limits.columns]])
print("No training, retuning, promotion, logging or table writes performed.")


# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Archived registered RF59-D2 reference — shadow inference only
# MAGIC
# MAGIC Following the original Gate-4 no-promotion decision, the project packaged an **RF59-D2 reference** in MLflow and froze a separate weekly inference contract for **shadow use pending prospective validation**. That registered reference and its 59 ordered predictors are *not the same object* as the later disease-aware `THREE_HEAD` research scores.
# MAGIC
# MAGIC This cell verifies **only the contract metadata and checksum**. It does not load an MLflow model, make fresh predictions, read 2026_H1 samples, score labels, or change the registry.
# MAGIC
# MAGIC A saved D2 reporting delay is an explicit modelling assumption; it does not establish that original surveillance observations were available at the claimed time in actual operations.
# MAGIC

# COMMAND ----------

frozen_inference = one_record("inference_contract")
if frozen_inference["status"] != "FROZEN_SHADOW_PENDING_PROSPECTIVE_VALIDATION":
    raise RuntimeError("Archived registered-reference inference contract status changed.")
raw_contract = str(frozen_inference["contract_json"])
if hashlib.sha256(raw_contract.encode("utf-8")).hexdigest() != str(frozen_inference["contract_sha256"]):
    raise RuntimeError("Frozen inference-contract checksum mismatch.")
inference = json.loads(raw_contract)
model_spec = inference["model"]
ordered = list(model_spec["ordered_features"])
if len(ordered) != 59 or len(set(ordered)) != 59:
    raise RuntimeError("Registered RF59 reference input feature contract changed.")
if model_spec["registered_model_name"] != f"{SCHEMA}.respiratory_early_warning_rf59_d2":
    raise RuntimeError("RF59 registered model identity changed.")

print("ARCHIVED RF59-D2 REGISTERED-REFERENCE CONTRACT: PASS")
print("Status:", frozen_inference["status"])
print("Registered reference:", model_spec["registered_model_name"])
print("Ordered input features:", len(ordered))
print("Model URI:", model_spec.get("registered_model_uri", "not recorded"))
print("The registered shadow reference is not evidence of THREE_HEAD registration.")
print("No MLflow or holdout dataset was accessed.")


# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Handoff to the later corrected temporal comparison
# MAGIC
# MAGIC **Original Notebook 04 finding, frozen V3 Gate 4:** no challenger passed the formal promotion standard, and RF59-D2 remained the archived reference.
# MAGIC
# MAGIC **Later Notebook 05 finding:** the `CORRECTED-10A` read-only analysis reconstructs **THREE_HEAD** as the maximum of the available influenza/RSV/COVID disease-specific scores and compares it with matched saved RF59 scores across **seven** periods (2023_H1 through 2026_H1). The original Notebook 05 also evaluates a separate **chronologically selected** score policy that is not identical to always taking THREE_HEAD.
# MAGIC
# MAGIC **Correct distinction:** a promising corrected seven-period research comparison ≠ a frozen production model, ≠ the V3 prespecified promotion gate, and ≠ evidence that a future or untouched independent holdout has been evaluated. The corrected work documents that historical reporting-time correctness remains unverified.
# MAGIC
# MAGIC **Handoff:** Notebook 05 should (a) reconstruct the exact saved comparator cohorts, (b) display the corrected seven-fold ROC-AUC/AP/within-week/Top-K comparisons, (c) label any research-preferred policy explicitly, and (d) document what is and is not registered or deployable. Do not alter the original archived gate or MLflow lineage while presenting the later research.
# MAGIC

# COMMAND ----------

# Metadata-only handoff: merely check the two later saved comparator identities.
# No row-level scores, 2026 samples, outcome labels, or MLflow models are read.
C4_RF59 = SCHEMA + ".health_risk_c4_02_20261024_211736_33602e_rf59_v1_predictions"
C4_HEADS = SCHEMA + ".health_risk_c4_06_20261024_211736_33602e_disease_heads_v1_rankings"
c4_rf59_exists = bool(spark.catalog.tableExists(C4_RF59))
c4_heads_exists = bool(spark.catalog.tableExists(C4_HEADS))

if not c4_rf59_exists or not c4_heads_exists:
    raise RuntimeError("Notebook 05 corrected-comparison saved inputs are missing. Stop before claiming corrected performance.")

handoff = pd.DataFrame([{
    "historical_v3_gate4_status": str(closeout["status"]),
    "historical_promoted_challenger": str(closeout["promoted_model"]),
    "historical_registered_reference": str(model_spec["registered_model_name"]),
    "original_v3_outer_periods": len(expected_folds),
    "original_candidate_configs": len(matrix),
    "historical_estimator_fits": int(closeout["estimator_fits_used"]),
    "later_rf59_comparator_persisted": c4_rf59_exists,
    "later_disease_heads_persisted": c4_heads_exists,
    "later_corrected_analysis": "CORRECTED-10A — see Notebook 05",
    "read_only": True,
}])

print("NOTEBOOK 04 — READ-ONLY SELECTION HISTORY: PASS")
display(handoff)
print("Original V3 no-promotion gate is preserved; RF59 remains the registered reference.")
print("Later THREE_HEAD corrected comparison must be verified separately in Notebook 05.")
print("No model training, saved-data writes, protected row-level reads, or model registration.")
