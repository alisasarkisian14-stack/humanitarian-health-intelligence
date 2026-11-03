# MLflow Experiment Tracking and Model Governance

**Project:** Humanitarian Health Intelligence  
**Platform:** Databricks, MLflow and Unity Catalog  
**Evidence reviewed:** 3 November 2026  
**Selected final research model:** THREE_HEAD  
**Operational model status:** Not deployed or formally promoted

---

## 1. Purpose and Scope

This document records the verified MLflow experimentation and model-governance evidence supporting the Humanitarian Health Intelligence respiratory early-warning research project.

The project used Databricks and MLflow to record historical development experiments, evaluation metrics, model parameters, saved artifacts and registered-model lineage.

The evidence covers four research activities:

1. Version 2 development experiments and reporting-delay comparisons.
2. Version 2 final RF59 evaluation and model preservation.
3. Version 3 controlled development and retained-reference registration.
4. Version 3 frozen experiment summary and formal promotion decision.

These records document the historical model-development process.

**Important:** The later corrected THREE_HEAD comparison was reproduced using saved Databricks predictions. An MLflow run recording THREE_HEAD's corrected seven-period results has not been established by the evidence reviewed here.

---

## 2. Verified MLflow Experiment Inventory

| Experiment | Verified evidence | Research purpose |
|---|---|---|
| Respiratory_Early_Warning_V2_Development | 16 visible runs | Regional sensitivity, ranking diagnostics, rolling horizons and operational evaluation |
| Respiratory_Early_Warning_V2_Final | Sealed RF59 run | Historical final-evaluation metrics and saved model |
| Respiratory_Early_Warning_V3_Controlled_Development | Retained RF59-D2 closeout run | Model configuration, saved pipeline and registry lineage |
| Respiratory_Early_Warning_V3_Model_Experiments | Frozen study summary run | Experiment inventory, aggregate performance and promotion decision |

A separate `v2_logistic_regression` experiment is visible in the MLflow experiment list, but its individual run contents were not audited for this document.

All descriptions below refer to previously recorded MLflow evidence. No training or experiment logging was performed to prepare this report.

---

## 3. V2 Development — Regional Reporting Sensitivity

### Verified MLflow run

**Run:** `V2-PT03 | rf59_current_region`  
**Status:** Finished  
**Experiment:** `Respiratory_Early_Warning_V2_Development`

This run records the historical evaluation of a 59-feature Random Forest using current-region information.

### Logged performance

| Recorded metric | Value |
|---|---:|
| Cross-validation ROC-AUC | 0.6837 |
| Cross-validation average precision | 0.3750 |
| Top-5 capture | 49.01% |
| Top-5 precision | 30.69% |
| Top-10 capture | 72.73% |

### Recorded model settings

| Parameter | Value |
|---|---|
| Model family | RandomForestClassifier |
| Feature count | 59 |
| Number of trees | 500 |
| Minimum samples per leaf | 10 |
| Maximum features | sqrt |
| Random state | 42 |
| Regional delay | 0 weeks |
| Target | elevated_at_least_once_next_4_weeks |

**Tracking evidence:** 5 logged metrics and 11 logged parameters.

The run metadata identifies it as a historical audit import rather than evidence of a newly fitted estimator during that logging operation.

### Interpretation

The current-region variant achieved relatively strong historical development performance, but its zero-week regional delay does not represent the conservative reporting-delay assumption preferred for the later RF59-D2 reference.

The regional sensitivity study also compared delayed-region alternatives and a no-region comparator.

The results informed a preference for conservative feature availability rather than simply selecting the highest retrospective ROC-AUC.

These metrics belong to the earlier V2 development protocol and must not be directly compared with the corrected C4 results as though they were evaluated on the same cohort.

---

## 4. V2 Final — Sealed RF59 Evaluation

### Verified MLflow run

**Run:** `V2_RF59_H1_2026_SEALED`  
**Status:** Finished  
**Experiment:** `Respiratory_Early_Warning_V2_Final`

### Logged evaluation metrics

| Recorded metric | Value |
|---|---:|
| ROC-AUC | 0.7458 |
| Average precision | 0.6865 |
| Average-precision lift over prevalence | 1.8048 |

The run contains 14 logged metrics, 11 logged parameters and one logged model.

### Model specification

The sealed reference used a Random Forest classifier with 59 features, 500 trees, square-root feature sampling, median imputation, random state 42 and a two-week regional lag.

The run also records model-reload parity and restrictions against post-holdout retuning or model switching.

### Interpretation

This record documents the historical V2 sealed evaluation and its corresponding model artifact.

The sealed evaluation is a separate historical protocol, not the matched seven-period corrected RF59 versus THREE_HEAD comparison.

**The historical SEALED status must not be interpreted as meaning that H1-2026 remains an untouched independent test today.**

The MLflow record does not show a registered model associated with this specific run.

---

## 5. V3 Controlled Development — Retained RF59-D2 Reference

### Verified MLflow run

**Run:** `V3_RF59_D2_RETAINED_REFERENCE_CLOSEOUT`  
**Status:** Finished  
**Experiment:** `Respiratory_Early_Warning_V3_Controlled_Development`

### Logged evidence

| Record | Verified value |
|---|---|
| Model family | RandomForestClassifier |
| Input features | 59 |
| Logged parameters | 15 |
| Logged metrics | 3 |
| Logged models | 1 |
| Historical development ROC-AUC | 0.6794 |
| Historical development AP lift | 1.7609 |
| Model-reload parity | PASS |
| Formal Gate-4 result | CLOSED_NO_PROMOTION |

The run records the retained model configuration and its saved pipeline.

### Registered-model lineage

**Registered model:** `humanitarian.model_ready.respiratory_early_warning_rf59_d2`

**Model version:** `1`

**Model URI:**

`models:/humanitarian.model_ready.respiratory_early_warning_rf59_d2/1`

The saved metadata identifies the retained model as a shadow reference pending prospective validation.

The model scores are uncalibrated ranking signals, not calibrated outbreak probabilities.

### Interpretation

This MLflow record provides evidence of model serialization, parameter logging, model lineage and registered-reference preservation.

The registered RF59-D2 shadow reference is historically distinct from the corrected RF59 comparator used in the final C4 research evaluation.

**THREE_HEAD was not registered through this run.**

---

## 6. V3 Frozen Experiment Study — Formal Model Selection

### Verified MLflow run

**Run:** `V3_FROZEN_MODEL_EXPERIMENT_STUDY`  
**Status:** Finished  
**Experiment:** `Respiratory_Early_Warning_V3_Model_Experiments`

### Experiment scope

| Experiment attribute | Recorded value |
|---|---:|
| Prespecified model configurations | 12 |
| Historical evaluation periods | 7 |
| Inner-validation splits | 14 |
| Historical estimator fits | 266 |
| Logged metrics | 9 |
| Logged parameters | 0 |

The archived study compared Logistic Regression, Random Forest, histogram gradient boosting, disease-aware approaches and an ensemble.

### Recorded aggregate findings

| Recorded study metric | Value |
|---|---:|
| Best model-family mean ROC-AUC | 0.7027 |
| Best model-family mean AP lift | 1.8823 |
| Best operational relative gain | 11.90% |

These metrics summarise different aspects of the historical study and must not be assumed to represent one identical configuration.

### Logged artifacts

- `frozen_results.json`
- `study_closeout.json`

The run metadata records that no model fitting occurred in the logging step itself.

The figure of 266 estimator fits describes the historical experiment workload. It does not mean that 266 individual MLflow training runs were logged.

### Formal promotion decision

**Decision:** No V3 challenger satisfied every frozen promotion requirement.

**Promoted challenger:** NONE

**Retained registered reference:** RF59-D2

This decision considered operational improvement, temporal stability, uncertainty and other predeclared promotion conditions rather than ROC-AUC alone.

The study-level MLflow closeout preserves this historical decision.

---

## 7. Relationship to the Final THREE_HEAD Research Model

The corrected C4 study subsequently evaluated:

- A 59-feature Random Forest comparator (RF59).
- Three pathogen-specific Random Forest heads (THREE_HEAD).

The combined THREE_HEAD country-week ranking is determined by the maximum available eligible pathogen-head score.

### Corrected retrospective findings

| Metric | Corrected RF59 | THREE_HEAD |
|---|---:|---:|
| Seven-period mean ROC-AUC | 0.6917 | **0.7187** |
| Known-positive Top-5 reviews | 342 | **372** |
| Top-5 detected onsets | 73/120 | **91/120** |

The paired mean ROC-AUC difference was +0.0270.

**Research selection:** THREE_HEAD is the preferred final model for the retrospective research handover.

**Formal governance:** No new deployment promotion or model registration has been demonstrated.

The existing RF59-D2 registered shadow reference therefore remains unchanged.

The corrected comparison includes previously inspected historical development periods, including `2026_H1`. Its results do not constitute independent prospective validation.

### Evidence boundary

The corrected THREE_HEAD results are supported by saved Databricks predictions and the verified corrected temporal modelling notebook.

They must not be presented as metrics logged in the V2 or V3 MLflow runs documented above.

---

## 8. Experiment-Tracking Limitations

The verified MLflow evidence has several important boundaries.

1. Some historical records are study-level summaries or audit imports rather than individual estimator-training runs.
2. Logged parameters and metrics vary across experiments.
3. Different model-development stages used different evaluation populations and temporal protocols.
4. The V2 sealed results, V3 controlled results and corrected C4 results are not directly interchangeable.
5. Model and experiment artifacts may require authorised Databricks or Unity Catalog access.
6. An independently logged and registered final THREE_HEAD model has not been verified.
7. No independently untouched prospective evaluation has been established for the final research selection.

These limitations should be stated explicitly when presenting the project.

---

## 9. Supporting Project Documentation

- [Main project README](../README.md)
- [Final Model Results and Evaluation](README.md)
- [Data Enrichment Notebook](../notebooks/03_data_enrichment_final.py)
- [Historical Model Selection Notebook](../notebooks/04_model_selection_final.py)
- [Corrected Temporal Modelling Notebook](../notebooks/05_corrected_temporal_modelling_final.py)

The underlying MLflow run records, experiment metadata and model artifacts remain in the project's Databricks environment.

Selected, appropriately redacted MLflow screenshots may be added separately as supporting evidence, subject to publication approval.

---

*Evidence reviewed against actual Databricks MLflow run screens on 3 November 2026. No new models, experiments or registry changes were created for this report.*
