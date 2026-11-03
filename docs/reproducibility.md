# Reproducibility and Technical Execution Guide

**Project:** Humanitarian Health Intelligence  
**Study:** Retrospective respiratory-surveillance early warning  
**Selected final research model:** THREE_HEAD  
**Platform:** Databricks, Python, PySpark, Delta Lake and MLflow  
**Documentation date:** 3 November 2026

---

## 1. Purpose and Reproducibility Boundary

This document explains how the research was organised, which software and saved artifacts it uses, and how its reported results were verified.

The project investigates whether historical influenza, RSV and COVID-19 surveillance can support weekly country-level analyst prioritisation.

Two different levels of reproducibility must be distinguished.

| Level | Scope | Current status |
|---|---|---|
| Source and methodology inspection | Review the notebooks, feature definitions, historical decisions and evaluation methodology | Available in GitHub |
| Saved-artifact reproduction | Execute cleaned notebooks against existing Databricks data and reproduce saved research results | Completed and verified in the project workspace |
| Independent end-to-end reconstruction | Rebuild the source datasets, historical features, fitted models and evaluation from independently available snapshots | Not established by this public repository |
| Prospective validation | Evaluate predictions generated without prior access to the evaluation outcomes | Not established |

**Reproducibility claim:** The project has undergone internal, saved-artifact-based verification.

It is not a fully self-contained public training pipeline, and it should not be represented as independently prospectively validated.

---

## 2. Repository Structure

The GitHub repository contains the following research and reporting components.

```text
humanitarian-health-intelligence/
├── README.md
├── .gitignore
├── notebooks/
│   ├── README.md
│   ├── 01_exploratory_data_analysis_clean.py
│   ├── 02_machine_learning_baselines_clean.py
│   ├── 03_data_enrichment_final.py
│   ├── 04_model_selection_final.py
│   └── 05_corrected_temporal_modelling_final.py
├── results/
│   ├── README.md
│   ├── MLFLOW_EVIDENCE.md
│   └── figures/
│       ├── README.md
│       ├── 01_roc_auc_by_period.png
│       ├── 02_topk_country_reviews.png
│       ├── 03_episode_detection_lead_times.png
│       ├── 04_bootstrap_uncertainty.png
│       └── 05_top3_winning_head_composition.png
└── docs/
    ├── feature_dictionary.md
    └── reproducibility.md
```

The original, longer experiment notebooks remain preserved in Databricks. They are not replaced by the presentation versions in GitHub.

---

## 3. Required Environment

### 3.1 Execution Platform

The cleaned notebooks were checked in a Databricks environment supporting:

- Python
- PySpark / Spark SQL
- Delta Lake and Databricks Unity Catalog
- pandas
- NumPy
- scikit-learn

Historical model artifacts and experiment-tracking evidence are additionally stored through MLflow and Unity Catalog.

### 3.2 Observed Package Versions

Notebook 02 reported the following versions during its successful verification on 2 November 2026.

| Library | Observed version |
|---|---|
| NumPy | 2.3.4 |
| pandas | 2.3.3 |
| scikit-learn | 1.7.2 |

These are observed versions from the verification environment, not a fully pinned environment specification.

The exact Python interpreter version, Databricks Runtime release, Spark version, MLflow version and cluster configuration have not all been independently captured in this documentation.

A future environment freeze should record those details before claiming exact cross-environment numerical reproduction.

### 3.3 Databricks Access

Executing the notebooks requires appropriate permissions for the original saved project data and artifacts under:

`humanitarian.model_ready`

The workspace must also provide access to relevant source files or saved tables used by the exploratory notebook.

GitHub access alone is insufficient.

The public repository does not distribute private workspace credentials, raw source snapshots, saved model binaries or the complete historical Delta datasets.

---

## 4. Notebook Execution Order

The five cleaned notebooks should be read in numerical order.

| Step | Notebook | Main purpose | Execution behaviour |
|---|---|---|---|
| 01 | Exploratory Data Analysis | Source suitability, quality checks and research pivot | Read-only exploratory checks |
| 02 | Machine Learning Baselines | Historical Version 1 heuristic and Logistic Regression | Fits the original baseline in memory |
| 03 | Data Enrichment | Accepted features, surveillance availability and historical V2 evidence | Read-only saved-artifact verification |
| 04 | Model Selection | Frozen V2/V3 experiments, selection and governance | Read-only aggregate verification |
| 05 | Corrected Temporal Modelling | Matched RF59 versus THREE_HEAD comparison and uncertainty | Read-only analysis of saved predictions |

The notebooks are exported as Databricks Python source files.

They preserve cell boundaries using Databricks source-file markers such as `# COMMAND ----------` and Markdown markers such as `# MAGIC %md`.

These files should be imported into Databricks as notebooks rather than treated as ordinary standalone Python scripts.

### Recommended Procedure

1. Import the five `.py` files into a Databricks workspace.
2. Attach suitable compute with the required dependencies and data permissions.
3. Review the opening Markdown section and safety contract for each notebook.
4. Execute the Python cells in their original order.
5. Check the printed `PASS` assertions, row counts, stored artifact identities and displayed summaries.
6. Stop if a required table, source schema, saved decision or expected denominator differs.
7. Do not change historical thresholds or expected results merely to obtain a passing check.

Notebook 02 intentionally fits only its original Logistic Regression baseline in memory.

The remaining cleaned notebooks primarily inspect existing historical evidence. They are not requests to retrain the selected final model.

---

## 5. Frozen Data and Artifact References

The following are important saved research artifacts referenced by the cleaned notebooks.

### 5.1 Version 1 Baseline Dataset

**Dataset:**

`humanitarian.model_ready.health_risk_v1_1_dev_20261004_132520_0c36ef`

**Snapshot ID:**

`20261004_132520_0c36ef`

**Delta version:** `0`

| Dataset property | Expected value |
|---|---:|
| Total country-weeks | 2,935 |
| Countries | 40 |
| Primary predictors | 35 |
| Training observations | 2,127 |
| Validation observations | 329 |
| Reserved retrospective observations | 479 |

The cleaned baseline notebook reads training and validation observations only.

The reserved retrospective split must not be rerun merely to tidy or reproduce the presentation notebook.

### 5.2 Version 2 Dynamic Feature Snapshot

**Feature table:**

`humanitarian.model_ready.health_risk_v2_dynamic_features_20261005_174033`

**Enrichment closeout:**

`humanitarian.model_ready.health_risk_v2_data_enrichment_closeout_20261013_134936`

| Feature snapshot property | Expected value |
|---|---:|
| Country-week rows | 8,360 |
| Countries | 40 |
| Dynamic feature candidates | 27 |
| Retained robust dynamic features | 24 |
| Minimum earlier observations for historical reference | 26 |

The exact definitions and feature-selection rules are documented in [Feature Dictionary](feature_dictionary.md).

### 5.3 Frozen RF59 Feature Manifest

**Feature manifest:**

`humanitarian.model_ready.health_risk_v2_finalsel01e_rf59_feature_manifest_20261013_142603`

This artifact contains the historical 59-feature Random Forest input specification.

The clean model-selection notebook checks the saved feature-list identity and fingerprint rather than rebuilding the specification from memory.

### 5.4 Version 3 Formal Gate-4 Record

**Historical closeout:**

`humanitarian.model_ready.health_risk_v3_gate4_closeout_v1`

The original V3 study records:

- 12 prespecified model configurations.
- Seven outer evaluation periods.
- 266 historical estimator fits.
- A formal decision that no challenger passed every promotion gate.
- The retained RF59-D2 registered shadow reference.

The formal historical decision must not be rewritten to imply that the later THREE_HEAD research model was promoted.

### 5.5 Corrected C4 Prediction Artifacts

**Corrected RF59 scores:**

`humanitarian.model_ready.health_risk_c4_02_20261024_211736_33602e_rf59_v1_predictions`

**Pathogen-head rankings:**

`humanitarian.model_ready.health_risk_c4_06_20261024_211736_33602e_disease_heads_v1_rankings`

**Corrected study definitions:**

`humanitarian.model_ready.health_risk_c4_01_20261024_211736_33602e_definitions`

Notebook 05 reads its allowlisted archived Delta sources using version `0`.

The THREE_HEAD country-week score is reconstructed as the maximum available eligible influenza, RSV and COVID-19 head score.

This process verifies saved score parity; it does not refit the individual Random Forest models.

**Versioning qualification:** Other cleaned notebooks read their specified saved tables according to their own code contracts. Do not assume every table read across all notebooks is automatically pinned to Delta version zero.

---

## 6. Expected Reproduction Checks

### Notebook 01 — Exploratory Data Analysis

The exploratory checks examine original surveillance-source suitability and document why the initial cholera-focused research question was replaced with respiratory-surveillance modelling.

The notebook also verifies basic data-quality findings, including missingness, reporting-source categories and limitations of the contributing surveillance datasets.

This stage does not train the final models.

### Notebook 02 — Machine Learning Baselines

Expected checks include:

- Frozen Version 1 manifest validation.
- Correct historical training and validation splits.
- Frozen target-threshold verification.
- Training-derived sharp-rise baseline.
- Original Logistic Regression fit in memory.
- Validation metrics and temporal diagnostics.
- Final read-only reproduction summary.

The Version 1 Logistic Regression baseline is not the final selected model.

### Notebook 03 — Data Enrichment

Expected historical evidence includes:

| Check | Expected value |
|---|---:|
| Dynamic feature rows | 8,360 |
| Retained dynamic features | 24 |
| Development country-weeks | 1,612 |
| Historical development folds | 4 |
| Exact-horizon prediction rows | 6,448 |
| Original RF59-era Top-5 reviews | 525 |
| Original RF59-era episode denominator | 129 |

The original Version 2 episode denominator is not interchangeable with the later corrected C4 denominator.

### Notebook 04 — Model Selection

Expected evidence includes:

| Check | Expected value |
|---|---:|
| Prespecified candidate configurations | 12 |
| Inner estimator fits | 210 |
| Historical family selections | 28 |
| Outer evaluation periods | 7 |
| Total historical estimator fits | 266 |
| Formal Gate-4 result | NO PROMOTION |

The retained registered reference is RF59-D2.

### Notebook 05 — Corrected Temporal Modelling

Expected matched evaluation population:

| Check | Expected value |
|---|---:|
| Historical development periods | 7 |
| Forecast weeks | 183 |
| Matched country-weeks | 2,491 |
| Known future outcomes | 2,094 |
| Unknown future outcomes | 397 |
| Known positive outcomes | 672 |

Expected principal results:

| Metric | Corrected RF59 | THREE_HEAD |
|---|---:|---:|
| Seven-period mean ROC-AUC | 0.6917 | 0.7187 |
| Top-5 known-positive reviews | 342 | 372 |
| Top-5 detected eligible onsets | 73 | 91 |

The corrected onset study identified 175 confirmed country-pathogen surveillance-elevation onsets, of which 120 had an eligible forecasting opportunity.

The model comparison uses the same matched country-week records and labels for both approaches.

Unknown future outcomes must remain unknown rather than being assigned negative labels.

---

## 7. Statistical Reproduction

Notebook 05 reproduces the original corrected paired-bootstrap analysis.

| Setting | Value |
|---|---|
| Bootstrap resamples | 2,000 |
| Block length | 4 weeks |
| Random seed | 20261030 |
| Primary comparison | THREE_HEAD minus corrected RF59 |

The procedure resamples blocks of historical forecast weeks while preserving paired model comparisons.

Expected results:

| Metric | Observed improvement | 95% bootstrap interval |
|---|---:|---:|
| Mean ROC-AUC | +0.0270 | +0.0004 to +0.0536 |
| Mean average precision | +0.0139 | -0.0109 to +0.0419 |

The intervals describe retrospective, post-selection uncertainty.

They are not evidence of independently validated prospective improvement or deployment readiness.

---

## 8. MLflow Tracking and Model Registry

MLflow evidence documents the earlier Version 2 and Version 3 research stages.

The reviewed evidence includes:

- Regional feature and reporting-delay sensitivity studies.
- The historical V2 RF59 sealed evaluation.
- The V3 retained RF59-D2 model and registry record.
- The V3 frozen experiment study and formal no-promotion decision.

Relevant tracked artifacts include saved model pipelines, recorded parameters, evaluation metrics and study closeout files.

The registered shadow reference is:

`models:/humanitarian.model_ready.respiratory_early_warning_rf59_d2/1`

**Important:** This is the registered historical RF59-D2 model, not a registered THREE_HEAD model.

The corrected THREE_HEAD study is supported by saved Databricks prediction evidence. A dedicated MLflow run recording and registering THREE_HEAD's corrected final performance has not been established.

See [MLflow Experiment Tracking and Model Governance](../results/MLFLOW_EVIDENCE.md).

---

## 9. Temporal Governance and Research Limitations

The project used chronology-aware modelling and conservative assumptions intended to reduce leakage.

The corrected C4 research applied an assumed 21-day surveillance-availability offset and a 42-day training-label maturity rule.

However, the historical publication and revision timestamps of every surveillance source could not be fully reconstructed.

The corrected seven-period comparison includes `2026_H1`.

**This period has been consumed as retrospective development evidence and must not be described as an untouched independent test.**

Some older V2 artifacts retain historical preholdout lock statuses. Those saved records document the state of an earlier project stage and do not establish the current independence of the later reused period.

Other limitations include:

- Uneven influenza, RSV and COVID-19 reporting coverage.
- Country-level differences in surveillance and testing.
- Distinct measurement scales for laboratory positivity and reported COVID-19 cases.
- Uncalibrated pathogen-head scores.
- Missing outcomes and incomplete forecasting opportunities.
- Limited evidence of improvement at longer warning lead times.
- No independently verified prospective deployment performance.

These limitations must accompany the model-performance findings.

---

## 10. Safety and Non-Destructive Execution

The cleaned presentation notebooks should not be used to regenerate or replace archived model-development artifacts.

Do not run original historical notebook cells that would:

- Overwrite or recreate saved Delta tables.
- Reopen one-time reserved or sealed evaluation procedures.
- Retune a model against historical evaluation results.
- Change the frozen experimental design.
- Replace the registered RF59-D2 reference.
- Promote or register THREE_HEAD without a separate approval process.

The cleaned notebooks were designed to preserve the original research history and verify existing artifacts.

Notebook 02's in-memory Logistic Regression fit is a deliberate exception to the otherwise primarily read-only workflow.

---

## 11. Public Repository and Data Governance

The public GitHub repository contains research code and documentation, not the complete underlying data or trained-model artifacts.

Before publishing additional material:

1. Confirm that the code and associated research materials are authorised for public release.
2. Exclude credentials, access tokens, personal data and unnecessary private workspace identifiers.
3. Respect source-provider attribution and usage conditions.
4. Avoid distributing restricted Databricks tables or saved models without permission.
5. Review screenshots and exported evidence for sensitive metadata.

No patient-level clinical dataset is required for the documented aggregate surveillance analyses.

---

## 12. Verification Summary

The five cleaned notebooks were executed and checked against the project's saved Databricks research artifacts on 2 November 2026.

The verification confirmed:

- Frozen baseline cohort identity.
- Historical source and feature-enrichment evidence.
- Frozen model-selection and governance records.
- Exact matched RF59 and THREE_HEAD score alignment.
- Fixed-budget country-ranking results.
- Corrected distinct-onset evaluation.
- Paired statistical uncertainty.
- No unapproved model promotion or registered-model replacement.

The final research selection is **THREE_HEAD**, based on stronger corrected retrospective mean discrimination and country-week prioritisation.

The existing RF59-D2 registered shadow reference remains unchanged.

**The next validation milestone is a prospectively specified, independent evaluation with reliable historical or prospective source-availability information.**

---

## 13. Supporting Documentation

- [Main Project README](../README.md)
- [Feature Dictionary](feature_dictionary.md)
- [Research Notebook Index](../notebooks/README.md)
- [Final Model Results and Evaluation](../results/README.md)
- [Evaluation Figures](../results/figures/README.md)
- [MLflow Experiment Evidence](../results/MLFLOW_EVIDENCE.md)

---

*This document records internal saved-artifact reproducibility and methodological safeguards. It does not claim full independent reconstruction, prospective validation or operational deployment.*
