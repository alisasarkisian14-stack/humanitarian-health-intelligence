# Research Notebooks

Five cleaned and Databricks-verified notebooks documenting the **Humanitarian Health Intelligence** retrospective respiratory-surveillance early-warning project.

The notebooks are organised in research order and preserve the reasoning, evaluation evidence and decisions behind the final research model.

## Notebook Guide

| Step | Notebook | Purpose |
|---|---|---|
| 01 | [Exploratory Data Analysis](01_exploratory_data_analysis_clean.py) | Source quality, feasibility analysis and the pivot from cholera to respiratory surveillance. |
| 02 | [Machine Learning Baselines](02_machine_learning_baselines_clean.py) | Frozen Version 1 dataset, sharp-rise benchmark, Logistic Regression and validation diagnostics. |
| 03 | [Data Enrichment](03_data_enrichment_final.py) | Past-only feature engineering, surveillance coverage, temporal folds and enrichment decisions. |
| 04 | [Model Selection](04_model_selection_final.py) | Historical model comparisons, evaluation criteria and the formal Gate-4 decision. |
| 05 | [Corrected Temporal Modelling](05_corrected_temporal_modelling_final.py) | Final RF59 versus THREE_HEAD research comparison, Top-K ranking, episode detection and uncertainty. |

## Final Research Model

**Selected research model: THREE_HEAD**

THREE_HEAD combines three pathogen-specific Random Forest models for influenza, RSV and COVID-19. Each country-week receives the maximum available eligible pathogen-head score.

The model was selected for the **retrospective research handover** based on the corrected comparative evaluation.

**Model governance:** THREE_HEAD has not been formally registered, promoted or deployed. The existing RF59-D2 registered shadow reference remains unchanged.

## Reproducibility and Scope

- The cleaned notebooks provide a structured presentation and verification of existing research.
- Notebook 02 reproduces the historical Logistic Regression baseline in memory.
- Notebooks 01 and 03–05 primarily verify saved Databricks research artifacts without retraining models or modifying Delta tables.
- Reproducing the checks requires authorised access to pinned Databricks datasets and saved research artifacts.
- The original experimental notebooks remain preserved in Databricks as the historical research record.
- The corrected seven-period analysis includes `2026_H1` as previously used development evidence, not an independent untouched holdout.
- The project is a retrospective proof of concept, not a validated operational outbreak-warning system.

## Main Project Documentation

For the complete methodology, data sources, feature engineering, model evaluation, results and limitations, see the [main project README](../README.md).
