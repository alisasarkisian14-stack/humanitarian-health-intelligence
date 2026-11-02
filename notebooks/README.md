Research Notebooks
This directory holds the five cleaned, verified presentation notebooks for the Humanitarian Health Intelligence respiratory early-warning project. They are organised in study order; the longer originals remain in the Databricks workspace to preserve the experimental audit trail.
Order	Notebook	What it verifies
01	[Exploratory Data Analysis](01_exploratory_data_analysis_clean.py)	Source feasibility, quality checks and rationale for pivoting from cholera to respiratory surveillance
02	[Machine Learning Baselines](02_machine_learning_baselines_clean.py)	Version 1 frozen dataset, sharp-rise heuristic, Logistic Regression, validation diagnostics
03	[Data Enrichment](03_data_enrichment_final.py)	Past-only feature engineering, surveillance availability, four-fold development queue and historical enrichment decisions
04	[Model Selection](04_model_selection_final.py)	V2/V3 selection history, frozen 12-candidate experiment, formal Gate-4 no-promotion record and retained RF59-D2 shadow reference
05	[Corrected Temporal Modelling](05_corrected_temporal_modelling_final.py)	Matched seven-period RF59 versus THREE_HEAD scores, ranking, disease-specific onset episodes and bootstrap uncertainty


Final research model: THREE_HEAD. This model is selected for the retrospective research handover; it has not been registered/promoted for operational deployment. The formally registered RF59-D2 shadow reference remains unchanged.

Reproducibility boundary: Notebook 02 fits only its historical Logistic Regression baseline in memory. Notebooks 01 and 03–05 are primarily read-only verification of saved Databricks research artifacts. These notebooks cannot be fully reproduced with GitHub access alone: the original pinned data, Delta artifacts and suitable runtime require authorised Databricks access. Do not rerun original archive notebooks end-to-end to regenerate evidence.

See the [main project README](../README.md) for the research question, data sources, feature engineering, evaluation, performance and limitations.
