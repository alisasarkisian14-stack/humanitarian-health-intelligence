Research Notebooks
This directory contains the five cleaned, Databricks-verified notebooks for Humanitarian Health Intelligence, a retrospective respiratory-surveillance early-warning research project.
Notebook guide
#	Notebook	Purpose
01	[Exploratory Data Analysis](01_exploratory_data_analysis_clean.py)	Assess source quality, feasibility and the shift from cholera to respiratory surveillance.
02	[Machine Learning Baselines](02_machine_learning_baselines_clean.py)	Reproduce the frozen Version 1 sharp-rise benchmark and Logistic Regression baseline.
03	[Data Enrichment](03_data_enrichment_final.py)	Verify past-only feature engineering, surveillance coverage and historical enrichment decisions.
04	[Model Selection](04_model_selection_final.py)	Audit the historical model experiments, promotion criteria and retained registered RF59-D2 shadow reference.
05	[Corrected Temporal Modelling](05_corrected_temporal_modelling_final.py)	Reproduce matched RF59 versus THREE_HEAD comparisons, Top-K rankings, episode detection and uncertainty.


Final research model
THREE_HEAD is the selected model for the retrospective research handover. It combines three pathogen-specific Random Forest scores (influenza, RSV and COVID-19) using the maximum available eligible head score.

Model governance: THREE_HEAD has not been formally promoted, deployed or registered. The existing RF59-D2 registered shadow reference remains unchanged.
Reproducibility and scope
- The cleaned notebooks are presentation and verification artifacts, not a complete raw-data-to-trained-model public pipeline.
- Notebook 02 fits its historical Logistic Regression baseline in memory. Notebooks 01 and 03–05 principally inspect existing saved research artifacts without model training or Delta table writes.
- Reproducing the checks requires authorised access to the pinned data and saved Delta tables in Databricks. Public GitHub access alone is insufficient.
- The original, longer notebooks remain archived in Databricks as the experimental record. Do not run the original training, table-writing or one-time holdout cells just to reproduce the documentation.
- The corrected seven-period comparison includes 2026_H1 as previously used development evidence, not an independent untouched test.
For the full methodology, data sources, numerical results and limitations, see the [main project README](../README.md).
