# Final Model Results & Evaluation

**Humanitarian Health Intelligence**  
*Retrospective respiratory-surveillance early warning and country-level analyst prioritisation*

**Research snapshot:** 2 November 2026  
**Selected final research model:** THREE_HEAD  
**Evaluation status:** Retrospective development study — not independent prospective validation

---

## 1. Executive Summary

This evaluation examines whether machine learning can prioritise countries for analyst review ahead of elevated influenza, respiratory syncytial virus (RSV) or COVID-19 surveillance activity.

The final corrected study compares two approaches:

- **RF59:** A single Random Forest classifier using 59 predictors.
- **THREE_HEAD:** Three pathogen-specific Random Forest models, combined using the maximum available eligible disease-head score.

THREE_HEAD was selected as the **final research model** based on its stronger retrospective mean discrimination and fixed-budget country-prioritisation performance.

Across seven historical development periods, THREE_HEAD achieved a mean ROC-AUC of **0.7187**, compared with **0.6917** for the corrected RF59 comparator.

Under a fixed Top-5 weekly analyst-review budget, THREE_HEAD identified **372 known-positive country-weeks**, compared with **342** for RF59.

These findings support further investigation of disease-aware modelling. They do not establish readiness for operational deployment.

## 2. Evaluation Population and Design

| Evaluation characteristic | Value |
|---|---:|
| Historical development periods | 7 |
| Forecast weeks | 183 |
| Matched eligible country-weeks | 2,491 |
| Known future outcomes | 2,094 |
| Unknown future outcomes | 397 |
| Known positive outcomes | 672 |
| Countries in the modelling population | 40 |
| Assessment window | Approximately 28 days |

The seven periods span `2023_H1` to `2026_H1`.

Both models were assessed on the same eligible country-week observations and future outcome labels.

All eligible countries were ranked before future outcomes were inspected. Unknown outcomes remained in the operational review queue but were excluded from known-outcome precision calculations.

**Important:** All seven evaluation periods, including `2026_H1`, were previously used during development. This comparison is not an untouched independent holdout.

Historical surveillance availability was approximated using a 21-day reporting offset. Actual historical publication and revision timestamps were not fully reconstructable.

## 3. Final Model Comparison

### 3.1 Discrimination and Country Prioritisation

| Performance measure | RF59 | THREE_HEAD |
|---|---:|---:|
| Seven-period mean ROC-AUC | 0.6917 | **0.7187** |
| Top-3 known-positive reviews | 230 | **243** |
| Top-3 precision among known outcomes | 47.52% | **50.73%** |
| Top-5 known-positive reviews | 342 | **372** |
| Top-5 precision among known outcomes | 43.18% | **47.03%** |

The paired mean ROC-AUC difference was **+0.0270** in favour of THREE_HEAD.

THREE_HEAD achieved higher ROC-AUC in **4 of 7** historical periods.

The improvement was not consistent across every period. During `2026_H1`, RF59 achieved ROC-AUC **0.7452**, compared with **0.7229** for THREE_HEAD.

This variation is relevant to model robustness and changing disease-surveillance conditions.

### 3.2 Operational Review Capacity

| Review measure | RF59 | THREE_HEAD |
|---|---:|---:|
| Total Top-3 reviews | 549 | 549 |
| Top-3 known outcomes | 484 | 479 |
| Top-3 known-positive reviews | 230 | **243** |
| Total Top-5 reviews | 915 | 915 |
| Top-5 known outcomes | 792 | 791 |
| Top-5 known-positive reviews | 342 | **372** |

Both models used identical weekly analyst-review budgets.

The selected countries and their known-outcome denominators differed slightly because each model ranked countries differently.

THREE_HEAD yielded **30 additional known-positive Top-5 reviews** without increasing the total review workload.

This result describes country-week prioritisation, not distinct disease-episode detection.

## 4. Statistical Uncertainty

A paired four-week moving-block bootstrap was applied to the saved model predictions, using **2,000 resamples**.

| Paired performance difference | Estimate | 95% bootstrap interval |
|---|---:|---:|
| Mean ROC-AUC | +0.0270 | +0.0004 to +0.0536 |
| Mean average precision | +0.0139 | -0.0109 to +0.0419 |

The ROC-AUC interval narrowly excludes zero, suggesting a positive difference in the retrospective comparison.

The average-precision interval includes zero. Therefore, a reliable improvement in average precision has not been established.

These intervals are conditional on previously fitted models, selected strategies and historical development data. They must not be interpreted as independent prospective confirmation.

## 5. Disease-Onset Detection and Early Warning

### 5.1 Corrected Episode Denominator

The corrected analysis identified **175 confirmed country-pathogen surveillance-elevation onsets**.

Of these, **120 onsets** had at least one eligible forecasting opportunity and therefore entered the detection-rate denominator.

A further 54 high-activity observations had incomplete preceding four-week surveillance histories and could not be confirmed as new onsets. These are not 54 additional confirmed episodes.

These surveillance-defined events are not equivalent to clinically confirmed disease outbreaks.

### 5.2 Top-5 Detection Results

| Detection measure | RF59 | THREE_HEAD |
|---|---:|---:|
| Detected episodes | 73 | **91** |
| Detection rate | 60.83% | **75.83%** |
| Detected at least 1 week early | 64 | **75** |
| Detected at least 2 weeks early | 49 | 49 |
| Detected at least 3 weeks early | **37** | 35 |

THREE_HEAD detected 18 additional eligible onsets under the same Top-5 review budget.

It also detected 11 more onsets at least one week early.

However, THREE_HEAD did not improve detection at least two weeks early and performed slightly worse at three-week lead time.

This suggests the relative advantage is more evident near onset than at longer warning horizons.

### 5.3 Random-Selection Benchmark

The matched random-selection expectation was **88.37 detected episodes** under the Top-5 review budget and the same forecasting-opportunity structure.

THREE_HEAD detected 91 episodes.

This difference is small. Consequently, the analysis does not establish a robust episode-detection advantage over random selection.

The country-week prioritisation and distinct episode-detection findings should be interpreted separately.

## 6. Disease-Aware Model Interpretation

THREE_HEAD uses separate influenza, RSV and COVID-19 Random Forest models.

The combined country-week score is the maximum available eligible pathogen-head score.

Among its 549 Top-3 country-review positions, the winning heads were:

| Winning disease head | Top-3 review positions | Share |
|---|---:|---:|
| Influenza | 292 | 53.19% |
| RSV | 225 | 40.98% |
| COVID-19 | 32 | 5.83% |
| **Total** | **549** | **100%** |

The winning head identifies which pathogen model produced the highest ranking score. It does not establish which pathogen caused a future episode.

The three disease-head scores are not independently calibrated against one another. Differences in score distributions and surveillance coverage may affect which head dominates the final ranking.

## 7. Historical Model Selection and Governance

The project progressed through several related but distinct experimental stages.

| Research stage | Main contribution | Interpretation |
|---|---|---|
| Version 1 | Sharp-rise heuristic and Logistic Regression baseline | Initial benchmark on an earlier frozen cohort |
| Version 2 | Dynamic enrichment, RF59 development and four historical folds | Earlier evaluation protocol with different denominators |
| Version 3 | Bounded challenger comparison and formal Gate-4 evaluation | No challenger satisfied all frozen promotion requirements |
| Corrected temporal study (C4) | Matched RF59 versus THREE_HEAD comparison over seven historical periods | THREE_HEAD selected as final retrospective research model |

Metrics from these stages must not be combined into a single leaderboard without accounting for differences in cohort construction, targets, temporal assumptions and evaluation denominators.

The formal Version 3 Gate-4 decision was **NO PROMOTION**.

The previously registered RF59-D2 shadow reference remains unchanged.

**THREE_HEAD is the selected final research model, but it has not been formally promoted, registered or deployed.**

## 8. Limitations and Responsible Interpretation

The main limitations are:

1. **Retrospective reuse:** All seven corrected evaluation periods were previously inspected during development.
2. **Point-in-time availability:** Actual publication and revision histories were not fully reconstructed.
3. **Surveillance missingness:** Missing reports do not imply zero disease activity.
4. **Cross-country variation:** Testing capacity, surveillance coverage and reporting consistency differ.
5. **Pathogen differences:** Influenza/RSV laboratory positivity and COVID-19 reported cases are not directly equivalent signals.
6. **Score calibration:** Pathogen-specific scores are ranking signals, not calibrated outbreak probabilities.
7. **Longer-lead performance:** Detection improvements are not consistent at two- and three-week warning horizons.
8. **Operational validation:** Safe deployment, prospective accuracy, model monitoring and independent generalisation remain unverified.

The system should be considered a research decision-support prototype rather than an automatic disease-outbreak alerting system.

## 9. Recommended Next Steps

Before any operational promotion, the research would benefit from:

- Independently collected prospective evaluation periods.
- Historical source-publication timestamps and revision-aware data snapshots.
- More consistent influenza, RSV and COVID-19 surveillance coverage.
- Additional hospitalisation and clinical-severity indicators.
- Further analysis of pathogen-head score calibration and dominance.
- Robustness analysis across countries and surveillance-quality groups.
- Prospective monitoring of ranking performance, missingness and temporal drift.
- A clearly defined analyst-review workflow with human oversight.

Further performance optimisation should follow these improvements rather than replace proper evaluation and governance.

## 10. Reproducibility and Supporting Evidence
- [MLflow Experiment Tracking and Model Governance](MLFLOW_EVIDENCE.md)
The cleaned research notebooks provide the methodological and historical evidence supporting this report.

- [Exploratory Data Analysis](../notebooks/01_exploratory_data_analysis_clean.py)
- [Machine Learning Baselines](../notebooks/02_machine_learning_baselines_clean.py)
- [Data Enrichment](../notebooks/03_data_enrichment_final.py)
- [Model Selection](../notebooks/04_model_selection_final.py)
- [Corrected Temporal Modelling](../notebooks/05_corrected_temporal_modelling_final.py)
- [Main Project Documentation](../README.md)

The full research artifacts, historical experiment records and saved model evidence remain in Databricks.

The cleaned notebooks require authorised access to the original pinned datasets and existing saved artifacts; this GitHub repository alone does not reproduce the entire raw-data-to-trained-model workflow.

Additional figures, aggregate evaluation files and authentic MLflow experiment evidence can be published as supporting artifacts after verification and publication review.

---

*Research results verified against saved Databricks development artifacts on 2 November 2026. This report does not constitute a live public-health forecast or independent deployment validation.*
