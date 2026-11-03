# Feature Dictionary and Model Input Specification

**Project:** Humanitarian Health Intelligence  
**Research scope:** Retrospective respiratory-surveillance early warning  
**Selected final research model:** THREE_HEAD  
**Documentation date:** 3 November 2026

## 1. Overview

The project uses weekly national surveillance indicators for influenza, respiratory syncytial virus (RSV) and COVID-19.

The historical RF59 Random Forest comparator uses 59 predictors:

| Feature family | Count |
|---|---:|
| Original local surveillance features | 21 |
| WHO-region peer features | 12 |
| Seasonal calendar features | 2 |
| Retained dynamic features | 24 |
| **Total** | **59** |

The selected final research model, THREE_HEAD, uses three separate pathogen-specific Random Forest models rather than one combined 59-feature classifier.

The exact feature specifications below were reconstructed from the archived feature-engineering code and frozen model manifests.

## 2. Original Local Surveillance Features (21)

### Activity history — 15 features

The following five features are constructed separately for each respiratory stream.

| Feature suffix | Definition |
|---|---|
| `activity_current` | Latest eligible weekly activity observation |
| `activity_lag1` | Previous weekly activity observation |
| `mean_2w` | Mean of current and previous activity |
| `mean_4w` | Mean across four weekly observations |
| `change_1w` | Current activity minus previous activity |

For influenza, the exact columns are:

- `flu_activity_current`
- `flu_activity_lag1`
- `flu_mean_2w`
- `flu_mean_4w`
- `flu_change_1w`

For RSV:

- `rsv_activity_current`
- `rsv_activity_lag1`
- `rsv_mean_2w`
- `rsv_mean_4w`
- `rsv_change_1w`

For COVID-19:

- `covid_activity_current`
- `covid_activity_lag1`
- `covid_mean_2w`
- `covid_mean_4w`
- `covid_change_1w`

### Testing-volume context — 2 features

| Feature | Definition |
|---|---|
| `flu_tested_log1p` | log(1 + eligible influenza testing volume) |
| `rsv_tested_log1p` | log(1 + eligible RSV testing volume) |

COVID-19 testing-volume features were not constructed using an unsupported positivity denominator.

### Availability and coverage — 4 features

| Feature | Definition |
|---|---|
| `flu_available_now` | Whether current influenza activity is available |
| `rsv_available_now` | Whether current RSV activity is available |
| `covid_available_now` | Whether current COVID-19 activity is available |
| `disease_streams_observed_now` | Number of respiratory streams with available current activity |

Missing surveillance activity is not interpreted as zero disease activity.

## 3. WHO-Region Peer Features (12)

Each pathogen has four regional context features.

| Feature suffix | Meaning |
|---|---|
| `regional_activity` | Mean activity among other available countries in the same WHO region |
| `regional_change_common_peers_1w` | Mean weekly activity change among peers with paired observations |
| `regional_peer_count` | Number of peers contributing current activity |
| `regional_common_peer_count` | Number of peers contributing paired weekly change |

The original feature names use these prefixes:

- `flu_regional_*`
- `rsv_regional_*`
- `covid_regional_*`

For the historical frozen RF59 input specification, all 12 regional predictors use a two-calendar-week delay.

Their final model input names therefore append:

`__delay2w`

For example:

`rsv_regional_activity__delay2w`

This represents a conservative regional reporting-delay assumption. It does not establish that the underlying observations were actually published by every historical forecast date.

## 4. Seasonal Calendar Features (2)

| Feature | Definition |
|---|---|
| `week_sin` | Sine transformation of ISO week-of-year |
| `week_cos` | Cosine transformation of ISO week-of-year |

The historical construction used a 52.18-week seasonal cycle.

These features provide cyclic seasonal context without introducing artificial discontinuities between the beginning and end of a calendar year.

## 5. Retained Dynamic Features (24)

The Version 2 enrichment stage evaluated 27 candidate dynamic predictors: nine candidates per respiratory stream.

Three sparse `threshold_ratio` candidates were rejected, leaving 24 retained predictors.

### Exact retained feature definitions

Each row below represents three features: one for influenza, one for RSV and one for COVID-19.

| Suffix | Definition |
|---|---|
| `threshold_distance` | Current activity minus the past-only historical 80th percentile |
| `historical_percentile` | Fraction of earlier non-missing observations at or below current activity |
| `slope_4w` | Four-week least-squares trend using current and three preceding observations |
| `acceleration_1w` | Most recent weekly change minus the previous weekly change |
| `consecutive_rising_weeks` | Count of immediately consecutive increasing transitions, maximum four |
| `max_4w` | Maximum activity across four completed observations |
| `min_4w` | Minimum activity across four completed observations |
| `volatility_4w` | Population standard deviation across four completed observations |

The exact 24 names are generated by combining each suffix with the prefixes `flu_`, `rsv_` and `covid_`.

For example:

- `flu_threshold_distance`
- `rsv_threshold_distance`
- `covid_threshold_distance`
- `flu_historical_percentile`
- `rsv_historical_percentile`
- `covid_historical_percentile`

The same prefix rule applies to all eight retained suffixes.

### Past-only construction safeguard

Historical reference statistics are calculated from strictly earlier, non-missing observations for each country and pathogen.

At least 26 earlier observations are required for the historical 80th-percentile reference and percentile rank.

The current observation is added to the reference history only after its own historical-reference features have been calculated.

Four-week rolling features require the necessary observation history. Missing observations are not silently replaced by zero.

## 6. THREE_HEAD Final Research Model

THREE_HEAD fits three independent pathogen-specific Random Forest classifiers.

Each model selects predictors from the frozen RF59 input manifest by pathogen prefix and adds the two seasonal features.

| Model head | Input selection | Predictor count |
|---|---|---:|
| Influenza | `flu_*` plus `week_sin`, `week_cos` | 21 |
| RSV | `rsv_*` plus `week_sin`, `week_cos` | 21 |
| COVID-19 | `covid_*` plus `week_sin`, `week_cos` | 20 |

These counts reflect the archived disease-aware feature-selection rule.

The global feature `disease_streams_observed_now` is part of the RF59 panel but is not included in an individual pathogen head under this prefix-selection rule.

### Pathogen-specific training and scoring

Each head is fitted using eligible historical observations with known pathogen-specific targets.

A head only produces a forecast score when its pathogen is within the issue-time eligible surveillance scope.

The combined country-week score is:

`THREE_HEAD_score = max(available eligible pathogen-head scores)`

Missing pathogen-head scores are not replaced with zero.

The combined score is an uncalibrated ranking signal, not a calibrated outbreak probability.

## 7. Random Forest Configuration

The corrected model comparison used the established Random Forest configuration:

| Hyperparameter | Value |
|---|---|
| `n_estimators` | 500 |
| `min_samples_leaf` | 10 |
| `max_features` | `sqrt` |
| `max_depth` | `None` |
| `class_weight` | `None` |
| `random_state` | 42 |
| `n_jobs` | -1 |
| Missing-value handling | Median imputation with empty-feature preservation |

The THREE_HEAD pathogen-specific models use the same Random Forest settings, but have different feature subsets and pathogen-specific targets.

## 8. Target and Outcome Definition

The prediction task concerns elevated respiratory surveillance activity during an approximately four-week assessment window.

Country- and pathogen-specific elevated-activity thresholds are derived from earlier historical observations using an 80th-percentile reference.

The study distinguishes several generations of target definitions:

| Study stage | Target interpretation |
|---|---|
| Version 1 | `high_risk_next_4w` |
| Version 2 | `high_risk_next_4w_foldaware` |
| Corrected C4 | At least one in-scope respiratory stream exceeds its historical activity threshold during the four-week outcome window |

In the corrected study, a known positive outcome requires an observed qualifying elevation.

A known negative outcome requires sufficient complete follow-up across relevant in-scope streams.

Otherwise the outcome remains unknown.

Pathogen-specific targets are used to fit individual THREE_HEAD models; these are distinct from the combined country-week evaluation outcome.

These targets represent surveillance-defined activity elevation, not official outbreak declarations or patient-level disease risks.

## 9. Temporal Availability and Leakage Controls

The corrected study used several conservative timing assumptions and validation safeguards:

- Surveillance predictors follow an assumed 21-day availability offset.
- Training-label maturity is conservatively assumed at 42 days.
- Historical training observations precede their corresponding evaluation periods.
- Country ranking is determined before future outcomes are inspected.
- Future outcome availability does not determine whether a country enters the ranking queue.
- Historical reference features are built from earlier observations.
- Missing future outcomes remain unknown, not negative.

These assumptions reduce some forms of temporal leakage but do not prove perfect point-in-time correctness.

Historical publication and revision timestamps were not fully reconstructable.

## 10. Source Provenance and Limitations

The activity streams use publicly reported aggregate WHO surveillance.

| Stream | Primary measurement |
|---|---|
| Influenza | Sentinel laboratory positivity |
| RSV | Sentinel laboratory positivity |
| COVID-19 | log(1 + nonnegative weekly reported cases) |

Influenza and RSV positivity require valid reported numerator and tested-denominator information.

Negative COVID-19 reporting corrections are not converted into artificial positive activity.

Different pathogen measurements are not directly interchangeable.

Reporting completeness and testing practices vary across countries and periods.

## 11. Frozen Specification and Reproducibility

The historical frozen RF59 feature manifest is retained in Databricks:

`humanitarian.model_ready.health_risk_v2_finalsel01e_rf59_feature_manifest_20261013_142603`

The manifest contains the ordered feature list, source names, final model input names and feature provenance.

The cleaned model-selection notebook checks the 59-feature contract and its saved feature-list fingerprint.

This document explains the observed feature specification; it does not replace the frozen source manifest or underlying historical model artifacts.

The corrected THREE_HEAD comparison is reproduced in:

[Notebook 05 — Corrected Temporal Modelling](../notebooks/05_corrected_temporal_modelling_final.py)

For the wider research context:

- [Main Project README](../README.md)
- [Final Evaluation Report](../results/README.md)
- [MLflow Evidence](../results/MLFLOW_EVIDENCE.md)

**Model governance:** THREE_HEAD is the selected final retrospective research model. It has not been formally promoted, registered or deployed. The existing RF59-D2 shadow reference remains unchanged.

---

*Feature definitions reconstructed from the archived development source and verified notebook specifications. The seven-period comparison is retrospective development evidence, not independent prospective validation.*
