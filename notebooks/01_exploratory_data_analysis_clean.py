# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "6"
# ///
# MAGIC %md
# MAGIC # Humanitarian Health Intelligence — Exploratory Data Analysis
# MAGIC
# MAGIC ## Purpose
# MAGIC This notebook documents the **problem-framing and source-selection stage** of the project.
# MAGIC
# MAGIC The work began with a country-level cholera early-warning concept. Exploratory analysis showed that the available cholera extract was a small cumulative country snapshot with unequal reporting periods, making it unsuitable for the intended short-horizon forecasting task. The project therefore pivoted to weekly WHO respiratory surveillance.
# MAGIC
# MAGIC The objective of this cleaned notebook is to show that decision clearly and reproducibly:
# MAGIC
# MAGIC 1. assess whether the initial cholera data support temporal forecasting;
# MAGIC 2. summarise the contextual data sources explored;
# MAGIC 3. assess WHO influenza, RSV and COVID-19 surveillance;
# MAGIC 4. document the data-quality principles carried forward into modelling.
# MAGIC
# MAGIC **Scope note:** detailed feature engineering, model fitting and the later corrected temporal evaluation are intentionally kept out of this notebook. They are handled in subsequent notebooks.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Setup and source locations
# MAGIC
# MAGIC The original source files remain unchanged. This notebook performs read-only exploratory checks.

# COMMAND ----------

from pyspark.sql import functions as F
import numpy as np
import pandas as pd

PATHS = {
    "cholera": (
        "/Volumes/humanitarian/source/source_files/WHO/"
        "cholera_adm0_public (5) (1).csv"
    ),
    "flunet": (
        "/Volumes/humanitarian/source/source_files/WHO/"
        "VIW_FNT.csv"
    ),
    "flunet_metadata": (
        "/Volumes/humanitarian/source/source_files/WHO/"
        "VIW_FLU_METADATA.csv"
    ),
    "fluid": (
        "/Volumes/humanitarian/source/source_files/WHO/"
        "VIW_FID_EPI.csv"
    ),
    "covid": (
        "/Volumes/humanitarian/source/source_files/WHO/"
        "WHO-COVID-19-global-data.csv"
    ),
}

print("Source paths configured:", len(PATHS))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Initial cholera feasibility assessment
# MAGIC
# MAGIC The original question was whether public humanitarian and health data could support a short-term country-level cholera-risk model.
# MAGIC
# MAGIC Before adding predictors, the outcome source itself must support a temporal forecasting design. The checks below therefore focus on observation count, geographic coverage, reporting windows and comparability.

# COMMAND ----------

df_cholera = (
    spark.read
    .format("csv")
    .option("header", "true")
    .option("inferSchema", "true")
    .load(PATHS["cholera"])
)

cholera_summary = df_cholera.select(
    F.count("*").alias("rows"),
    F.countDistinct("iso_3_code").alias("countries"),
    F.min("first_epiwk").alias("earliest_reporting_start"),
    F.max("last_epiwk").alias("latest_reporting_end"),
    F.sum(F.col("case_total").isNull().cast("int")).alias("missing_cases"),
    F.sum(F.col("death_total").isNull().cast("int")).alias("missing_deaths"),
).toPandas()

display(cholera_summary)

duplicate_rows = df_cholera.count() - df_cholera.distinct().count()

reporting_duration = (
    df_cholera
    .withColumn(
        "reporting_days",
        F.datediff(F.col("last_epiwk"), F.col("first_epiwk")) + F.lit(1)
    )
    .select(
        "adm0_name",
        "iso_3_code",
        "case_total",
        "death_total",
        "reporting_days",
    )
    .orderBy("reporting_days")
)

print("Exact duplicate rows:", duplicate_rows)
display(reporting_duration)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2.1 Cholera feasibility conclusion
# MAGIC
# MAGIC The cholera extract was useful for understanding burden and possible contextual predictors, but it was **not a defensible outcome dataset for the intended early-warning task**:
# MAGIC
# MAGIC - only 28 country observations were present;
# MAGIC - each country was represented by cumulative cases/deaths rather than a regular country-week time series;
# MAGIC - countries covered different reporting durations;
# MAGIC - raw totals were therefore not directly comparable without additional temporal and population adjustment;
# MAGIC - a genuine 28–30 day future target could not be constructed from this snapshot alone.
# MAGIC
# MAGIC This was the key reason for changing the modelling problem rather than forcing an unsuitable dataset into a forecasting framework.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Contextual sources explored during the cholera phase
# MAGIC
# MAGIC Several public sources were assessed for potential explanatory context:
# MAGIC
# MAGIC | Source | Candidate information | Main EDA conclusion |
# MAGIC |---|---|---|
# MAGIC | WHO GHO WASH | drinking water, sanitation, open defecation | useful structural vulnerability context, but relatively slow-moving |
# MAGIC | CHIRPS | rainfall | potentially relevant dynamic exposure, but the available files did not span the full later modelling period consistently |
# MAGIC | UNICEF/WUENIC | DTP3 and immunisation continuity | useful health-system proxy, not a cholera-specific exposure |
# MAGIC | IDMC | conflict/disaster displacement | potentially useful humanitarian context, but highly skewed and partly incomplete |
# MAGIC | WorldPop / World Bank | population | useful for normalisation and demographic context |
# MAGIC | WHO GHO capacity | hospital beds, medical doctors | useful structural context, but historical publication timing was not fully reconstructable |
# MAGIC
# MAGIC These explorations were informative, but the most important limitation remained the absence of a suitable longitudinal cholera outcome.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Pivot to weekly respiratory surveillance
# MAGIC
# MAGIC WHO respiratory data provided the temporal structure the project required.
# MAGIC
# MAGIC The core sources evaluated were:
# MAGIC
# MAGIC - **FluNet** — weekly laboratory surveillance for influenza and RSV;
# MAGIC - **FluID** — weekly epidemiological/clinical respiratory surveillance;
# MAGIC - **WHO COVID-19** — weekly country-level COVID-19 reporting.
# MAGIC
# MAGIC Weekly surveillance allowed the unit of analysis to become **country × forecast week**, which is much better aligned with a four-week early-warning objective than the original cumulative cholera snapshot.

# COMMAND ----------

# MAGIC %md
# MAGIC ### 4.1 FluNet structure and reporting coverage

# COMMAND ----------

READ_OPTIONS = {
    "dtype": "string",
    "encoding": "utf-8-sig",
    "keep_default_na": False,
}

flunet_columns = [
    "COUNTRY_CODE",
    "COUNTRY_AREA_TERRITORY",
    "ISO_WEEKSTARTDATE",
    "ORIGIN_SOURCE",
    "SPEC_PROCESSED_NB",
    "INF_ALL",
    "INF_NEGATIVE",
    "RSV_PROCESSED",
    "RSV",
]

flunet = pd.read_csv(
    PATHS["flunet"],
    usecols=flunet_columns,
    **READ_OPTIONS,
)

for col in [
    "COUNTRY_CODE",
    "COUNTRY_AREA_TERRITORY",
    "ORIGIN_SOURCE",
]:
    flunet[col] = flunet[col].str.strip()

flunet["week_start"] = pd.to_datetime(
    flunet["ISO_WEEKSTARTDATE"],
    errors="coerce",
)

print("FluNet rows:", len(flunet))
print(
    "Countries/territories:",
    flunet.loc[flunet["COUNTRY_CODE"].ne(""), "COUNTRY_CODE"].nunique(),
)
print("First week:", flunet["week_start"].min())
print("Last week:", flunet["week_start"].max())

print("\nReporting-source categories")
print(
    flunet["ORIGIN_SOURCE"]
    .value_counts(dropna=False)
    .to_string()
)

# Exact duplicate copies are audit noise, not separate observations.
exact_duplicate = flunet.duplicated(keep="first")
print("\nExact duplicate copies:", int(exact_duplicate.sum()))

flunet_clean = flunet.loc[~exact_duplicate].copy()

# The primary laboratory analysis used sentinel surveillance.
sentinel = flunet_clean.loc[
    flunet_clean["ORIGIN_SOURCE"].eq("SENTINEL")
].copy()

for col in [
    "SPEC_PROCESSED_NB",
    "INF_ALL",
    "INF_NEGATIVE",
    "RSV_PROCESSED",
    "RSV",
]:
    sentinel[col] = pd.to_numeric(
        sentinel[col].str.strip().replace("", pd.NA),
        errors="coerce",
    )

print("Sentinel rows after exact-deduplication:", len(sentinel))

# COMMAND ----------

# MAGIC %md
# MAGIC ### 4.2 Missing values are not automatically zero
# MAGIC
# MAGIC A central EDA finding was that surveillance absence and zero disease activity are not equivalent.
# MAGIC
# MAGIC For influenza, a missing `INF_ALL` value was only considered an inferred zero when:
# MAGIC
# MAGIC - processed specimens were positive in number;
# MAGIC - influenza-negative specimens were observed; and
# MAGIC - `INF_NEGATIVE == SPEC_PROCESSED_NB`.
# MAGIC
# MAGIC A later sensitivity analysis showed that inferred zeros materially changed country coverage and historical thresholds. The primary data-engineering path therefore moved to **explicitly reported influenza observations only**, retaining inferred-zero records as sensitivity information rather than primary measurements.

# COMMAND ----------

processed = sentinel["SPEC_PROCESSED_NB"]
reported_positive = sentinel["INF_ALL"]
reported_negative = sentinel["INF_NEGATIVE"]

valid_reported_flu = (
    processed.gt(0)
    & reported_positive.notna()
    & reported_positive.ge(0)
    & reported_positive.le(processed)
)

inferred_zero_flu = (
    reported_positive.isna()
    & processed.gt(0)
    & reported_negative.notna()
    & reported_negative.eq(processed)
)

rsv_valid = (
    sentinel["RSV_PROCESSED"].gt(0)
    & sentinel["RSV"].notna()
    & sentinel["RSV"].ge(0)
    & sentinel["RSV"].le(sentinel["RSV_PROCESSED"])
)

quality_summary = pd.DataFrame([
    {
        "stream": "influenza",
        "reported_valid_rows": int(valid_reported_flu.sum()),
        "inferred_zero_rows": int(inferred_zero_flu.sum()),
        "usable_countries_reported_only": int(
            sentinel.loc[valid_reported_flu, "COUNTRY_CODE"].nunique()
        ),
    },
    {
        "stream": "RSV",
        "reported_valid_rows": int(rsv_valid.sum()),
        "inferred_zero_rows": 0,
        "usable_countries_reported_only": int(
            sentinel.loc[rsv_valid, "COUNTRY_CODE"].nunique()
        ),
    },
])

display(quality_summary)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 4.3 COVID-19 and FluID coverage
# MAGIC
# MAGIC COVID-19 offered broad weekly geographic coverage, although later reporting became less complete and negative weekly case/death values could represent retrospective corrections. Those negative values were therefore flagged rather than silently converted to zero.
# MAGIC
# MAGIC FluID supplied useful clinical/severity context (ILI, SARI, deaths, reporting sites and qualitative indicators), but coverage varied substantially by measure. It was therefore treated as supplementary context rather than a universally required core stream.

# COMMAND ----------

covid = pd.read_csv(
    PATHS["covid"],
    dtype="string",
    encoding="utf-8-sig",
    keep_default_na=False,
)

covid["report_date"] = pd.to_datetime(
    covid["Date_reported"],
    errors="coerce",
)

covid["new_cases_num"] = pd.to_numeric(
    covid["New_cases"],
    errors="coerce",
)

covid["new_deaths_num"] = pd.to_numeric(
    covid["New_deaths"],
    errors="coerce",
)

fluid_header = pd.read_csv(
    PATHS["fluid"],
    nrows=0,
    dtype="string",
    encoding="utf-8-sig",
).columns.tolist()

print("COVID rows:", len(covid))
print(
    "COVID countries/territories:",
    covid["Country_code"].replace("", pd.NA).nunique(),
)
print("COVID first report:", covid["report_date"].min())
print("COVID last report:", covid["report_date"].max())
print("Negative weekly case corrections:", int(covid["new_cases_num"].lt(0).sum()))
print("Negative weekly death corrections:", int(covid["new_deaths_num"].lt(0).sum()))

print("\nFluID fields available:", len(fluid_header))
print(
    "Key clinical fields present:",
    [
        c for c in [
            "ILI_CASE",
            "ILI_OUTPATIENTS",
            "SARI_CASE",
            "SARI_INPATIENTS",
            "SARI_DEATHS",
            "INTENSITY",
            "TREND",
        ]
        if c in fluid_header
    ],
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Data harmonisation decisions carried forward
# MAGIC
# MAGIC The EDA established several rules that became important for later data engineering and modelling:
# MAGIC
# MAGIC 1. **ISO3 as canonical geography** — source-specific country codes were harmonised before integration.
# MAGIC 2. **Monday weekly calendar** — weekly reporting dates were aligned to a common week-start convention.
# MAGIC 3. **Missing ≠ zero** — unavailable surveillance remained missing unless a zero could be justified from source fields.
# MAGIC 4. **Disease streams stay separate** — influenza, RSV and COVID-19 were not combined as raw counts.
# MAGIC 5. **Surveillance quality is visible** — testing volumes, stream availability and peer counts were retained as quality/context signals.
# MAGIC 6. **Non-country entities are excluded** — special subnational surveillance codes were not treated as independent countries in a country-level model.
# MAGIC 7. **Regional signals exclude the focal country** — later regional pressure features used leave-one-country-out aggregation.
# MAGIC 8. **Historical availability matters** — contextual variables without reconstructable publication timing were not allowed to dominate the primary retrospective benchmark.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. EDA decision: respiratory surveillance becomes the modelling core
# MAGIC
# MAGIC The exploratory phase led to a clear change in project direction.
# MAGIC
# MAGIC **Rejected as the primary forecasting outcome**
# MAGIC - the cumulative country-level cholera snapshot, because it did not provide the repeated temporal observations required for a defensible 28–30 day forecast.
# MAGIC
# MAGIC **Selected as the modelling core**
# MAGIC - weekly influenza surveillance;
# MAGIC - weekly RSV surveillance;
# MAGIC - weekly COVID-19 surveillance;
# MAGIC - calendar/seasonality information;
# MAGIC - regional respiratory pressure and surveillance-quality features where available.
# MAGIC
# MAGIC The historical Version 1.1 data-engineering stage subsequently produced a 40-country respiratory development package. That package was an important checkpoint, but later notebooks further corrected the temporal availability/label design. Therefore, **the final performance claims should always come from the corrected temporal-modelling notebook, not from this EDA notebook**.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Handover to machine learning
# MAGIC
# MAGIC ### What this notebook establishes
# MAGIC
# MAGIC - The project pivot from cholera to respiratory surveillance was evidence-driven.
# MAGIC - Weekly WHO respiratory surveillance is substantially better matched to the early-warning objective.
# MAGIC - Influenza, RSV and COVID-19 require separate handling because coverage and measurement processes differ.
# MAGIC - Missingness and reporting quality are part of the modelling problem, not preprocessing noise to be erased.
# MAGIC - The analysis is retrospective: exact historical publication/revision vintages are not fully reconstructed.
# MAGIC
# MAGIC ### What is intentionally deferred
# MAGIC
# MAGIC - final feature engineering;
# MAGIC - temporal cohort/label correction;
# MAGIC - Random Forest / Logistic Regression benchmarking;
# MAGIC - disease-specific model heads;
# MAGIC - model selection and uncertainty analysis.
# MAGIC
# MAGIC Those steps are documented in the subsequent modelling notebooks.
