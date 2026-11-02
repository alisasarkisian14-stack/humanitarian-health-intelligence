# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "6"
# ///
# MAGIC %md
# MAGIC # 05 — Corrected Temporal Modelling and Final Research Comparison
# MAGIC
# MAGIC **Humanitarian Respiratory Early Warning | 30 October 2026 corrected audit | Databricks read-only chapter**
# MAGIC
# MAGIC **Purpose.** Reproduce `CORRECTED-10A` using the **already saved** seven-period RF59 scores and three pathogen-specific Random Forest head scores. Recheck the matched cohort, AUC/AP, within-week ranking, fixed analyst budgets, distinct disease-onset episodes, score composition and paired uncertainty. **Do not fit, tune, register or promote any model.**
# MAGIC
# MAGIC **Research lineage.** Notebook 01 establishes the surveillance-data rationale; Notebook 02 reproduces the V1 Logistic Regression baseline; Notebook 03 verifies V2 data enrichment; Notebook 04 audits formal V2/V3 model selection and the original V3 Gate-4 **no-promotion** decision; **this notebook** evaluates the *later*, explicitly separate corrected research comparison.
# MAGIC
# MAGIC **Models and scores.** `RF59` is the saved matched comparator. `THREE_HEAD` is the maximum of the **available** influenza, RSV and COVID disease-head scores. The combination is an **uncalibrated prioritisation score**, not an outbreak probability and not a sum or independence-based joint probability. `SELECTED` is a distinct, historically selected combination policy evaluated for sensitivity; it must not be called the same model as THREE_HEAD.
# MAGIC
# MAGIC **Key distinction.** A later research comparison favouring THREE_HEAD in seven-period ROC-AUC **does not retroactively change** Notebook 04's formal `GATE4_CLOSED_NO_PROMOTION` decision, or demonstrate that THREE_HEAD is registered or deployment-approved. The archived RF59-D2 registered reference remains a separate object. The `2026_H1` period **has already been consumed in historical research** and cannot be portrayed as an untouched prospective/independent holdout.
# MAGIC
# MAGIC **Limits.** Surveillance-defined elevated activity is not a clinically verified outbreak; some observations and future labels are missing; source-publication vintages and genuine point-in-time availability have not been fully reconstructed. Country-week observations are dependent over time. Report chronology and uncertainty, and do not use these results to conduct new post-hoc model searches.
# MAGIC
# MAGIC **Safety and provenance.** Source tables are **explicitly allowlisted** and read as Delta **version 0**. No Delta writes, MLflow operations, model fitting/scoring, registrations, feature rebuilds, or source refreshes. Keep the original `05_corrected_temporal_modelling.py` as the unchanged full analysis archive. This notebook's metrics are *new in-memory calculations of previously saved predictions*, not newly trained forecasts.

# COMMAND ----------

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

CATALOG = "humanitarian.model_ready"
TABLES = {
    "rf59": "health_risk_c4_02_20261024_211736_33602e_rf59_v1_predictions",
    "heads": "health_risk_c4_06_20261024_211736_33602e_disease_heads_v1_rankings",
    "definitions": "health_risk_c4_01_20261024_211736_33602e_definitions",
    "activity": "health_risk_v2_finaleval04b_dynamic_extended_20261013_152927",
    "gate4": "health_risk_v3_gate4_closeout_v1",
}
FOLDS = ["2023_H1", "2023_H2", "2024_H1", "2024_H2",
         "2025_H1", "2025_H2", "2026_H1"]
HEADS = ["flu_head_score", "rsv_head_score", "covid_head_score"]
KEYS = ["fold", "iso3", "prediction_date"]
FOLD_DATES = [
    ("2023_H1", "2023-01-02", "2023-07-03"),
    ("2023_H2", "2023-07-03", "2024-01-01"),
    ("2024_H1", "2024-01-01", "2024-07-01"),
    ("2024_H2", "2024-07-01", "2025-01-06"),
    ("2025_H1", "2025-01-06", "2025-07-07"),
    ("2025_H2", "2025-07-07", "2026-01-05"),
    ("2026_H1", "2026-01-05", "2026-07-06"),
]

def require(condition, message):
    if not bool(condition):
        raise RuntimeError(message)

def table_name(label):
    require(label in TABLES, "Unapproved source label: " + str(label))
    return CATALOG + "." + TABLES[label]

def quoted_name(name):
    return ".".join("`" + p.replace("`", "``") + "`" for p in name.split("."))

def read_v0(label, columns, limit):
    """Exact archived Delta-version read. No discovery, fitting or writes."""
    name = table_name(label)
    require(spark.catalog.tableExists(name), "Missing archived table: " + name)
    q = quoted_name(name)
    history = spark.sql(f"DESCRIBE HISTORY {q} LIMIT 1").first()
    require(history is not None and int(history["version"]) >= 0,
            "Cannot verify Delta history for " + name)
    data = spark.sql(f"SELECT * FROM {q} VERSION AS OF 0")
    require(set(columns).issubset(set(data.columns)),
            "Missing required archived columns from " + name)
    output = data.select(*columns).limit(limit + 1).toPandas()
    require(len(output) <= limit, "Unexpectedly large archived source: " + name)
    return output

for label in TABLES:
    require(spark.catalog.tableExists(table_name(label)),
            "Missing archived research artifact: " + table_name(label))

gate4 = read_v0("gate4", ["status", "promoted_model", "retained_reference"], 2)
require(len(gate4) == 1, "Expected exactly one formal Gate-4 closeout record.")
require(gate4.iloc[0]["status"] == "GATE4_CLOSED_NO_PROMOTION", "Gate-4 status changed.")
require(gate4.iloc[0]["promoted_model"] == "NONE", "Unexpected Gate-4 promotion.")
require(gate4.iloc[0]["retained_reference"] == "RF59_D2_MATCHED_REFERENCE",
        "Archived Gate-4 reference identity changed.")

print("NOTEBOOK 05 PRE-FLIGHT: PASS")
print("Archived RF59 and disease-head score tables: FOUND")
print("Archived Gate-4 decision: NO PROMOTION — VERIFIED")
print("Historical periods: 7, including 2026_H1 as consumed development evidence")
print("No model fitted, registered or promoted; no Delta writes.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Exact matched saved-score reconstruction
# MAGIC
# MAGIC The two version-0 C4 artifacts contain the **same historical forecasting opportunities**, keyed by `(fold, iso3, prediction_date)`. First compare exact row identities and known/unknown outcome labels. Then independently reconstruct the three-head maximum and demand exact agreement with the archived disease-head score (tolerance `1e-12`). A missing disease head is **not** replaced with zero; at least one usable score is required. The source has seven periods and **2,491** saved country-weeks.
# MAGIC
# MAGIC This is not a model refit: it simply verifies previously generated scores and target alignment. All seven periods, including 2026_H1, are **retrospective research evidence**.

# COMMAND ----------

rf = read_v0("rf59", KEYS + ["target", "score"], 3000)
heads = read_v0("heads", KEYS + ["target", "score"] + HEADS, 3000)

require(len(rf) == len(heads) == 2491, "Archived C4 queue sizes differ from 2,491.")
for frame in (rf, heads):
    frame["prediction_date"] = pd.to_datetime(frame["prediction_date"], errors="raise").dt.normalize()
    require(not frame[KEYS].isna().any().any(), "Null C4 identity key.")
    require(not frame.duplicated(KEYS).any(), "Duplicate C4 identity key.")
    require(set(frame["fold"]) == set(FOLDS), "Historical fold set changed.")
    frame["target"] = pd.to_numeric(frame["target"], errors="raise")
    require(frame["target"].dropna().isin([0, 1]).all(), "Invalid historical outcome label.")

matched = heads.merge(rf, on=KEYS, how="outer", indicator=True,
                      validate="one_to_one", suffixes=("_head", "_rf"))
require(len(matched) == 2491 and matched["_merge"].eq("both").all(),
        "Saved model prediction keys are not identical.")
require(np.allclose(matched["target_head"].to_numpy(float),
                    matched["target_rf"].to_numpy(float), equal_nan=True,
                    rtol=0, atol=0), "Saved labels differ across models.")

matched["target"] = matched["target_rf"]
head_array = matched[HEADS].apply(pd.to_numeric, errors="raise").to_numpy(float)
require(not np.isinf(head_array).any(), "An archived disease-head score is infinite.")
safe_head_array = np.where(np.isfinite(head_array), head_array, -np.inf)
require(np.isfinite(safe_head_array).any(axis=1).all(),
        "A saved country-week has no usable disease-head score.")
three_score = safe_head_array.max(axis=1)
matched["rf59_score"] = pd.to_numeric(matched["score_rf"], errors="raise")
matched["three_score"] = three_score
require(np.allclose(three_score, matched["score_head"].to_numpy(float),
                    rtol=0, atol=1e-12), "Reconstructed THREE_HEAD differs from saved score.")
require(np.isfinite(matched[["rf59_score", "three_score"]].to_numpy(float)).all(),
        "Non-finite saved model score.")
require(matched[["rf59_score", "three_score"]].ge(0).all().all() and
        matched[["rf59_score", "three_score"]].le(1).all().all(),
        "Scores outside the expected RF vote-score range [0,1].")
matched["winning_head"] = np.array(["flu", "rsv", "covid"])[
    np.argmax(safe_head_array, axis=1)]

def score_breakdown(x):
    return pd.Series({"country_weeks":len(x),
                      "known_outcomes":int(x["target"].notna().sum()),
                      "unknown_outcomes":int(x["target"].isna().sum()),
                      "known_positive":int(x["target"].eq(1).sum()),
                      "forecast_weeks":x["prediction_date"].nunique()})

coverage = pd.DataFrame([{"fold":fold, **score_breakdown(g).to_dict()}
                         for fold,g in matched.groupby("fold",sort=True)])
for fold, start, end in FOLD_DATES:
    actual = pd.DatetimeIndex(sorted(matched.loc[matched.fold.eq(fold),
                                                "prediction_date"].unique()))
    expected = pd.date_range(start, pd.Timestamp(end)-pd.Timedelta(weeks=1), freq="W-MON")
    require(actual.equals(expected), "Incomplete saved forecast calendar: " + fold)
print("C4 MATCHED SCORE IDENTITY: PASS")
print("Saved matched country-weeks:", len(matched), "| Periods:", len(FOLDS))
print("THREE_HEAD = maximum available pathogen head: EXACT SAVED PARITY")
display(coverage)
print("Unknown outcomes remain unknown, never recoded to negative.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Discrimination: exact seven-period, matched-head comparison
# MAGIC
# MAGIC Compute **ROC-AUC** and **average precision (AP)** on *known* outcomes, using exactly the same rows for RF59 and THREE_HEAD. AP depends on each fold's prevalence; report that prevalence. Also calculate within-week ROC-AUC, weighting each informative forecast week by the number of positive-negative pairs; weeks with only one observed class are not assigned an invented AUC.
# MAGIC
# MAGIC The original `CORRECTED-10A` seven-period THREE_HEAD fold ROC-AUC values, rounded to four decimals, were: **0.6137, 0.6354, 0.7370, 0.7342, 0.7848, 0.8028, 0.7229**. This cleaned version asserts numerical parity, rather than substituting the older four-fold RF59-era results. The resulting seven-period mean is approximately **0.7187**; RF59 was approximately **0.6917** (difference **+0.0270**). These are historical comparisons, not independently validated live-forecast performance.

# COMMAND ----------

THREE_AUC_EXPECTED = {
    "2023_H1": .6137, "2023_H2": .6354, "2024_H1": .7370,
    "2024_H2": .7342, "2025_H1": .7848, "2025_H2": .8028,
    "2026_H1": .7229,
}

def within_week_auc(frame, score):
    auc_weighted, n_pairs = 0.0, 0
    for _, week in frame.loc[frame.target.notna()].groupby("prediction_date"):
        y = week["target"].astype(int)
        p, n = int(y.sum()), int(len(y)-y.sum())
        if p and n:
            pairs = p*n
            auc_weighted += roc_auc_score(y, week[score])*pairs
            n_pairs += pairs
    return float(auc_weighted/n_pairs) if n_pairs else np.nan

metric_rows = []
for fold, block in matched.groupby("fold", sort=True):
    known = block.loc[block.target.notna()]
    y = known["target"].astype(int)
    require(y.nunique() == 2, "Single-class or empty evaluation period: " + fold)
    row = {"fold":fold, "known_n":len(known), "positive_n":int(y.sum()),
           "prevalence":float(y.mean())}
    for label, score in [("rf59", "rf59_score"), ("three", "three_score")]:
        row[f"{label}_auc"] = float(roc_auc_score(y, known[score]))
        row[f"{label}_ap"] = float(average_precision_score(y, known[score]))
        row[f"{label}_within_week_auc"] = within_week_auc(block,score)
    require(abs(row["three_auc"]-THREE_AUC_EXPECTED[fold])<=.00015,
            "CORRECTED-10A THREE_HEAD AUC parity failed: " + fold)
    row["delta_auc"] = row["three_auc"]-row["rf59_auc"]
    row["delta_ap"] = row["three_ap"]-row["rf59_ap"]
    row["delta_within_week_auc"] = (row["three_within_week_auc"]
                                    -row["rf59_within_week_auc"])
    metric_rows.append(row)
fold_metrics = pd.DataFrame(metric_rows)
mean_auc_three = float(fold_metrics.three_auc.mean())
mean_auc_rf59 = float(fold_metrics.rf59_auc.mean())
require(abs(mean_auc_three-.7187)<.00015,
        "Recorded seven-period THREE_HEAD mean AUC did not reproduce.")
require(abs(mean_auc_rf59-.6917)<.00020,
        "Recorded seven-period RF59 mean AUC did not reproduce.")

print("CORRECTED-10A FOLD AUC / AP: PASS")
display(fold_metrics.round(4))
print("Seven-period macro-mean ROC-AUC — RF59:",round(mean_auc_rf59,4),
      "| THREE_HEAD:",round(mean_auc_three,4))
print("Paired macro-mean ROC-AUC difference:",round(float(fold_metrics.delta_auc.mean()),4))
print("Seven-period macro-mean AP difference:",round(float(fold_metrics.delta_ap.mean()),4))
print("AUC fold wins:",int(fold_metrics.delta_auc.gt(0).sum()),"/7")
print("This seven-period comparison is not the older four-fold V2 development estimate.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Separate historically chosen score-combination policy
# MAGIC
# MAGIC The original corrected audit additionally evaluated an alternative, chronological **policy** called `SELECTED`. Its rule was pre-recorded for each historical period: THREE_HEAD in 2023 H1/H2; flu/RSV preferred when available in 2024 H1/H2; and a 50%-weighted COVID head in 2025 H1/H2 and 2026 H1. These are **score combinations of saved models**, not additional fitted classifiers. They were investigated as a sensitivity to disease-head scale changes, not proof of production readiness.
# MAGIC
# MAGIC Verify the separately recorded `SELECTED` AUC values. **Do not confuse `SELECTED` with the default `THREE_HEAD` score or with an approved registered model.**

# COMMAND ----------

POLICY = {
    "2023_H1":"THREE_HEAD", "2023_H2":"THREE_HEAD",
    "2024_H1":"FLU_RSV_FIRST", "2024_H2":"FLU_RSV_FIRST",
    "2025_H1":"COVID_HALF", "2025_H2":"COVID_HALF", "2026_H1":"COVID_HALF",
}
EXPECTED_SELECTED_AUC = {
    "2023_H1": .6137, "2023_H2": .6354, "2024_H1": .7197,
    "2024_H2": .7150, "2025_H1": .7854, "2025_H2": .8027,
    "2026_H1": .7229,
}
flu_rsv = safe_head_array[:,:2].max(axis=1)
flu_rsv_first = np.where(np.isfinite(flu_rsv),flu_rsv,head_array[:,2])
covid_half = np.maximum(flu_rsv,
                        np.where(np.isfinite(head_array[:,2]),.5*head_array[:,2],-np.inf))
matched["selected_score"] = np.nan
policy_rows=[]
for fold in FOLDS:
    mask=matched["fold"].eq(fold)
    candidate={"THREE_HEAD":three_score,
               "FLU_RSV_FIRST":flu_rsv_first,
               "COVID_HALF":covid_half}[POLICY[fold]]
    matched.loc[mask,"selected_score"] = candidate[mask.to_numpy()]
    known=matched.loc[mask & matched.target.notna()]
    y=known.target.astype(int)
    selected_auc=float(roc_auc_score(y, known.selected_score))
    require(abs(selected_auc-EXPECTED_SELECTED_AUC[fold]) <= .00015,
            "Historical selected-policy AUC did not reproduce: " + fold)
    policy_rows.append({"fold":fold,"policy":POLICY[fold],
                        "rf59_auc":float(fold_metrics.set_index("fold").loc[fold,"rf59_auc"]),
                        "three_head_auc":float(fold_metrics.set_index("fold").loc[fold,"three_auc"]),
                        "selected_policy_auc":selected_auc})
require(np.isfinite(matched.selected_score).all(), "Missing policy scores.")
policy_results=pd.DataFrame(policy_rows)
print("CHRONOLOGICAL COMBINATION-POLICY AUDIT: PASS")
display(policy_results.round(4))
print("SELECTED is a retrospective policy sensitivity, NOT the default THREE_HEAD.")
print("No additional model fitting or post-evaluation policy search performed.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Fixed-budget operational Top-3 / Top-5 ranking
# MAGIC
# MAGIC Rank **every** eligible country in each forecast week by descending uncalibrated score; use ascending ISO3 as the deterministic tie-breaker. **Then** examine which selected outcomes became known. Unknown outcomes stay in the review workload and are not counted as negatives. For each budget, distinguish `reviews` (all selections), `known`, `unknown`, `positives`, `precision_among_known` and `observed_positive_yield_per_review` (the conservative observed positive count divided by *all* reviews).
# MAGIC
# MAGIC Compare the same archived country-week queue for RF59, THREE_HEAD and the historical SELECTED policy. This is a prioritisation experiment, not a recommendation to issue automated clinical/public-health alerts.

# COMMAND ----------

APPROACHES = {"RF59":"rf59_score", "THREE_HEAD":"three_score",
              "SELECTED":"selected_score"}
rank_parts=[]
summary_rows=[]
for name, score in APPROACHES.items():
    ranked = matched.sort_values(["fold","prediction_date",score,"iso3"],
                                 ascending=[True,True,False,True],kind="mergesort").copy()
    ranked["rank"] = ranked.groupby(["fold","prediction_date"]).cumcount()+1
    ranked["pool"] = ranked.groupby(["fold","prediction_date"])["iso3"].transform("size")
    ranked["approach"] = name
    rank_parts.append(ranked[KEYS+["approach","rank","pool","target"]])
    for k in (3,5):
        selected=ranked.loc[ranked["rank"].le(k)]
        weeks = ranked[["fold","prediction_date"]].drop_duplicates().shape[0]
        require(len(selected)==k*weeks,"An archived forecast pool is too small for Top-"+str(k))
        known=selected.loc[selected.target.notna()]
        pos=int(known.target.eq(1).sum())
        summary_rows.append({"approach":name,"budget_k":k,"forecast_weeks":weeks,
                             "reviews":len(selected),"known":len(known),
                             "unknown":int(selected.target.isna().sum()),"positives":pos,
                             "precision_among_known":pos/len(known) if len(known) else np.nan,
                             "observed_positive_yield_per_review":pos/len(selected)})
rankings=pd.concat(rank_parts,ignore_index=True)
review_summary=pd.DataFrame(summary_rows)
require(len(rankings)==3*2491,"Incomplete matching rank records.")
require((review_summary["known"]+review_summary["unknown"]==review_summary["reviews"]).all(),
        "Top-K known/unknown accounting discrepancy.")
print("CORRECTED TOP-K ACCOUNTING: PASS")
display(review_summary.round(4))
print("Unknown outcomes were NOT recoded as negatives.")
print("No rank is produced using future outcome information.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Distinct pathogen-specific onset episodes and lead times
# MAGIC
# MAGIC Reproduce the `CORRECTED-09B` event definition, **not** the earlier Notebook 03 denominator of 129 episodes. Here an episode begins when a disease-stream activity exceeds its country/fold threshold and the **previous four observed weeks** for that stream are complete and non-elevated. Episodes are **country–stream–onset** units: simultaneous influenza and RSV onsets count as separate disease-stream events. Events with incomplete four-week history are not asserted to be confirmed onsets.
# MAGIC
# MAGIC For each eligible onset, check forecast opportunities from onset through **three weeks earlier**. A Top-K selection before/on onset counts as detected; one, two, and three weeks early are reported separately. Only episodes with at least one eligible forecast opportunity contribute to the detection-rate denominator. Compute a **same-capacity random-selection expectation** for context, not an outbreak null model.
# MAGIC
# MAGIC **Interpretation:** These are internally defined surveillance elevations. Historic publication delays, changing testing practices and incomplete evidence remain material limitations. This computation reads saved source activity/definitions, makes no new forecasts and writes nothing.

# COMMAND ----------

def strict_bool(value):
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if str(value).strip().lower() in ("true","1","1.0"):
        return True
    if str(value).strip().lower() in ("false","0","0.0"):
        return False
    raise RuntimeError("Unexpected eligibility flag: " + str(value))

definitions=read_v0("definitions",["fold","iso3"] +
    [s+"_eligible" for s in ("flu","rsv","covid")]+
    [s+"_threshold" for s in ("flu","rsv","covid")],400)
raw=read_v0("activity",["iso3","week_start","flu_activity",
                         "rsv_activity","covid_activity"],12000)
require(len(definitions)==280, "Unexpected 7-fold/40-country definition count.")
require(not definitions.duplicated(["fold","iso3"]).any(),"Duplicate country/fold definitions.")
raw["week_start"]=pd.to_datetime(raw["week_start"],errors="raise").dt.normalize()
require(not raw.duplicated(["iso3","week_start"]).any(),"Duplicate weekly activity keys.")
for s in ("flu","rsv","covid"):
    raw[s+"_activity"]=pd.to_numeric(raw[s+"_activity"],errors="raise")
raw=raw.set_index(["iso3","week_start"]).sort_index()

onset_rows=[]
uncertain_onsets=0
for fold, start, end in FOLD_DATES:
    df=definitions.loc[definitions.fold.eq(fold)]
    require(len(df)==40,"Expected 40 country definitions for "+fold)
    for rec in df.itertuples(index=False):
        for s in ("flu","rsv","covid"):
            if not strict_bool(getattr(rec,s+"_eligible")):
                continue
            threshold=float(getattr(rec,s+"_threshold"))
            require(np.isfinite(threshold),"Eligible stream without finite threshold.")
            country_activity=raw.loc[rec.iso3,s+"_activity"]
            for onset in pd.date_range(start,pd.Timestamp(end)-pd.Timedelta(weeks=1),freq="W-MON"):
                v=country_activity.get(onset,np.nan)
                if not (np.isfinite(v) and v>threshold and v>0):
                    continue
                prev=country_activity.reindex([onset-pd.Timedelta(weeks=i)
                                              for i in range(1,5)]).to_numpy(float)
                if not np.isfinite(prev).all():
                    uncertain_onsets+=1
                    continue
                if (prev<=threshold).all():
                    onset_rows.append({"fold":fold,"iso3":rec.iso3,
                                       "stream":s,"onset_week":onset})
onsets=pd.DataFrame(onset_rows)
require(not onsets.empty,"No distinct episodes identified.")
onsets=onsets.sort_values(["fold","iso3","stream","onset_week"]).reset_index(drop=True)
onsets["episode_id"]=np.arange(1,len(onsets)+1)
lookup={(r.fold,r.approach,r.iso3,r.prediction_date):(int(r.rank),int(r.pool))
        for r in rankings.itertuples(index=False)}
require(len(lookup)==len(rankings),"Duplicated archived rankings.")

records=[]
random_records=[]
for e in onsets.itertuples(index=False):
    for name in APPROACHES:
        opportunities=[]
        for lead in range(4):
            week=e.onset_week-pd.Timedelta(weeks=lead)
            item=lookup.get((e.fold,name,e.iso3,week))
            if item is not None:
                opportunities.append((lead,item[0],item[1]))
        for k in (3,5):
            alerts=[lead for lead,rank,pool in opportunities if rank<=k]
            earliest=max(alerts) if alerts else np.nan
            records.append({"episode_id":e.episode_id,"fold":e.fold,"stream":e.stream,
                            "approach":name,"k":k,"opportunities":len(opportunities),
                            "detected":bool(alerts),
                            "early_1w":bool(alerts and earliest>=1),
                            "early_2w":bool(alerts and earliest>=2),
                            "early_3w":bool(alerts and earliest>=3),
                            "earliest_lead":earliest})
            if name=="RF59" and opportunities:
                for minimum_lead in range(4):
                    pools=[pool for lead,rank,pool in opportunities if lead>=minimum_lead]
                    probability=(1-np.prod([1-k/pool for pool in pools])) if pools else 0.0
                    random_records.append({"k":k,"minimum_lead":minimum_lead,
                                           "expected_episodes":probability})
episode_results=pd.DataFrame(records)
require(len(episode_results)==len(onsets)*len(APPROACHES)*2,
        "Incomplete distinct-event evaluation.")

rows=[]
for (approach,k),group in episode_results.groupby(["approach","k"]):
    eligible=group.loc[group.opportunities.gt(0)]
    rows.append({"approach":approach,"k":k,"confirmed_onsets":len(group),
                 "onsets_with_opportunity":len(eligible),
                 "detected":int(eligible.detected.sum()),
                 "detection_rate":float(eligible.detected.mean()) if len(eligible) else np.nan,
                 "early_1w":int(eligible.early_1w.sum()),
                 "early_2w":int(eligible.early_2w.sum()),
                 "early_3w":int(eligible.early_3w.sum())})
episode_summary=pd.DataFrame(rows)
random_summary=(pd.DataFrame(random_records).groupby(["k","minimum_lead"],as_index=False)
                .agg(expected_episodes=("expected_episodes","sum")))
print("CORRECTED-09B DISTINCT ONSET EPISODES: PASS")
print("Country-stream onsets:",len(onsets),"| High-activity rows with incomplete four-week history:",uncertain_onsets)
display(episode_summary.round(4))
print("MATCHED RANDOM-SELECTION EXPECTATION")
display(random_summary.round(2))
print("Do not substitute the older 129-episode V2 cohort for this denominator.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Which pathogen head drives the result?
# MAGIC
# MAGIC A maximum-of-heads score can favour a head with a different score distribution or more consistently available input data. Report the winning head across the whole queue and in the Top-3 reviewed set, plus each head's score quantiles and availability. This is a **diagnostic**, not an explanation of causal disease importance or an invitation to retune weights. A disease head can win because of score scale, not because its corresponding pathogen caused the target event.

# COMMAND ----------

head_rows=[]
for fold, group in matched.groupby("fold",sort=True):
    for s,col in zip(("flu","rsv","covid"),HEADS):
        numeric=pd.to_numeric(group[col],errors="raise").dropna().astype(float)
        head_rows.append({"fold":fold,"stream":s,"available_n":len(numeric),
                          "missing_n":len(group)-len(numeric),
                          "median_score":float(numeric.median()) if len(numeric) else np.nan,
                          "p90_score":float(numeric.quantile(.90)) if len(numeric) else np.nan,
                          "winner_pct":100*float(group.winning_head.eq(s).mean())})
head_scale=pd.DataFrame(head_rows)
ranked_three=rankings.loc[(rankings.approach=="THREE_HEAD") & (rankings["rank"].le(3))]
three_top3=ranked_three.merge(matched[KEYS+["winning_head"]],on=KEYS,validate="one_to_one")
composition=(three_top3.groupby(["fold","winning_head"]).size()
             .unstack(fill_value=0).reindex(columns=["flu","rsv","covid"],fill_value=0))
require(composition.sum(axis=1).ge(3).all(),"Unexpected Top-3 head winner composition.")
print("THREE_HEAD SCORE-SCALE / COMPOSITION DIAGNOSTIC: PASS")
display(head_scale.round(4))
print("TOP-3 winning-head counts by historical period")
display(composition.reset_index())
print("Winning-head distributions do not establish score calibration or causal contribution.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Paired four-week block bootstrap: uncertainty around the matched AUC/AP difference
# MAGIC
# MAGIC Use the original `CORRECTED-10A` settings: **2,000** draws, **four-week moving blocks**, and seed **20261030**. The resampling unit is a whole calendar week (all countries in that week move together), with blocks sampled independently in each historical period. This preserves some short-term time structure but does not fully capture between-country or long-term dependence.
# MAGIC
# MAGIC The bootstrap **reuses already saved scores**, never trains a classifier. Its output provides an uncertainty interval for the seven-period *macro-average paired* THREE_HEAD minus RF59 AUC/AP difference. The previously recorded AUC difference was approximately **+0.0270**, with a recorded 95% interval of about **+0.0004 to +0.0536**. Verify against current calculation before reporting externally. This is a post-selection historical comparison; do not claim genuinely prospective validation or a promotion decision from an interval alone.

# COMMAND ----------

B=2000
BLOCK_WEEKS=4
rng=np.random.default_rng(20261030)
weeks_by_fold={}
for fold,part in matched.groupby("fold",sort=True):
    # Array-based replication of the original CORRECTED-10A week-block sampling.
    part=part.reset_index(drop=True)
    index_by_week=[part.index[part.prediction_date.eq(w)].to_numpy()
                   for w in sorted(part.prediction_date.unique())]
    weeks_by_fold[fold]=(part,index_by_week)

bootstrap_rows=[]
for iteration in range(B):
    auc_deltas=[]
    ap_deltas=[]
    good=True
    for fold,(part,week_idx) in weeks_by_fold.items():
        n=len(week_idx)
        chosen=[]
        while len(chosen)<n:
            start=int(rng.integers(0,n))
            chosen.extend(((start+np.arange(BLOCK_WEEKS))%n).tolist())
        row_idx=np.concatenate([week_idx[i] for i in chosen[:n]])
        sample=part.iloc[row_idx]
        known=sample.loc[sample.target.notna()]
        y=known.target.astype(int)
        if y.nunique()<2:
            good=False
            break
        auc_deltas.append(roc_auc_score(y,known.three_score)
                          -roc_auc_score(y,known.rf59_score))
        ap_deltas.append(average_precision_score(y,known.three_score)
                         -average_precision_score(y,known.rf59_score))
    if good:
        bootstrap_rows.append({"delta_auc":float(np.mean(auc_deltas)),
                               "delta_ap":float(np.mean(ap_deltas))})
boot=pd.DataFrame(bootstrap_rows)
require(len(boot)>1800, "Too many invalid four-week bootstrap draws.")
uncertainty=pd.DataFrame([
    {"metric":"macro_mean_auc", "observed_delta":float(fold_metrics.delta_auc.mean()),
     "ci95_lower":float(boot.delta_auc.quantile(.025)),
     "ci95_upper":float(boot.delta_auc.quantile(.975)),
     "bootstrap_fraction_positive":float(boot.delta_auc.gt(0).mean()),"draws":len(boot)},
    {"metric":"macro_mean_ap", "observed_delta":float(fold_metrics.delta_ap.mean()),
     "ci95_lower":float(boot.delta_ap.quantile(.025)),
     "ci95_upper":float(boot.delta_ap.quantile(.975)),
     "bootstrap_fraction_positive":float(boot.delta_ap.gt(0).mean()),"draws":len(boot)},
])
print("CORRECTED-10A PAIRED FOUR-WEEK BOOTSTRAP: PASS")
display(uncertainty.round(4))
print("No estimator fits; 2,000 in-memory resamples of existing country-week scores.")
print("Intervals are historical/post-selection uncertainty, not promotion or prospective proof.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Reproduction summary, governance and presentation handoff
# MAGIC
# MAGIC A complete Databricks execution should verify **all** previous steps before this summary is considered complete. Present results with precise qualifiers:
# MAGIC
# MAGIC - Historical seven-period macro ROC-AUC, AP, within-week discrimination and Top-3/Top-5 unknown-outcome denominators come from the matched `C4` saved predictions, **not** from the earlier V2 1,612-queue / 129-episode cohort.
# MAGIC - Distinct pathogen-specific onset counts follow `CORRECTED-09B`, rather than assuming earlier episodes are the same estimand.
# MAGIC - `2026_H1` is already **consumed retrospective development evidence**; the work does not establish a new untouched holdout or publication-vintage-correct forecast deployment.
# MAGIC - The formal `GATE4_CLOSED_NO_PROMOTION` record in Notebook 04 and the registered RF59-D2 **shadow** reference remain unchanged. A later THREE_HEAD research advantage is **not** documented production promotion.
# MAGIC - If a check fails, **stop and diagnose**; do not overwrite an archived table, rerun model-fitting cells, alter historical labels or select a different model on the same outcomes merely to make a notebook pass.
# MAGIC
# MAGIC Next deliverables for Muj: GitHub-ready README, reproducibility guide, model and data cards, references/limitations register, evidence visualisations, and an 8–10-slide final presentation. Retain original notebooks as immutable historical archives.

# COMMAND ----------

require(len(matched)==2491 and len(fold_metrics)==7,"Final comparison inputs incomplete.")
require(len(review_summary)==6 and len(episode_summary)==6,"Operational summaries incomplete.")
require(len(uncertainty)==2,"Bootstrap uncertainty summary incomplete.")
require(gate4.iloc[0]["status"]=="GATE4_CLOSED_NO_PROMOTION",
        "Historical formal promotion conclusion was modified.")

handoff=pd.DataFrame([{
    "comparison":"CORRECTED-10A_MATCHED_SAVED_SCORES",
    "historical_periods":len(FOLDS),
    "saved_country_weeks":len(matched),
    "seven_period_rf59_mean_auc":float(fold_metrics.rf59_auc.mean()),
    "seven_period_three_head_mean_auc":float(fold_metrics.three_auc.mean()),
    "paired_auc_difference":float(fold_metrics.delta_auc.mean()),
    "paired_auc_bootstrap_ci_lower":float(uncertainty.loc[0,"ci95_lower"]),
    "paired_auc_bootstrap_ci_upper":float(uncertainty.loc[0,"ci95_upper"]),
    "distinct_confirmed_country_stream_onsets":len(onsets),
    "unknown_outcomes_kept_unknown":True,
    "prior_gate4_promotion":"NONE",
    "registered_reference":"RF59_D2_MATCHED_REFERENCE",
    "three_head_registered_or_promoted_here":False,
    "2026_h1_is_unconsumed_holdout":False,
    "training_performed":False,
    "delta_tables_written":False,
}])
print("NOTEBOOK 05 — CORRECTED TEMPORAL REPRODUCTION SUMMARY: PASS")
display(handoff.round(4))
print("All saved predictions, onsets and statistical uncertainty were analysed read-only.")
print("Historical Gate-4 NO PROMOTION preserved; RF59 registered shadow reference unchanged.")
print("THREE_HEAD improvement is a later research finding, not an approved promotion.")
print("No model fitting, model registration, Delta writes or source refresh occurred.")