import os
import logging
import numpy as np
import pandas as pd
import joblib
from scipy.stats import percentileofscore, ttest_ind
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# =====================================================================
# 1. LOAD AND VALIDATE
# =====================================================================

def load_and_validate(input_path: str):
    logger.info(f"Loading Stage 4 data from {input_path}")
    df = pd.read_parquet(input_path)

    npi_candidates = ["PRF_PHYSN_NPI_1", "RNDR_NPI", "PRF_NPI", "ORG_NPI_NUM", "PROVIDER_NPI", "Provider_NPI", "NPI", "PRVDR_NUM", "PROVIDER_ID"]
    hcpcs_candidates = ["ORIGINAL_HCPCS_CD", "HCPCS_CD_1", "HCPCS_CD", "HCPCS", "CPT_CODE", "PROC_CODE"]
    working_pmt_candidates = ["WORKING_PMT_AMT", "LINE_NCH_PMT_AMT_1", "CLM_PMT_AMT", "PAYMENT_AMOUNT"]
    expected_pmt_candidates = ["EXPECTED_PMT_FINAL", "EXPECTED_PAYMENT", "EXPECTED_PMT", "PREDICTED_PAYMENT"]

    npi_col = next((c for c in npi_candidates if c in df.columns), None)
    hcpcs_col = next((c for c in hcpcs_candidates if c in df.columns), None)
    working_pmt_col = next((c for c in working_pmt_candidates if c in df.columns), None)
    expected_pmt_col = next((c for c in expected_pmt_candidates if c in df.columns), None)

    if not npi_col:
        raise ValueError(f"Could not find an NPI/Provider ID column in dataframe. Checked: {npi_candidates}")
    if not hcpcs_col:
        raise ValueError(f"Could not find a Procedure/HCPCS column in dataframe. Checked: {hcpcs_candidates}")
    if not working_pmt_col:
        raise ValueError(f"Could not find a Working Payment column in dataframe. Checked: {working_pmt_candidates}")
    if not expected_pmt_col:
        raise ValueError(f"Could not find an Expected Payment column in dataframe. Checked: {expected_pmt_candidates}")

    col_mapping = {
        "NPI": npi_col,
        "HCPCS": hcpcs_col,
        "WORKING_PMT": working_pmt_col,
        "EXPECTED_PMT": expected_pmt_col,
    }
    logger.info(f"Successfully mapped core columns: {col_mapping}")

    # --- ADDED: Flag Missing or Invalid HCPCS Rows ---
    df["STAGE5_MISSING_HCPCS_FLAG"] = (
        df[hcpcs_col].isna() |
        df[hcpcs_col].astype(str).str.strip().isin(["", "None", "nan", "NaN"])
    ).astype(int)

    if "kmeans_cluster" not in df.columns:
        raise ValueError("kmeans_cluster column is missing from input dataset.")

    df["STAGE5_CLUSTER_KEY"] = df["kmeans_cluster"].astype(str).str.strip()

    if "MODEL_PARTITION" not in df.columns:
        df["MODEL_PARTITION"] = "holdout"

    if "SOURCE_CLAIM_LINE_ID" in df.columns and "CLAIM_LINE_ID" in df.columns:
        partition_map = df.dropna(subset=['CLAIM_LINE_ID']).set_index('CLAIM_LINE_ID')['MODEL_PARTITION'].to_dict()
        missing_part = df["MODEL_PARTITION"].isna()
        df.loc[missing_part, "MODEL_PARTITION"] = df.loc[missing_part, "SOURCE_CLAIM_LINE_ID"].map(partition_map)
    df["MODEL_PARTITION"] = df["MODEL_PARTITION"].fillna("holdout")

    return df, col_mapping

# =====================================================================
# 2 & 3. PARTITIONS AND FEATURE ENGINEERING
# =====================================================================
def engineer_contextual_features(df: pd.DataFrame, col_map: dict) -> pd.DataFrame:
    logger.info("Defining partitions and engineering train-only contextual features...")

    npi_col = col_map["NPI"]
    hcpcs_col = col_map["HCPCS"]
    pmt_col = col_map["WORKING_PMT"]

    clean_original_mask = (df["SYNTHETIC_RECORD_CREATED"] == 0) & (df["IS_ANOMALY_INJECTED"] == 0)
    clean_train_mask = clean_original_mask & (df["MODEL_PARTITION"] == "train")

    train_df = df[clean_train_mask]

    # Provider-level aggregations (Train only)
    prov_stats = train_df.groupby(npi_col).agg(
        STAGE5_PROVIDER_CLAIM_VOLUME=(pmt_col, 'count'),
        STAGE5_PROVIDER_HCPCS_DIVERSITY=(hcpcs_col, 'nunique'),
        STAGE5_PROVIDER_MEDIAN_PAYMENT=(pmt_col, 'median'),
        STAGE5_PROVIDER_MEAN_PAYMENT=(pmt_col, 'mean')
    ).reset_index()

    # Provider-HCPCS level aggregations (Train only)
    prov_hcpcs_stats = train_df.groupby([npi_col, hcpcs_col]).agg(
        STAGE5_PROVIDER_HCPCS_COUNT=(pmt_col, 'count')
    ).reset_index()

    # Map back to full df
    df = df.merge(prov_stats, on=npi_col, how='left')
    df = df.merge(prov_hcpcs_stats, on=[npi_col, hcpcs_col], how='left')

    # Unseen provider logic
    df["STAGE5_UNSEEN_PROVIDER_FLAG"] = df["STAGE5_PROVIDER_CLAIM_VOLUME"].isna().astype(int)

    # Impute unseen
    df["STAGE5_PROVIDER_CLAIM_VOLUME"] = df["STAGE5_PROVIDER_CLAIM_VOLUME"].fillna(0)
    df["STAGE5_PROVIDER_HCPCS_COUNT"] = df["STAGE5_PROVIDER_HCPCS_COUNT"].fillna(0)

    # Concentration
    df["STAGE5_PROVIDER_HCPCS_CONCENTRATION"] = np.where(
        df["STAGE5_PROVIDER_CLAIM_VOLUME"] > 0,
        df["STAGE5_PROVIDER_HCPCS_COUNT"] / df["STAGE5_PROVIDER_CLAIM_VOLUME"],
        0.0
    )

    global_median_pmt = train_df[pmt_col].median()
    global_mean_pmt = train_df[pmt_col].mean()

    df["STAGE5_PROVIDER_MEDIAN_PAYMENT"] = df["STAGE5_PROVIDER_MEDIAN_PAYMENT"].fillna(global_median_pmt)
    df["STAGE5_PROVIDER_MEAN_PAYMENT"] = df["STAGE5_PROVIDER_MEAN_PAYMENT"].fillna(global_mean_pmt)
    df["STAGE5_PROVIDER_HCPCS_DIVERSITY"] = df["STAGE5_PROVIDER_HCPCS_DIVERSITY"].fillna(1)

    # Log Transforms
    df["LOG_WORKING_PAYMENT"] = np.log1p(df[pmt_col].clip(lower=0))
    df["LOG_EXPECTED_PAYMENT"] = np.log1p(df[col_map["EXPECTED_PMT"]].clip(lower=0))
    df["LOG_PROVIDER_CLAIM_VOLUME"] = np.log1p(df["STAGE5_PROVIDER_CLAIM_VOLUME"].clip(lower=0))
    df["LOG_PROVIDER_HCPCS_COUNT"] = np.log1p(df["STAGE5_PROVIDER_HCPCS_COUNT"].clip(lower=0))

    if "PEER_GROUP_COUNT" in df.columns:
        df["LOG_PEER_GROUP_COUNT"] = np.log1p(df["PEER_GROUP_COUNT"].clip(lower=0))
    else:
        df["LOG_PEER_GROUP_COUNT"] = 0.0

    return df

# =====================================================================
# 4, 5, 6, 7. FEATURE SELECTION, TRAINING, AND SCORING
# =====================================================================

def build_and_score_models(df: pd.DataFrame, model_out_dir: str):
    logger.info("Building cluster-stratified Isolation Forest models...")

    candidate_features = [
        "RELATIVE_SURGE_RATIO_FINAL", "STAGE4_POSITIVE_PAYMENT_RESIDUAL",
        "STAGE4_RELATIVE_SURGE_PERCENTILE", "STAGE4_POSITIVE_RESIDUAL_PERCENTILE",
        "LOG_WORKING_PAYMENT", "LOG_EXPECTED_PAYMENT", "LOG_PEER_GROUP_COUNT",
        "LOG_PROVIDER_CLAIM_VOLUME", "LOG_PROVIDER_HCPCS_COUNT",
        "STAGE5_PROVIDER_HCPCS_CONCENTRATION", "CHRONIC_CONDITION_COUNT",
        "PATIENT_AGE", "AGE_MISSING_FLAG"
    ]

    features = [f for f in candidate_features if f in df.columns]
    logger.info(f"Selected candidate features ({len(features)}): {features}")

    df["STAGE5_MODEL_SOURCE"] = "unassigned"
    df["STAGE5_IF_RAW_ANOMALY_SCORE"] = np.nan
    df["STAGE5_IF_RISK_SCORE"] = np.nan
    df["STAGE5_SCORE_THRESHOLD_SOURCE"] = "unassigned"

    # --- ADDED: Define Eligibility Mask & Update Clean Training Masks ---
    stage5_score_eligible_mask = (df["STAGE5_MISSING_HCPCS_FLAG"] == 0)
    missing_hcpcs_mask = (df["STAGE5_MISSING_HCPCS_FLAG"] == 1)

    clean_original_mask = (
        (df["SYNTHETIC_RECORD_CREATED"] == 0) &
        (df["IS_ANOMALY_INJECTED"] == 0) &
        stage5_score_eligible_mask
    )
    clean_train_mask = clean_original_mask & (df["MODEL_PARTITION"] == "train")
    clean_holdout_mask = clean_original_mask & (df["MODEL_PARTITION"] == "holdout")

    clusters = df["STAGE5_CLUSTER_KEY"].unique()
    os.makedirs(model_out_dir, exist_ok=True)

    # Global Fallback Training
    global_train_df = df[clean_train_mask]
    if len(global_train_df) == 0:
        logger.warning("clean_train_mask returned 0 records! Falling back to clean_original_mask for global fit.")
        global_train_df = df[clean_original_mask]

    global_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy="median")),
        ('scaler', RobustScaler()),
        ('iforest', IsolationForest(n_estimators=300, max_samples="auto", contamination="auto", random_state=42, n_jobs=-1))
    ])

    global_pipeline.fit(global_train_df[features])

    holdout_subset = df[clean_holdout_mask]
    if len(holdout_subset) == 0:
        holdout_subset = global_train_df

    global_holdout_scores = -global_pipeline.score_samples(holdout_subset[features])

    for cluster in clusters:
        # Score ONLY eligible rows within this cluster
        cluster_mask = (df["STAGE5_CLUSTER_KEY"] == cluster) & stage5_score_eligible_mask
        c_train_mask = clean_train_mask & (df["STAGE5_CLUSTER_KEY"] == cluster)
        c_train_df = df[c_train_mask]

        if len(c_train_df) > 0:
            c_features = [f for f in features if c_train_df[f].nunique() > 1]
        else:
            c_features = []

        if len(c_train_df) >= 200 and len(c_features) > 0:
            logger.info(f"Training cluster '{cluster}' model on {len(c_train_df)} clean records. Features: {len(c_features)}")
            pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy="median")),
                ('scaler', RobustScaler()),
                ('iforest', IsolationForest(n_estimators=300, max_samples="auto", contamination="auto", random_state=42, n_jobs=-1))
            ])
            pipeline.fit(c_train_df[c_features])
            joblib.dump(pipeline, os.path.join(model_out_dir, f"stage5_isolation_forest_cluster_{cluster}.joblib"))

            raw_scores = -pipeline.score_samples(df.loc[cluster_mask, c_features])
            df.loc[cluster_mask, "STAGE5_IF_RAW_ANOMALY_SCORE"] = raw_scores
            df.loc[cluster_mask, "STAGE5_MODEL_SOURCE"] = "cluster_specific"

            c_holdout_mask = clean_holdout_mask & (df["STAGE5_CLUSTER_KEY"] == cluster)
            if c_holdout_mask.sum() > 50:
                calib_scores = -pipeline.score_samples(df.loc[c_holdout_mask, c_features])
                df.loc[cluster_mask, "STAGE5_IF_RISK_SCORE"] = df.loc[cluster_mask, "STAGE5_IF_RAW_ANOMALY_SCORE"].apply(
                    lambda x: percentileofscore(calib_scores, x) / 100.0
                )
                df.loc[cluster_mask, "STAGE5_SCORE_THRESHOLD_SOURCE"] = f"cluster_{cluster}_holdout"
            else:
                df.loc[cluster_mask, "STAGE5_IF_RISK_SCORE"] = df.loc[cluster_mask, "STAGE5_IF_RAW_ANOMALY_SCORE"].apply(
                    lambda x: percentileofscore(global_holdout_scores, x) / 100.0
                )
                df.loc[cluster_mask, "STAGE5_SCORE_THRESHOLD_SOURCE"] = "global_holdout_fallback"
        else:
            logger.info(f"Cluster '{cluster}' insufficient for local model ({len(c_train_df)} rows). Using global fallback.")
            df.loc[cluster_mask, "STAGE5_IF_RAW_ANOMALY_SCORE"] = -global_pipeline.score_samples(df.loc[cluster_mask, features])
            df.loc[cluster_mask, "STAGE5_MODEL_SOURCE"] = "global_clean_fallback"
            df.loc[cluster_mask, "STAGE5_IF_RISK_SCORE"] = df.loc[cluster_mask, "STAGE5_IF_RAW_ANOMALY_SCORE"].apply(
                lambda x: percentileofscore(global_holdout_scores, x) / 100.0
            )
            df.loc[cluster_mask, "STAGE5_SCORE_THRESHOLD_SOURCE"] = "global_holdout_fallback"

    # Assign Tiers & Flags for Eligible Scored Rows
    df["STAGE5_IF_TIER"] = "standard"
    df.loc[stage5_score_eligible_mask & (df["STAGE5_IF_RISK_SCORE"] >= 0.95), "STAGE5_IF_TIER"] = "elevated"
    df.loc[stage5_score_eligible_mask & (df["STAGE5_IF_RISK_SCORE"] >= 0.99), "STAGE5_IF_TIER"] = "extreme"

    df["STAGE5_IF_FLAG"] = df["STAGE5_IF_TIER"].isin(["elevated", "extreme"]).astype(int)

    # --- ADDED: Explicit Handling for Missing-HCPCS Rows ---
    if missing_hcpcs_mask.any():
        logger.info(f"Flagged {missing_hcpcs_mask.sum()} records with missing HCPCS codes for data quality review.")
        df.loc[missing_hcpcs_mask, "STAGE5_IF_RAW_ANOMALY_SCORE"] = np.nan
        df.loc[missing_hcpcs_mask, "STAGE5_IF_RISK_SCORE"] = np.nan
        df.loc[missing_hcpcs_mask, "STAGE5_IF_TIER"] = "data_quality_review"
        df.loc[missing_hcpcs_mask, "STAGE5_IF_FLAG"] = 0
        df.loc[missing_hcpcs_mask, "STAGE5_MODEL_SOURCE"] = "not_scored_missing_hcpcs"
        df.loc[missing_hcpcs_mask, "STAGE5_SCORE_THRESHOLD_SOURCE"] = "not_applicable"

    return df, features

# =====================================================================
# 8 & 9. EVALUATION AND WELCH INFERENCE CHECK
# =====================================================================
def evaluate_and_test(df: pd.DataFrame):
    logger.info("Executing Stage 5 Evaluation and Welch's t-Test...")

    df["IS_STAGE5_EVALUATION_SCENARIO"] = df["SCENARIO_TYPE"].isin([
        "extreme_payment_deviation", "moderate_payment_deviation", "duplicate_like_billing"
    ]).astype(int)

    clean_holdout = df[(df["SYNTHETIC_RECORD_CREATED"] == 0) &
                       (df["IS_ANOMALY_INJECTED"] == 0) &
                       (df["MODEL_PARTITION"] == "holdout")]

    logger.info("=== Queue Metrics (Holdout + Injected) ===")
    eval_pool = pd.concat([clean_holdout, df[df["IS_STAGE5_EVALUATION_SCENARIO"] == 1]])

    def calc_queue(mask_name, mask):
        sub_df = eval_pool[eval_pool["IS_ANOMALY_INJECTED"] == 0 | mask]
        sub_df = sub_df.sort_values(by="STAGE5_IF_RISK_SCORE", ascending=False)
        positives = mask.sum()
        for pct in [0.05, 0.10]:
            k = int(len(sub_df) * pct)
            top_k = sub_df.head(k)
            captured = top_k["IS_STAGE5_EVALUATION_SCENARIO"].sum()
            logger.info(f"{mask_name} @ {pct*100:.0f}%: Precision={captured/k if k else 0:.2%} | Recall={captured/positives if positives else 0:.2%}")

    ext_mask = eval_pool["SCENARIO_TYPE"] == "extreme_payment_deviation"
    mod_mask = eval_pool["SCENARIO_TYPE"] == "moderate_payment_deviation"
    dup_mask = eval_pool["SCENARIO_TYPE"] == "duplicate_like_billing"
    all_mask = eval_pool["IS_STAGE5_EVALUATION_SCENARIO"] == 1

    calc_queue("All Scenarios", all_mask)
    calc_queue("Extreme", ext_mask)
    calc_queue("Moderate", mod_mask)
    calc_queue("Duplicates", dup_mask)

    # Welch's T-Test
    logger.info("=== Optional Welch's t-Test Inference Check ===")
    logger.info("DISCLAIMER: Welch’s t-test is an exploratory comparison of score distributions "
                "in this synthetic POC. It does not establish clinical validity, payment error, "
                "fraud, causality, or production performance.")

    clean_scores = clean_holdout["STAGE5_IF_RISK_SCORE"].dropna()

    scenarios = {
        "Extreme Payment": df[df["SCENARIO_TYPE"] == "extreme_payment_deviation"],
        "Moderate Payment": df[df["SCENARIO_TYPE"] == "moderate_payment_deviation"],
        "Duplicate Billing": df[df["SCENARIO_TYPE"] == "duplicate_like_billing"],
        "All Injected": df[df["IS_STAGE5_EVALUATION_SCENARIO"] == 1]
    }

    for name, sub_df in scenarios.items():
        scen_scores = sub_df["STAGE5_IF_RISK_SCORE"].dropna()
        if len(scen_scores) == 0:
            continue

        t_stat, p_val = ttest_ind(clean_scores, scen_scores, equal_var=False, nan_policy="omit")

        # Cohen's d
        mean_c, mean_s = clean_scores.mean(), scen_scores.mean()
        var_c, var_s = clean_scores.var(), scen_scores.var()
        pooled_std = np.sqrt((var_c + var_s) / 2)
        cohens_d = (mean_s - mean_c) / pooled_std if pooled_std > 0 else 0

        logger.info(f"Comparison: Clean Holdout vs {name}")
        logger.info(f"  Clean N: {len(clean_scores)} | Scenario N: {len(scen_scores)}")
        logger.info(f"  Clean Mean: {mean_c:.4f} | Scenario Mean: {mean_s:.4f} | Diff: {mean_s - mean_c:.4f}")
        logger.info(f"  t-stat: {t_stat:.4f} | p-value: {p_val:.4e} | Cohen's d: {cohens_d:.4f}\n")

# =====================================================================
# 10 & 11. ASSERTIONS AND OUTPUT
# =====================================================================
def run_assertions(df: pd.DataFrame, features: list):
    logger.info("Running quality assertions...")

    # Model leak checks
    leak_cols = ["IS_ANOMALY_INJECTED", "SCENARIO_TYPE", "STAGE4_PRICE_FLAG", "NPI", "CLAIM_ID"]
    for col in leak_cols:
        assert not any(col in f for f in features), f"Assertion Failed: {col} leaked into features."

    assert df["STAGE5_IF_RISK_SCORE"].between(0.0, 1.0).all() or df["STAGE5_IF_RISK_SCORE"].isna().all() == False, "Risk scores bounds invalid"
    assert not df["STAGE5_MODEL_SOURCE"].isnull().any(), "Missing model source assignments"

    # Threshold exceedance approx (clean holdout)
    c_holdout = df[(df["SYNTHETIC_RECORD_CREATED"] == 0) & (df["IS_ANOMALY_INJECTED"] == 0) & (df["MODEL_PARTITION"] == "holdout")]
    if len(c_holdout) > 0:
        p95_rate = (c_holdout["STAGE5_IF_RISK_SCORE"] >= 0.95).mean()
        p99_rate = (c_holdout["STAGE5_IF_RISK_SCORE"] >= 0.99).mean()
        logger.info(f"Clean Holdout Exceedance -> P95 Rate: {p95_rate:.2%}, P99 Rate: {p99_rate:.2%}")

    assert df["STAGE5_CLUSTER_KEY"].dtype == 'object', "Cluster key must be string"
    assert not np.isinf(df[features].values).any(), "Infinite values found in features"

def main():
    input_path = "data/processed/cms_claims_stage4_price_scored.parquet"
    out_parquet = "data/processed/cms_claims_stage5_iforest_scored.parquet"
    out_csv = "data/processed/cms_claims_stage5_iforest_scored.csv"
    model_dir = "models/"

    # Execution Flow
    df, col_map = load_and_validate(input_path)
    df = engineer_contextual_features(df, col_map)
    df, features = build_and_score_models(df, model_dir)

    run_assertions(df, features)
    evaluate_and_test(df)

    # Filter final columns to save
    output_cols = [c for c in df.columns if c not in ['STAGE4_POSITIVE_PAYMENT_RESIDUAL']] # Keep all core + requested

    logger.info(f"Saving Stage 5 outputs to {out_parquet}...")
    os.makedirs(os.path.dirname(out_parquet), exist_ok=True)
    df.to_parquet(out_parquet, index=False)
    df.to_csv(out_csv, index=False)
    logger.info("Stage 5 Execution Complete.")

if __name__ == "__main__":
    main()
    
    
    
import os
import pandas as pd
import numpy as np

# -------------------------------------------------------------------------
# 1. Load Stage 5 Output Dataset
# -------------------------------------------------------------------------
parquet_path = "data/processed/cms_claims_stage5_iforest_scored.parquet"
csv_path = "data/processed/cms_claims_stage5_iforest_scored.csv"

if os.path.exists(parquet_path):
    print(f"Loading from: {parquet_path}")
    df = pd.read_parquet(parquet_path)
elif os.path.exists(csv_path):
    print(f"Loading from: {csv_path}")
    df = pd.read_csv(csv_path)
else:
    raise FileNotFoundError("Stage 5 output file not found. Ensure Stage 5 script has completed.")

print(f"Dataset successfully loaded. Total rows: {len(df):,}\n")

# -------------------------------------------------------------------------
# 2. Stage 5 Tier & Flag Summary
# -------------------------------------------------------------------------
print("=" * 75)
print("1. STAGE 5 ANOMALY RISK TIERS & FLAGS")
print("=" * 75)
tier_summary = df.groupby(["STAGE5_IF_TIER", "STAGE5_IF_FLAG"]).size().reset_index(name="Claim_Count")
tier_summary["Percentage"] = (tier_summary["Claim_Count"] / len(df) * 100).map("{:.2f}%".format)
print(tier_summary.to_string(index=False))

# -------------------------------------------------------------------------
# 3. Performance Across Injected Scenarios
# -------------------------------------------------------------------------
print("\n" + "=" * 75)
print("2. RISK SCORES BY SCENARIO TYPE")
print("=" * 75)
if "SCENARIO_TYPE" in df.columns:
    scen_summary = df.groupby("SCENARIO_TYPE").agg(
        Total_Claims=("STAGE5_IF_RISK_SCORE", "count"),
        Mean_Risk_Score=("STAGE5_IF_RISK_SCORE", "mean"),
        Median_Risk_Score=("STAGE5_IF_RISK_SCORE", "median"),
        Flagged_Count=("STAGE5_IF_FLAG", "sum"),
        Pct_Flagged=("STAGE5_IF_FLAG", "mean")
    ).reset_index()

    scen_summary["Pct_Flagged"] = (scen_summary["Pct_Flagged"] * 100).map("{:.2f}%".format)
    scen_summary["Mean_Risk_Score"] = scen_summary["Mean_Risk_Score"].map("{:.4f}".format)
    scen_summary["Median_Risk_Score"] = scen_summary["Median_Risk_Score"].map("{:.4f}".format)
    print(scen_summary.to_string(index=False))

# -------------------------------------------------------------------------
# 4. Cluster-Level Diagnostics
# -------------------------------------------------------------------------
print("\n" + "=" * 75)
print("3. CLUSTER-LEVEL BREAKDOWN")
print("=" * 75)
if "STAGE5_CLUSTER_KEY" in df.columns:
    cluster_summary = df.groupby("STAGE5_CLUSTER_KEY").agg(
        Total_Claims=("STAGE5_IF_RISK_SCORE", "count"),
        Model_Source=("STAGE5_MODEL_SOURCE", lambda x: x.iloc[0] if len(x) > 0 else "N/A"),
        Median_Risk_Score=("STAGE5_IF_RISK_SCORE", "median"),
        Elevated_Count=("STAGE5_IF_TIER", lambda x: (x == "elevated").sum()),
        Extreme_Count=("STAGE5_IF_TIER", lambda x: (x == "extreme").sum())
    ).reset_index()

    cluster_summary["Median_Risk_Score"] = cluster_summary["Median_Risk_Score"].map("{:.4f}".format)
    print(cluster_summary.to_string(index=False))

# -------------------------------------------------------------------------
# 5. Top 5 High-Risk Queue Preview
# -------------------------------------------------------------------------
print("\n" + "=" * 75)
print("4. TOP 5 HIGHEST RISK CLAIMS (QUEUE PREVIEW)")
print("=" * 75)
display_cols = [
    col for col in [
        "CLAIM_LINE_ID", "DESYNPUF_ID", "PRF_PHYSN_NPI_1", "ORIGINAL_HCPCS_CD", "HCPCS_CD_1",
        "SCENARIO_TYPE", "STAGE5_CLUSTER_KEY", "STAGE5_IF_RAW_ANOMALY_SCORE",
        "STAGE5_IF_RISK_SCORE", "STAGE5_IF_TIER"
    ] if col in df.columns
]

top_10 = df.sort_values(by="STAGE5_IF_RISK_SCORE", ascending=False).head(5)
print(top_10[display_cols].to_string(index=False))
print("=" * 75 + "\n")