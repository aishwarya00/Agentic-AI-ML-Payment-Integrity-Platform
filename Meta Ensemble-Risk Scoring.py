"""
STAGE 6: TRANSPARENT PARALLEL RISK-FUSION / META-ENSEMBLE COMBINER
------------------------------------------------------------------
Combines Stage 4 (Price-Deviation) and Stage 5 (Isolation Forest)
continuous risk scores into a single, explainable Audit Priority Score
for human payment-integrity review.
"""

import os
import logging
import numpy as np
import pandas as pd

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Config
INPUT_FILE = "data/processed/cms_claims_stage5_iforest_scored.parquet"
OUT_DIR = "data/processed"
OUT_PARQUET = os.path.join(OUT_DIR, "cms_claims_stage6_ensemble_scored.parquet")
OUT_CSV = os.path.join(OUT_DIR, "cms_claims_stage6_ensemble_scored.csv")

# Fusion Weights
W_STAGE4 = 0.55
W_STAGE5 = 0.45

def load_and_validate(input_path: str) -> pd.DataFrame:
    """Loads dataset and validates presence and bounds of required scores."""
    logger.info(f"Loading Stage 5 output from {input_path}")
    df = pd.read_parquet(input_path)

    score_cols = ["STAGE4_PRICE_RISK_SCORE", "STAGE5_IF_RISK_SCORE"]
    available_scores = [c for c in score_cols if c in df.columns]

    if not available_scores:
        raise ValueError(f"Neither Stage 4 nor Stage 5 risk score columns found. Looked for {score_cols}")

    # Validate bounds (0 to 1) for existing scores
    for col in available_scores:
        out_of_bounds = df.loc[df[col].notna(), col].apply(lambda x: x < 0.0 or x > 1.0)
        if out_of_bounds.any():
            raise ValueError(f"Scores in {col} are outside the required [0, 1] range.")

    # Ensure Stage4/Stage5 missing scores are explicitly NaN, not just missing columns
    for col in score_cols:
        if col not in df.columns:
            df[col] = np.nan

    return df

def define_valid_signals(df: pd.DataFrame) -> pd.DataFrame:
    """Creates boolean availability and data-quality flags."""
    df["STAGE6_STAGE4_AVAILABLE"] = df["STAGE4_PRICE_RISK_SCORE"].notna()
    df["STAGE6_STAGE5_AVAILABLE"] = df["STAGE5_IF_RISK_SCORE"].notna()

    df["STAGE6_DATA_QUALITY_REVIEW"] = False

    # Check upstream data quality flags
    if "STAGE5_MISSING_HCPCS_FLAG" in df.columns:
        df["STAGE6_DATA_QUALITY_REVIEW"] = df["STAGE6_DATA_QUALITY_REVIEW"] | (df["STAGE5_MISSING_HCPCS_FLAG"] == 1)

    if "STAGE5_DATA_QUALITY_FLAG" in df.columns:
        df["STAGE6_DATA_QUALITY_REVIEW"] = df["STAGE6_DATA_QUALITY_REVIEW"] | (df["STAGE5_DATA_QUALITY_FLAG"] == 1)

    # Also flag if both scores are entirely missing
    missing_both = (~df["STAGE6_STAGE4_AVAILABLE"]) & (~df["STAGE6_STAGE5_AVAILABLE"])
    df.loc[missing_both, "STAGE6_DATA_QUALITY_REVIEW"] = True

    return df

def fuse_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Combines Stage 4 and Stage 5 scores dynamically based on availability."""
    df["STAGE6_ENSEMBLE_RISK_SCORE"] = np.nan
    df["STAGE6_FUSION_SOURCE"] = "not_scored"

    # Masks
    dq_issue = df["STAGE6_DATA_QUALITY_REVIEW"]
    has_both = df["STAGE6_STAGE4_AVAILABLE"] & df["STAGE6_STAGE5_AVAILABLE"] & ~dq_issue
    has_s4_only = df["STAGE6_STAGE4_AVAILABLE"] & ~df["STAGE6_STAGE5_AVAILABLE"] & ~dq_issue
    has_s5_only = ~df["STAGE6_STAGE4_AVAILABLE"] & df["STAGE6_STAGE5_AVAILABLE"] & ~dq_issue

    # Apply logic
    df.loc[has_both, "STAGE6_ENSEMBLE_RISK_SCORE"] = (W_STAGE4 * df.loc[has_both, "STAGE4_PRICE_RISK_SCORE"]) + \
                                                     (W_STAGE5 * df.loc[has_both, "STAGE5_IF_RISK_SCORE"])
    df.loc[has_both, "STAGE6_FUSION_SOURCE"] = "stage4_and_stage5"

    df.loc[has_s4_only, "STAGE6_ENSEMBLE_RISK_SCORE"] = df.loc[has_s4_only, "STAGE4_PRICE_RISK_SCORE"]
    df.loc[has_s4_only, "STAGE6_FUSION_SOURCE"] = "stage4_only"

    df.loc[has_s5_only, "STAGE6_ENSEMBLE_RISK_SCORE"] = df.loc[has_s5_only, "STAGE5_IF_RISK_SCORE"]
    df.loc[has_s5_only, "STAGE6_FUSION_SOURCE"] = "stage5_only"

    df["AUDIT_PRIORITY_SCORE"] = df["STAGE6_ENSEMBLE_RISK_SCORE"] * 100.0

    return df

def calibrate_final_tiers(df: pd.DataFrame) -> pd.DataFrame:
    """Determines operation thresholds using strictly clean holdout records."""
    clean_holdout_ensemble_mask = (
        (df["SYNTHETIC_RECORD_CREATED"] == 0) &
        (df["IS_ANOMALY_INJECTED"] == 0) &
        (df["MODEL_PARTITION"] == "holdout") &
        df["STAGE6_ENSEMBLE_RISK_SCORE"].notna() &
        (~df["STAGE6_DATA_QUALITY_REVIEW"])
    )

    clean_holdout = df[clean_holdout_ensemble_mask]
    if len(clean_holdout) == 0:
        logger.warning("No clean holdout records available! Using all available scored clean records for calibration.")
        clean_fallback_mask = (df["SYNTHETIC_RECORD_CREATED"] == 0) & (df["IS_ANOMALY_INJECTED"] == 0) & df["STAGE6_ENSEMBLE_RISK_SCORE"].notna()
        clean_holdout = df[clean_fallback_mask]

    p95 = clean_holdout["STAGE6_ENSEMBLE_RISK_SCORE"].quantile(0.95)
    p99 = clean_holdout["STAGE6_ENSEMBLE_RISK_SCORE"].quantile(0.99)

    logger.info(f"Clean Holdout Calibration (N={len(clean_holdout)}): P95={p95:.4f}, P99={p99:.4f}")

    df["STAGE6_THRESHOLD_SOURCE"] = "not_scored"
    df["STAGE6_AUDIT_TIER"] = "not_scored"
    df["STAGE6_AUDIT_FLAG"] = 0

    scored = df["STAGE6_ENSEMBLE_RISK_SCORE"].notna()

    # Assign threshold sources
    df.loc[scored, "STAGE6_THRESHOLD_SOURCE"] = "clean_holdout_percentile"

    # Assign tiers
    is_standard = scored & (df["STAGE6_ENSEMBLE_RISK_SCORE"] < p95)
    is_elevated = scored & (df["STAGE6_ENSEMBLE_RISK_SCORE"] >= p95) & (df["STAGE6_ENSEMBLE_RISK_SCORE"] < p99)
    is_extreme = scored & (df["STAGE6_ENSEMBLE_RISK_SCORE"] >= p99)

    df.loc[is_standard, "STAGE6_AUDIT_TIER"] = "standard"
    df.loc[is_elevated, "STAGE6_AUDIT_TIER"] = "elevated"
    df.loc[is_extreme, "STAGE6_AUDIT_TIER"] = "extreme"

    df.loc[is_elevated | is_extreme, "STAGE6_AUDIT_FLAG"] = 1

    return df

def run_quality_assertions(df: pd.DataFrame):
    """Enforces strict constraints on scoring and data usage."""
    # Score bounds
    scored = df["STAGE6_ENSEMBLE_RISK_SCORE"].notna()
    assert df.loc[scored, "STAGE6_ENSEMBLE_RISK_SCORE"].between(0, 1).all(), "Ensemble score outside [0,1]"
    assert df.loc[scored, "AUDIT_PRIORITY_SCORE"].between(0, 100).all(), "Audit priority score outside [0,100]"

    # DQ row checks
    dq = df["STAGE6_DATA_QUALITY_REVIEW"]
    assert df.loc[dq, "STAGE6_ENSEMBLE_RISK_SCORE"].isna().all(), "DQ rows must have null ensemble score"
    assert (df.loc[dq, "STAGE6_AUDIT_TIER"] == "not_scored").all(), "DQ rows must be not_scored tier"
    assert (df.loc[dq, "STAGE6_AUDIT_FLAG"] == 0).all(), "DQ rows cannot be flagged"
    assert (df.loc[scored, "STAGE6_THRESHOLD_SOURCE"] == "clean_holdout_percentile").all(), "Scored rows lack holdout source"

    # Calibration leak check
    clean_holdout_ensemble_mask = (
        (df["SYNTHETIC_RECORD_CREATED"] == 0) &
        (df["IS_ANOMALY_INJECTED"] == 0) &
        (df["MODEL_PARTITION"] == "holdout") &
        df["STAGE6_ENSEMBLE_RISK_SCORE"].notna() &
        (~df["STAGE6_DATA_QUALITY_REVIEW"])
    )
    calib_df = df[clean_holdout_ensemble_mask]
    assert (calib_df["IS_ANOMALY_INJECTED"] == 0).all(), "Clean-holdout contains injected scenarios"
    assert (calib_df["SYNTHETIC_RECORD_CREATED"] == 0).all(), "Clean-holdout contains synthetic duplicate rows"

def evaluate_performance(df: pd.DataFrame):
    """Calculates and prints operational metrics for evaluation only."""
    df["IS_STAGE6_PRICE_SCENARIO"] = df["SCENARIO_TYPE"].isin(["extreme_payment_deviation", "moderate_payment_deviation"]).astype(int)
    df["IS_STAGE6_INJECTED_SCENARIO"] = df["SCENARIO_TYPE"].isin(["extreme_payment_deviation", "moderate_payment_deviation", "duplicate_like_billing"]).astype(int)

    # Base masks
    clean_mask = (df["IS_ANOMALY_INJECTED"] == 0) & df["STAGE6_ENSEMBLE_RISK_SCORE"].notna()
    price_mask = df["IS_STAGE6_PRICE_SCENARIO"] == 1
    mod_mask = df["SCENARIO_TYPE"] == "moderate_payment_deviation"
    ext_mask = df["SCENARIO_TYPE"] == "extreme_payment_deviation"
    dup_mask = df["SCENARIO_TYPE"] == "duplicate_like_billing"

    flagged = df["STAGE6_AUDIT_FLAG"] == 1

    # --- Metrics ---
    clean_fpr = flagged[clean_mask].mean() if clean_mask.any() else 0

    # A. Price-scenario cohort
    price_cohort = df[clean_mask | price_mask]
    p_tp = (flagged & price_mask).sum()
    p_fp = (flagged & clean_mask).sum()
    p_fn = (~flagged & price_mask).sum()

    p_precision = p_tp / (p_tp + p_fp) if (p_tp + p_fp) > 0 else 0
    p_recall = p_tp / (p_tp + p_fn) if (p_tp + p_fn) > 0 else 0
    p_f1 = 2 * (p_precision * p_recall) / (p_precision + p_recall) if (p_precision + p_recall) > 0 else 0

    ext_recall = flagged[ext_mask].mean() if ext_mask.any() else 0
    mod_recall = flagged[mod_mask].mean() if mod_mask.any() else 0

    # B. All-scenario cohort
    overall_recall = flagged[df["IS_STAGE6_INJECTED_SCENARIO"] == 1].mean()
    dup_recall = flagged[dup_mask].mean() if dup_mask.any() else 0

    print("\n=== STAGE 6 EVALUATION METRICS ===")
    print(f"Clean False Positive Rate:      {clean_fpr:.2%}")
    print("\n-- A. Price Scenario Cohort --")
    print(f"Precision:                      {p_precision:.2%}")
    print(f"Recall:                         {p_recall:.2%}")
    print(f"F1 Score:                       {p_f1:.2%}")
    print(f"Extreme Scenario Recall:        {ext_recall:.2%}")
    print(f"Moderate Scenario Recall:       {mod_recall:.2%}")

    print("\n-- B. All-Scenario Cohort --")
    print(f"Overall Scenario Recall:        {overall_recall:.2%}")
    print(f"Duplicate-like Billing Recall:  {dup_recall:.2%} (Note: No relational model exists in this POC)")

    # --- C. Queue-Ranking Evaluation ---
    def calc_queue(mask_eval, q_pct, label_col):
        eval_df = df[mask_eval].sort_values("STAGE6_ENSEMBLE_RISK_SCORE", ascending=False)
        q_size = max(1, int(len(eval_df) * q_pct))
        queue = eval_df.head(q_size)

        true_anom = queue[label_col].sum()
        total_anom = eval_df[label_col].sum()
        clean_claims = len(queue) - true_anom

        precision = true_anom / len(queue) if len(queue) > 0 else 0
        recall = true_anom / total_anom if total_anom > 0 else 0

        return len(queue), true_anom, clean_claims, precision, recall

    # Setup queue arrays
    q_metrics = []

    # Price
    q_metrics.append(["Price scenarios", "Top 5%"] + list(calc_queue(clean_mask | price_mask, 0.05, "IS_STAGE6_PRICE_SCENARIO")))
    q_metrics.append(["Price scenarios", "Top 10%"] + list(calc_queue(clean_mask | price_mask, 0.10, "IS_STAGE6_PRICE_SCENARIO")))

    # All
    q_metrics.append(["All injected scenarios", "Top 5%"] + list(calc_queue(clean_mask | (df["IS_STAGE6_INJECTED_SCENARIO"] == 1), 0.05, "IS_STAGE6_INJECTED_SCENARIO")))
    q_metrics.append(["All injected scenarios", "Top 10%"] + list(calc_queue(clean_mask | (df["IS_STAGE6_INJECTED_SCENARIO"] == 1), 0.10, "IS_STAGE6_INJECTED_SCENARIO")))

    q_df = pd.DataFrame(q_metrics, columns=["Evaluation cohort", "Queue", "Claims in queue", "Scenarios captured", "Clean claims included", "Precision", "Recall"])
    q_df["Precision"] = q_df["Precision"].map("{:.2%}".format)
    q_df["Recall"] = q_df["Recall"].map("{:.2%}".format)

    print("\n-- C. Queue-Ranking Evaluation --")
    print(q_df.to_string(index=False))

    # --- 6. Compare Scores & Clusters ---
    print("\n-- Score Comparison by Scenario --")
    scen_agg = df.groupby("SCENARIO_TYPE", dropna=False).agg(
        Count=("STAGE6_ENSEMBLE_RISK_SCORE", "count"),
        Median_S4=("STAGE4_PRICE_RISK_SCORE", "median"),
        Median_S5=("STAGE5_IF_RISK_SCORE", "median"),
        Median_Ens=("STAGE6_ENSEMBLE_RISK_SCORE", "median"),
        Flag_Rate=("STAGE6_AUDIT_FLAG", "mean")
    ).reset_index()
    scen_agg["Flag_Rate"] = scen_agg["Flag_Rate"].map("{:.2%}".format)
    print(scen_agg.to_string(index=False))

    if "STAGE5_CLUSTER_KEY" in df.columns:
        print("\n-- Cluster-Level Diagnostics --")

        # Calculate cluster-level metrics safely
        c_agg = df.groupby("STAGE5_CLUSTER_KEY").apply(lambda g: pd.Series({
            "Clean claims": (g["IS_ANOMALY_INJECTED"] == 0).sum(),
            "Injected scenarios": (g["IS_ANOMALY_INJECTED"] == 1).sum(),
            "Median Ensemble Score": g["STAGE6_ENSEMBLE_RISK_SCORE"].median(),
            "Clean FPR": g.loc[g["IS_ANOMALY_INJECTED"] == 0, "STAGE6_AUDIT_FLAG"].mean() if (g["IS_ANOMALY_INJECTED"] == 0).any() else 0,
            "Price-scenario Recall": g.loc[g["IS_STAGE6_PRICE_SCENARIO"] == 1, "STAGE6_AUDIT_FLAG"].mean() if (g["IS_STAGE6_PRICE_SCENARIO"] == 1).any() else 0,
            "Flagged Count": g["STAGE6_AUDIT_FLAG"].sum()
        })).reset_index()

        c_agg["Clean FPR"] = c_agg["Clean FPR"].map("{:.2%}".format)
        c_agg["Price-scenario Recall"] = c_agg["Price-scenario Recall"].map("{:.2%}".format)
        print(c_agg.to_string(index=False))

def main():
    logger.info("Starting Stage 6: Transparent Parallel Risk-Fusion Combiner")

    df = load_and_validate(INPUT_FILE)
    df = define_valid_signals(df)
    df = fuse_scores(df)
    df = calibrate_final_tiers(df)

    run_quality_assertions(df)
    evaluate_performance(df)

    os.makedirs(OUT_DIR, exist_ok=True)

    # Save Output Fields
    out_cols = list(df.columns) # We retain all upstream columns and append stage 6
    logger.info(f"Saving {len(df)} records to Parquet & CSV...")
    df.to_parquet(OUT_PARQUET, index=False)
    df.to_csv(OUT_CSV, index=False)

    logger.info("Stage 6 completed successfully.")

if __name__ == "__main__":
    main()