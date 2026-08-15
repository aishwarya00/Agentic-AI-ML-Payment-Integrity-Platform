"""
Stage 4: Transaction-Level Price-Deviation Anomaly Engine
(Colab Output & File Storage Enabled)
"""

import os
import logging
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def run_stage4_price_deviation_engine(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies Stage 4 Price-Deviation logic to CMS claims data.
    (Model logic and assertions preserved exactly as defined)
    """
    df = df.copy()

    logging.info("Starting Stage 4: Transaction-Level Price-Deviation Engine...")

    # ---------------------------------------------------------
    # 1. Cluster Key Normalization
    # ---------------------------------------------------------
    df["STAGE4_CLUSTER_KEY"] = df["kmeans_cluster"].astype(str).str.strip()

    # ---------------------------------------------------------
    # 2. Correct Zero-Payment & Ineligible Handling
    # ---------------------------------------------------------
    if "WORKING_PMT_AMT" not in df.columns:
        possible_pmt_cols = ["LINE_CVRD_PD_AMT", "PMT_AMT", "PAYMENT_AMOUNT"]
        for col in possible_pmt_cols:
            if col in df.columns:
                df["WORKING_PMT_AMT"] = df[col]
                break

    df["PRICE_SIGNAL_ELIGIBLE"] = (
        df["WORKING_PMT_AMT"].gt(0) &
        df["EXPECTED_PMT_FINAL"].gt(0) &
        df["RELATIVE_SURGE_RATIO_FINAL"].notna() &
        df["PAYMENT_RESIDUAL_FINAL"].notna()
    )

    # Initialize Stage 4 columns for all rows with "not_eligible" safe defaults
    df["STAGE4_POSITIVE_PAYMENT_RESIDUAL"] = np.nan
    df["STAGE4_RELATIVE_SURGE_PERCENTILE"] = np.nan
    df["STAGE4_POSITIVE_RESIDUAL_PERCENTILE"] = np.nan
    df["STAGE4_PRICE_RISK_SCORE"] = np.nan
    df["STAGE4_PRICE_TIER"] = "not_eligible"
    df["STAGE4_PRICE_FLAG"] = 0
    df["STAGE4_THRESHOLD_SOURCE"] = "not_eligible"

    df["STAGE4_APPLIED_SURGE_P95"] = np.nan
    df["STAGE4_APPLIED_SURGE_P99"] = np.nan
    df["STAGE4_APPLIED_RESIDUAL_P90"] = np.nan
    df["STAGE4_APPLIED_RESIDUAL_P95"] = np.nan

    # For eligible rows only: Calculate positive residual
    eligible_mask = df["PRICE_SIGNAL_ELIGIBLE"] == True
    df.loc[eligible_mask, "STAGE4_POSITIVE_PAYMENT_RESIDUAL"] = np.maximum(
        df.loc[eligible_mask, "PAYMENT_RESIDUAL_FINAL"],
        0
    )

# ---------------------------------------------------------
    # 3. Threshold Calibration
    # ---------------------------------------------------------
    # Safe mask construction in case split/injection metadata columns are missing
    syn_mask = (df["SYNTHETIC_RECORD_CREATED"] == 0) if "SYNTHETIC_RECORD_CREATED" in df.columns else True
    inj_mask = (df["IS_ANOMALY_INJECTED"] == 0) if "IS_ANOMALY_INJECTED" in df.columns else True
    part_mask = (df["MODEL_PARTITION"] == "holdout") if "MODEL_PARTITION" in df.columns else True

    clean_holdout_mask = (
        syn_mask &
        inj_mask &
        part_mask &
        (df["PRICE_SIGNAL_ELIGIBLE"] == True)
    )

    clean_holdout_df = df[clean_holdout_mask]

    # Calculate global fallback metrics
    global_surge_p95 = clean_holdout_df["RELATIVE_SURGE_RATIO_FINAL"].quantile(0.95)
    global_surge_p99 = clean_holdout_df["RELATIVE_SURGE_RATIO_FINAL"].quantile(0.99)
    global_res_p90 = clean_holdout_df["STAGE4_POSITIVE_PAYMENT_RESIDUAL"].quantile(0.90)
    global_res_p95 = clean_holdout_df["STAGE4_POSITIVE_PAYMENT_RESIDUAL"].quantile(0.95)

    # Extract arrays for efficient percentile calculations later
    global_surge_dist = np.sort(clean_holdout_df["RELATIVE_SURGE_RATIO_FINAL"].dropna().values)
    global_res_dist = np.sort(clean_holdout_df["STAGE4_POSITIVE_PAYMENT_RESIDUAL"].dropna().values)

    # Process metrics using standard cluster keys
    for cluster_key in df["STAGE4_CLUSTER_KEY"].unique():
        cluster_overall_mask = (df["STAGE4_CLUSTER_KEY"] == cluster_key) & eligible_mask
        cluster_clean_holdout = clean_holdout_df[clean_holdout_df["STAGE4_CLUSTER_KEY"] == cluster_key]

        if len(cluster_clean_holdout) >= 30:
            surge_p95 = cluster_clean_holdout["RELATIVE_SURGE_RATIO_FINAL"].quantile(0.95)
            surge_p99 = cluster_clean_holdout["RELATIVE_SURGE_RATIO_FINAL"].quantile(0.99)
            res_p90 = cluster_clean_holdout["STAGE4_POSITIVE_PAYMENT_RESIDUAL"].quantile(0.90)
            res_p95 = cluster_clean_holdout["STAGE4_POSITIVE_PAYMENT_RESIDUAL"].quantile(0.95)
            thresh_source = "cluster_holdout"

            surge_dist = np.sort(cluster_clean_holdout["RELATIVE_SURGE_RATIO_FINAL"].dropna().values)
            res_dist = np.sort(cluster_clean_holdout["STAGE4_POSITIVE_PAYMENT_RESIDUAL"].dropna().values)
        else:
            surge_p95 = global_surge_p95
            surge_p99 = global_surge_p99
            res_p90 = global_res_p90
            res_p95 = global_res_p95
            thresh_source = "global_holdout_fallback"

            surge_dist = global_surge_dist
            res_dist = global_res_dist

        # Store applied threshold values to the main dataframe
        df.loc[cluster_overall_mask, "STAGE4_APPLIED_SURGE_P95"] = surge_p95
        df.loc[cluster_overall_mask, "STAGE4_APPLIED_SURGE_P99"] = surge_p99
        df.loc[cluster_overall_mask, "STAGE4_APPLIED_RESIDUAL_P90"] = res_p90
        df.loc[cluster_overall_mask, "STAGE4_APPLIED_RESIDUAL_P95"] = res_p95
        df.loc[cluster_overall_mask, "STAGE4_THRESHOLD_SOURCE"] = thresh_source

        # Fast percentile ranking using searchsorted
        if len(surge_dist) > 0 and len(res_dist) > 0:
            df.loc[cluster_overall_mask, "STAGE4_RELATIVE_SURGE_PERCENTILE"] = (
                np.searchsorted(surge_dist, df.loc[cluster_overall_mask, "RELATIVE_SURGE_RATIO_FINAL"]) / len(surge_dist)
            )
            df.loc[cluster_overall_mask, "STAGE4_POSITIVE_RESIDUAL_PERCENTILE"] = (
                np.searchsorted(res_dist, df.loc[cluster_overall_mask, "STAGE4_POSITIVE_PAYMENT_RESIDUAL"]) / len(res_dist)
            )

    # ---------------------------------------------------------
    # 4. Risk Scoring and Tiers
    # ---------------------------------------------------------
    df.loc[eligible_mask, "STAGE4_PRICE_RISK_SCORE"] = (
        0.70 * df.loc[eligible_mask, "STAGE4_RELATIVE_SURGE_PERCENTILE"] +
        0.30 * df.loc[eligible_mask, "STAGE4_POSITIVE_RESIDUAL_PERCENTILE"]
    )

    df.loc[eligible_mask, "STAGE4_PRICE_TIER"] = "standard"

    elevated_mask = eligible_mask & (
        df["RELATIVE_SURGE_RATIO_FINAL"] >= df["STAGE4_APPLIED_SURGE_P95"]
    ) & (
        df["STAGE4_POSITIVE_PAYMENT_RESIDUAL"] >= df["STAGE4_APPLIED_RESIDUAL_P90"]
    )
    df.loc[elevated_mask, "STAGE4_PRICE_TIER"] = "elevated"
    df.loc[elevated_mask, "STAGE4_PRICE_FLAG"] = 1

    extreme_mask = eligible_mask & (
        df["RELATIVE_SURGE_RATIO_FINAL"] >= df["STAGE4_APPLIED_SURGE_P99"]
    ) & (
        df["STAGE4_POSITIVE_PAYMENT_RESIDUAL"] >= df["STAGE4_APPLIED_RESIDUAL_P95"]
    )
    df.loc[extreme_mask, "STAGE4_PRICE_TIER"] = "extreme"
    df.loc[extreme_mask, "STAGE4_PRICE_FLAG"] = 1

    # ---------------------------------------------------------
    # 5. Evaluation Correction
    # ---------------------------------------------------------
    if "SCENARIO_TYPE" in df.columns:
        df["IS_STAGE4_PRICE_SCENARIO"] = df["SCENARIO_TYPE"].isin([
            "extreme_payment_deviation",
            "moderate_payment_deviation"
        ]).astype(int)

    # ---------------------------------------------------------
    # 6. Assertions
    # ---------------------------------------------------------
    # 6a. Calibration purity assertions
    if "IS_ANOMALY_INJECTED" in clean_holdout_df.columns:
        assert clean_holdout_df["IS_ANOMALY_INJECTED"].sum() == 0, "Assertion Failed: Injected rows used in threshold calibration."
    if "SYNTHETIC_RECORD_CREATED" in clean_holdout_df.columns:
        assert clean_holdout_df["SYNTHETIC_RECORD_CREATED"].sum() == 0, "Assertion Failed: Synthetic duplicate rows used in threshold calibration."

    ineligible_mask = ~df["PRICE_SIGNAL_ELIGIBLE"]

    assert clean_holdout_df["IS_ANOMALY_INJECTED"].sum() == 0, "Assertion Failed: Injected rows used in threshold calibration."
    assert clean_holdout_df["SYNTHETIC_RECORD_CREATED"].sum() == 0, "Assertion Failed: Synthetic duplicate rows used in threshold calibration."

    assert df.loc[ineligible_mask, "STAGE4_PRICE_RISK_SCORE"].isna().all(), "Assertion Failed: Price-ineligible row has a non-null Stage 4 score."
    assert (df.loc[ineligible_mask, "STAGE4_PRICE_FLAG"] == 0).all(), "Assertion Failed: Price-ineligible row has STAGE4_PRICE_FLAG = 1."
    assert (df.loc[ineligible_mask, "STAGE4_PRICE_TIER"] == "not_eligible").all(), "Assertion Failed: Price-ineligible row has a tier other than 'not_eligible'."

    assert df.loc[eligible_mask, "STAGE4_PRICE_RISK_SCORE"].between(0, 1).all(), "Assertion Failed: Non-null Stage 4 scores are not between 0 and 1."

    required_threshold_cols = [
        "STAGE4_APPLIED_SURGE_P95",
        "STAGE4_APPLIED_SURGE_P99",
        "STAGE4_APPLIED_RESIDUAL_P90",
        "STAGE4_APPLIED_RESIDUAL_P95"
    ]
    assert df.loc[eligible_mask, required_threshold_cols].notna().all().all(), "Assertion Failed: Not all eligible rows have applied threshold values populated."

    assert df.loc[eligible_mask, "STAGE4_THRESHOLD_SOURCE"].isin(["cluster_holdout", "global_holdout_fallback"]).all(), "Assertion Failed: Invalid threshold source on eligible row."

    logging.info("Stage 4 Processing & Assertions completed successfully.")

    return df


def save_stage4_outputs(df: pd.DataFrame, output_dir: str = "data/processed"):
    """
    Saves Stage 4 scored output to parquet and csv, creating directories if missing.
    """
    os.makedirs(output_dir, exist_ok=True)

    parquet_path = os.path.abspath(os.path.join(output_dir, "cms_claims_stage4_price_scored.parquet"))
    csv_path = os.path.abspath(os.path.join(output_dir, "cms_claims_stage4_price_scored.csv"))

    df.to_parquet(parquet_path, index=False)
    df.to_csv(csv_path, index=False)

    print("\n" + "="*80)
    print("STAGE 4 FILE SAVE VERIFICATION")
    print("="*80)
    print(f"Parquet saved -> {parquet_path} ({os.path.getsize(parquet_path) / (1024*1024):.2f} MB)")
    print(f"CSV saved     -> {csv_path} ({os.path.getsize(csv_path) / (1024*1024):.2f} MB)")
    print("="*80)


def display_stage4_results(df: pd.DataFrame):
    """
    Prints a clean summary of Stage 4 execution directly in Colab output cell.
    """
    print("\n" + "="*80)
    print("STAGE 4 PRICE-DEVIATION ANOMALY ENGINE RESULTS SUMMARY")
    print("="*80)
    print(f"Total Claims Processed: {len(df):,}")
    print(f"Eligible Claims:       {df['PRICE_SIGNAL_ELIGIBLE'].sum():,} ({df['PRICE_SIGNAL_ELIGIBLE'].mean():.2%})")
    print(f"Ineligible Claims:     {(~df['PRICE_SIGNAL_ELIGIBLE']).sum():,}\n")

    print("--- TIER BREAKDOWN ---")
    tier_counts = df["STAGE4_PRICE_TIER"].value_counts(dropna=False)
    for tier, count in tier_counts.items():
        print(f"  {tier:<15}: {count:>8,} ({count/len(df):.2%})")

    print("\n--- PRICE FLAG BREAKDOWN ---")
    flag_counts = df["STAGE4_PRICE_FLAG"].value_counts(dropna=False)
    for flag, count in flag_counts.items():
        print(f"  Flag = {flag}: {count:>8,} ({count/len(df):.2%})")

    print("\n--- THRESHOLD SOURCE DISTRIBUTION ---")
    source_counts = df["STAGE4_THRESHOLD_SOURCE"].value_counts(dropna=False)
    for src, count in source_counts.items():
        print(f"  {src:<25}: {count:>8,} ({count/len(df):.2%})")

    eligible_scores = df.loc[df["PRICE_SIGNAL_ELIGIBLE"], "STAGE4_PRICE_RISK_SCORE"]
    if len(eligible_scores) > 0:
        print("\n--- RISK SCORE STATS (ELIGIBLE ROWS) ---")
        print(f"  Min:  {eligible_scores.min():.4f}")
        print(f"  P25:  {eligible_scores.quantile(0.25):.4f}")
        print(f"  Mean: {eligible_scores.mean():.4f}")
        print(f"  P75:  {eligible_scores.quantile(0.75):.4f}")
        print(f"  Max:  {eligible_scores.max():.4f}")

    print("\n--- SAMPLE OUTPUT PREVIEW ---")
    preview_cols = [col for col in [
        "STAGE4_CLUSTER_KEY", "PRICE_SIGNAL_ELIGIBLE",
        "STAGE4_PRICE_RISK_SCORE", "STAGE4_PRICE_TIER",
        "STAGE4_PRICE_FLAG", "STAGE4_THRESHOLD_SOURCE"
    ] if col in df.columns]

    # Display top flagged anomalies if present, else head
    sample_df = df[df["STAGE4_PRICE_FLAG"] == 1]
    if sample_df.empty:
        sample_df = df
    display_cols = preview_cols if len(preview_cols) > 0 else df.columns[:6]
    print(sample_df[display_cols].head(5).to_string(index=False))
    print("="*80 + "\n")


# ==============================================================================
# COLAB EXECUTION BLOCK
# Reads Stage 3 input, runs Stage 4 Engine, outputs summary, saves results
# ==============================================================================
input_path = "data/processed/cms_claims_stage3_calibrated.parquet"

if os.path.exists(input_path):
    print(f"Loading Stage 3 input from: {input_path}")
    df_stage3 = pd.read_parquet(input_path)

    # Run Stage 4 Pipeline
    df_stage4_scored = run_stage4_price_deviation_engine(df_stage3)

    # Save Outputs
    save_stage4_outputs(df_stage4_scored, output_dir="data/processed")

    # Print Results directly to Colab cell
    display_stage4_results(df_stage4_scored)
else:
    print(f"WARNING: Stage 3 file not found at '{input_path}'.")
    print("Please ensure the prior stage notebook has saved 'data/processed/cms_claims_stage3_calibrated.parquet'.")
    
    
import os
import logging
import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def evaluate_performance(df: pd.DataFrame):
    """
    Generates all requested evaluation tables and diagnostics with defensive safety guards.
    """
    logger.info("Evaluating top-of-queue and classification metrics...")
    df = df.copy()

    # ---------------------------------------------------------
    # 0. Defensive Column Normalization & Guarantees
    # ---------------------------------------------------------
    if "SCENARIO_TYPE" not in df.columns:
        df["SCENARIO_TYPE"] = "clean"

    if "IS_PRICE_DEVIATION_SCENARIO" not in df.columns:
        df["IS_PRICE_DEVIATION_SCENARIO"] = df["SCENARIO_TYPE"].isin([
            "extreme_payment_deviation", "moderate_payment_deviation"
        ]).astype(int)

    if "IS_ANOMALY_INJECTED" not in df.columns:
        df["IS_ANOMALY_INJECTED"] = (df["SCENARIO_TYPE"] != "clean").astype(int)

    if "PRICE_SIGNAL_ELIGIBLE" not in df.columns:
        df["PRICE_SIGNAL_ELIGIBLE"] = True

    if "STAGE4_PRICE_FLAG" not in df.columns:
        df["STAGE4_PRICE_FLAG"] = 0

    if "STAGE4_PRICE_RISK_SCORE" not in df.columns:
        df["STAGE4_PRICE_RISK_SCORE"] = 0.0

    if "STAGE4_PRICE_TIER" not in df.columns:
        df["STAGE4_PRICE_TIER"] = "standard"

    # Cluster column fallback
    if "kmeans_cluster" in df.columns:
        cluster_col = "kmeans_cluster"
    elif "STAGE4_CLUSTER_KEY" in df.columns:
        cluster_col = "STAGE4_CLUSTER_KEY"
    else:
        df["STAGE4_CLUSTER_KEY"] = "0"
        cluster_col = "STAGE4_CLUSTER_KEY"

    # ---------------------------------------------------------
    # 1. Primary Metrics
    # ---------------------------------------------------------
    primary_mask = (df["IS_ANOMALY_INJECTED"] == 0) | (df["IS_PRICE_DEVIATION_SCENARIO"] == 1)
    eval_df = df[primary_mask & (df["PRICE_SIGNAL_ELIGIBLE"] == True)].copy()

    y_true = eval_df["IS_PRICE_DEVIATION_SCENARIO"]
    y_pred = eval_df["STAGE4_PRICE_FLAG"]

    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        clean_fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    else:
        clean_fpr = 0.0

    tier_counts = eval_df["STAGE4_PRICE_TIER"].value_counts()

    ext_mask = eval_df["SCENARIO_TYPE"] == "extreme_payment_deviation"
    mod_mask = eval_df["SCENARIO_TYPE"] == "moderate_payment_deviation"

    ext_recall = eval_df[ext_mask]["STAGE4_PRICE_FLAG"].mean() if ext_mask.sum() > 0 else 0.0
    mod_recall = eval_df[mod_mask]["STAGE4_PRICE_FLAG"].mean() if mod_mask.sum() > 0 else 0.0

    primary_metrics = {
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "Clean FPR": clean_fpr,
        "Elevated Count": tier_counts.get("elevated", 0),
        "Extreme Count": tier_counts.get("extreme", 0),
        "Extreme Scenario Recall": ext_recall,
        "Moderate Scenario Recall": mod_recall
    }

    # ---------------------------------------------------------
    # 2. Duplicate-like Billing Diagnostic
    # ---------------------------------------------------------
    dup_mask = (df["SCENARIO_TYPE"] == "duplicate_like_billing")
    dup_df = df[dup_mask]
    dup_diagnostic = {
        "Total Rows": len(dup_df),
        "Pct Eligible": dup_df["PRICE_SIGNAL_ELIGIBLE"].mean() if len(dup_df) > 0 else 0.0,
        "Pct Flagged": dup_df["STAGE4_PRICE_FLAG"].mean() if len(dup_df) > 0 else 0.0,
        "Median Score": dup_df["STAGE4_PRICE_RISK_SCORE"].median() if len(dup_df) > 0 else 0.0
    }

    # ---------------------------------------------------------
    # 3. Top of Queue Ranking Evaluation
    # ---------------------------------------------------------
    eval_df_sorted = eval_df.sort_values(by="STAGE4_PRICE_RISK_SCORE", ascending=False)
    total_eligible = len(eval_df_sorted)
    total_positives = y_true.sum()

    queue_tables = []
    for cutoff_pct in [0.05, 0.10]:
        k = int(total_eligible * cutoff_pct)
        top_k = eval_df_sorted.head(k)

        captured_pos = top_k["IS_PRICE_DEVIATION_SCENARIO"].sum()
        clean_included = (top_k["IS_ANOMALY_INJECTED"] == 0).sum()

        q_prec = captured_pos / k if k > 0 else 0.0
        q_rec = captured_pos / total_positives if total_positives > 0 else 0.0

        queue_tables.append({
            "Queue Threshold": f"Top {int(cutoff_pct*100)}%",
            "Number of Claims": k,
            "Price Scenarios Captured": captured_pos,
            "Clean Claims Included": clean_included,
            "Precision": f"{q_prec*100:.1f}%",
            "Recall": f"{q_rec*100:.1f}%"
        })

    queue_df = pd.DataFrame(queue_tables)

    # ---------------------------------------------------------
    # 4. Cluster-Level Diagnostics
    # ---------------------------------------------------------
    cluster_diags = []
    for c_id in df[cluster_col].dropna().unique():
        c_sub = eval_df[eval_df[cluster_col] == c_id]
        if len(c_sub) == 0:
            continue

        c_clean = c_sub[c_sub["IS_ANOMALY_INJECTED"] == 0]
        c_pos = c_sub[c_sub["IS_PRICE_DEVIATION_SCENARIO"] == 1]

        c_ext_mask = c_sub["SCENARIO_TYPE"] == "extreme_payment_deviation"
        c_mod_mask = c_sub["SCENARIO_TYPE"] == "moderate_payment_deviation"

        cfpr = c_clean["STAGE4_PRICE_FLAG"].mean() if len(c_clean) > 0 else 0.0
        thresh_src = c_sub["STAGE4_THRESHOLD_SOURCE"].iloc[0] if "STAGE4_THRESHOLD_SOURCE" in c_sub.columns and len(c_sub) > 0 else "N/A"

        cluster_diags.append({
            "Cluster": c_id,
            "Clean Eligible Claims": len(c_clean),
            "Price Deviation Scenarios": len(c_pos),
            "Total Flags": c_sub["STAGE4_PRICE_FLAG"].sum(),
            "Clean FPR": f"{cfpr*100:.2f}%",
            "Extreme Recall": f"{(c_sub[c_ext_mask]['STAGE4_PRICE_FLAG'].mean() if c_ext_mask.sum() > 0 else 0.0)*100:.1f}%",
            "Moderate Recall": f"{(c_sub[c_mod_mask]['STAGE4_PRICE_FLAG'].mean() if c_mod_mask.sum() > 0 else 0.0)*100:.1f}%",
            "Median Risk Score": f"{c_sub['STAGE4_PRICE_RISK_SCORE'].median():.3f}",
            "Threshold Source": thresh_src
        })

    cluster_df = pd.DataFrame(cluster_diags)

    return primary_metrics, dup_diagnostic, queue_df, cluster_df


def print_evaluation_report(primary_metrics, dup_diagnostic, queue_df, cluster_df):
    """
    Helper function to cleanly format and display the evaluation output.
    """
    print("\n" + "="*70)
    print("STAGE 4: PRIMARY EVALUATION METRICS")
    print("="*70)
    for k, v in primary_metrics.items():
        if isinstance(v, float):
            print(f"  {k:<28}: {v:.4f} ({v*100:.2f}%)" if "Count" not in k else f"  {k:<28}: {v:.4f}")
        else:
            print(f"  {k:<28}: {v:,}")

    print("\n" + "="*70)
    print("STAGE 4: DUPLICATE-LIKE BILLING DIAGNOSTIC")
    print("="*70)
    for k, v in dup_diagnostic.items():
        if isinstance(v, float):
            print(f"  {k:<28}: {v:.4f}" + (f" ({v*100:.2f}%)" if "Pct" in k else ""))
        else:
            print(f"  {k:<28}: {v:,}")

    print("\n" + "="*70)
    print("STAGE 4: TOP OF QUEUE RANKING EVALUATION")
    print("="*70)
    print(queue_df.to_string(index=False))

    print("\n" + "="*70)
    print("STAGE 4: CLUSTER-LEVEL DIAGNOSTICS")
    print("="*70)
    print(cluster_df.to_string(index=False))
    print("="*70 + "\n")


# ==============================================================================
# MAIN EXECUTION
# ==============================================================================
file_path = "data/processed/cms_claims_stage4_price_scored.parquet"

if not os.path.exists(file_path):
    file_path = file_path.replace(".parquet", ".csv")

if os.path.exists(file_path):
    print(f"Loading scored Stage 4 data from: {file_path}...")
    df_scored = pd.read_parquet(file_path) if file_path.endswith(".parquet") else pd.read_csv(file_path)

    # Execute Evaluation
    primary_metrics, dup_diagnostic, queue_df, cluster_df = evaluate_performance(df_scored)

    # Print Clean Output
    print_evaluation_report(primary_metrics, dup_diagnostic, queue_df, cluster_df)
else:
    print(f"Error: Scored Stage 4 file not found at '{file_path}'. Please run the Stage 4 script first.")