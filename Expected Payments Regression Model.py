import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

# Setup Directories
os.makedirs('data/processed', exist_ok=True)
os.makedirs('models', exist_ok=True)

INPUT_PATH = "data/processed/cms_claims_injected.parquet"

if not os.path.exists(INPUT_PATH):
    raise FileNotFoundError(
        f"Stage 2 input not found: {INPUT_PATH}. "
        "Run Stage 2 before Stage 3."
    )

def calculate_clinical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derives PATIENT_AGE, AGE_MISSING_FLAG, and CHRONIC_CONDITION_COUNT
    from standard CMS claims fields.
    """
    df = df.copy()

    # -------------------------------------------------------------------------
    # 1. PATIENT_AGE & AGE_MISSING_FLAG
    # -------------------------------------------------------------------------
    # Auto-detect standard CMS birth date columns
    birth_col = next((c for c in ['BENE_BIRTH_DT', 'BENE_BIRTH_DATE', 'DOB', 'PATIENT_DOB', 'BIRTH_DT'] if c in df.columns), None)

    # Auto-detect standard CMS claim / service date columns
    claim_date_col = next((c for c in ['CLM_FROM_DT', 'CLM_THRU_DT', 'LINE_1ST_EXPNS_DT', 'SRVC_DT', 'CLAIM_DATE'] if c in df.columns), None)

    if "PATIENT_AGE" not in df.columns or df["PATIENT_AGE"].isna().all():
        if birth_col and claim_date_col:
            birth_dt = pd.to_datetime(df[birth_col], errors='coerce')
            claim_dt = pd.to_datetime(df[claim_date_col], errors='coerce')

            # Calculate age at time of service
            df["PATIENT_AGE"] = (claim_dt - birth_dt).dt.days / 365.25
        else:
            df["PATIENT_AGE"] = np.nan

    # Identify missing or out-of-range ages (< 0 or > 120)
    invalid_age_mask = df["PATIENT_AGE"].isna() | (df["PATIENT_AGE"] < 0) | (df["PATIENT_AGE"] > 120)
    df["AGE_MISSING_FLAG"] = invalid_age_mask.astype(int)

    # Impute missing ages using the median age (default to 65.0 Medicare baseline if median is empty)
    median_age = df.loc[~invalid_age_mask, "PATIENT_AGE"].median()
    if pd.isna(median_age):
        median_age = 65.0

    df["PATIENT_AGE"] = df["PATIENT_AGE"].where(~invalid_age_mask, median_age)

    # -------------------------------------------------------------------------
    # 2. CHRONIC_CONDITION_COUNT
    # -------------------------------------------------------------------------
    if "CHRONIC_CONDITION_COUNT" not in df.columns or df["CHRONIC_CONDITION_COUNT"].isna().all():
        # Look for standard CMS SynPUF condition columns (e.g., SP_ALZHMR, SP_CHF, SP_DIABETES, SP_CRDHYP)
        sp_cols = [c for c in df.columns if c.startswith("SP_") or "CHRONIC" in c.upper()]

        if sp_cols:
            # In CMS DE-SynPUF: Code 1 indicates 'Yes' for the condition (Code 2 is 'No')
            df["CHRONIC_CONDITION_COUNT"] = (df[sp_cols] == 1).sum(axis=1)
        else:
            # Fallback: Count populated ICD diagnosis code fields if SP_ columns are absent
            diag_cols = [c for c in df.columns if "ICD_DGNS" in c or "DIAG" in c]
            if diag_cols:
                df["CHRONIC_CONDITION_COUNT"] = df[diag_cols].notna().sum(axis=1)
            else:
                df["CHRONIC_CONDITION_COUNT"] = 0

    return df


# Load data
df = pd.read_parquet(INPUT_PATH)
print(f"INFO: Loaded Dataset. Total Rows: {len(df)}")

# =====================================================================
# 1. PRESERVE THE CLEAN GROUPED SPLIT
# =====================================================================

original_record_mask = (df["SYNTHETIC_RECORD_CREATED"] == 0)
clean_original_mask = (
    original_record_mask &
    (df["IS_ANOMALY_INJECTED"] == 0) &
    df["ORIGINAL_PMT_AMT"].notna()
)

# Calculate missing clinical features
df = calculate_clinical_features(df)

print("Derived features successfully:")
# print(df[['PATIENT_AGE', 'AGE_MISSING_FLAG', 'CHRONIC_CONDITION_COUNT']].head())

clean_df = df[clean_original_mask].copy()
group_col = 'CLM_ID' if 'CLM_ID' in clean_df.columns else clean_df.index
groups = clean_df[group_col] if 'CLM_ID' in clean_df.columns else clean_df.index.values

gss = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
train_idx_loc, holdout_idx_loc = next(gss.split(clean_df, groups=groups))

train_clean_indices = clean_df.iloc[train_idx_loc].index
holdout_clean_indices = clean_df.iloc[holdout_idx_loc].index

clean_train_mask = df.index.isin(train_clean_indices)
clean_holdout_mask = df.index.isin(holdout_clean_indices)

# Validate group overlap
if 'CLM_ID' in clean_df.columns:
    train_groups = set(df[clean_train_mask]['CLM_ID'])
    holdout_groups = set(df[clean_holdout_mask]['CLM_ID'])
    assert len(train_groups.intersection(holdout_groups)) == 0, "Group overlap detected!"

print(f"INFO: Clean Train Records: {clean_train_mask.sum()}, Clean Holdout Records: {clean_holdout_mask.sum()}")

# =====================================================================
# 2. CREATE A POSITIVE-PAYMENT EXPECTED-PAYMENT POPULATION
# =====================================================================
positive_payment_train_mask = clean_train_mask & (df['ORIGINAL_PMT_AMT'] > 0)
positive_payment_holdout_mask = clean_holdout_mask & (df['ORIGINAL_PMT_AMT'] > 0)

print(f"INFO: Positive-Payment Train Records: {positive_payment_train_mask.sum()}, Holdout: {positive_payment_holdout_mask.sum()}")

# Peer Benchmark calculation function
def compute_leakage_free_peer_benchmarks(train_source, target_df):
    cl_hcpcs = train_source.groupby(['kmeans_cluster', 'ORIGINAL_HCPCS_CD'])['ORIGINAL_PMT_AMT'].agg(
        tr_cl_hcpcs_count='count', tr_cl_hcpcs_median='median').reset_index()
    gl_hcpcs = train_source.groupby('ORIGINAL_HCPCS_CD')['ORIGINAL_PMT_AMT'].agg(
        tr_hcpcs_count='count', tr_hcpcs_median='median').reset_index()
    cl_gl = train_source.groupby('kmeans_cluster')['ORIGINAL_PMT_AMT'].agg(
        tr_cl_count='count', tr_cl_median='median').reset_index()

    global_median = train_source['ORIGINAL_PMT_AMT'].median()

    mapped_df = target_df.copy()
    mapped_df = mapped_df.merge(cl_hcpcs, on=['kmeans_cluster', 'ORIGINAL_HCPCS_CD'], how='left')
    mapped_df = mapped_df.merge(gl_hcpcs, on='ORIGINAL_HCPCS_CD', how='left')
    mapped_df = mapped_df.merge(cl_gl, on='kmeans_cluster', how='left')

    mapped_df['tr_cl_hcpcs_count'] = mapped_df['tr_cl_hcpcs_count'].fillna(0)
    mapped_df['tr_hcpcs_count'] = mapped_df['tr_hcpcs_count'].fillna(0)
    mapped_df['tr_cl_count'] = mapped_df['tr_cl_count'].fillna(0)

    conds = [
        mapped_df['tr_cl_hcpcs_count'] >= 5,
        mapped_df['tr_hcpcs_count'] >= 10,
        mapped_df['tr_cl_count'] >= 30
    ]

    mapped_df['MODEL_PEER_EXPECTED_PMT'] = np.select(
        conds,
        [mapped_df['tr_cl_hcpcs_median'], mapped_df['tr_hcpcs_median'], mapped_df['tr_cl_median']],
        default=global_median
    )
    mapped_df['MODEL_PEER_GROUP_LEVEL'] = np.select(
        conds, ['cluster_hcpcs', 'hcpcs_only', 'cluster_only'], default='global_fallback'
    )
    mapped_df['MODEL_PEER_GROUP_COUNT'] = np.select(
        conds, [mapped_df['tr_cl_hcpcs_count'], mapped_df['tr_hcpcs_count'], mapped_df['tr_cl_count']], default=len(train_source)
    )
    mapped_df['LOG_MODEL_PEER_EXPECTED_PMT'] = np.log1p(np.maximum(0.0, mapped_df['MODEL_PEER_EXPECTED_PMT']))

    mapped_df.drop(columns=['tr_cl_hcpcs_count', 'tr_cl_hcpcs_median', 'tr_hcpcs_count', 'tr_hcpcs_median', 'tr_cl_count', 'tr_cl_median'], inplace=True)
    return mapped_df

# =====================================================================
# 3. COMPARE THREE BASELINES ON CLEAN POSITIVE-PAYMENT HOLDOUT
# =====================================================================
train_pos_df = df[positive_payment_train_mask].copy()
holdout_pos_df = df[positive_payment_holdout_mask].copy()

# Compute peer benchmarks strictly from positive-payment training data
train_pos_df = compute_leakage_free_peer_benchmarks(train_pos_df, train_pos_df)
holdout_pos_df = compute_leakage_free_peer_benchmarks(train_pos_df, holdout_pos_df)

numeric_features = ['PATIENT_AGE', 'AGE_MISSING_FLAG', 'CHRONIC_CONDITION_COUNT', 'LOG_MODEL_PEER_EXPECTED_PMT', 'MODEL_PEER_GROUP_COUNT']
candidate_cat = ['ORIGINAL_HCPCS_CD', 'kmeans_cluster', 'MODEL_PEER_GROUP_LEVEL']
categorical_features = [c for c in candidate_cat if c in train_pos_df.columns]

for c in categorical_features:
    train_pos_df[c] = train_pos_df[c].astype(str)
    holdout_pos_df[c] = holdout_pos_df[c].astype(str)

preprocessor = ColumnTransformer([
    ('num', Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), numeric_features),
    ('cat', Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))]), categorical_features)
])

# Baseline A: Peer
y_holdout = holdout_pos_df['ORIGINAL_PMT_AMT'].values
pred_A = holdout_pos_df['MODEL_PEER_EXPECTED_PMT'].values

# Baseline B: Ridge Raw
X_train = train_pos_df[numeric_features + categorical_features]
y_train_raw = train_pos_df['ORIGINAL_PMT_AMT'].values
ridge_raw = Pipeline([('prep', preprocessor), ('model', Ridge(alpha=1.0, random_state=42))])
ridge_raw.fit(X_train, y_train_raw)
pred_B_raw = ridge_raw.predict(holdout_pos_df[numeric_features + categorical_features])
pred_B = np.maximum(0, pred_B_raw)

# Baseline C: Ridge Log + Smearing
y_train_log = np.log1p(y_train_raw)
ridge_log = Pipeline([('prep', preprocessor), ('model', Ridge(alpha=1.0, random_state=42))])
ridge_log.fit(X_train, y_train_log)

train_pred_log = ridge_log.predict(X_train)
train_log_residuals = y_train_log - train_pred_log
smearing_factor = np.mean(np.exp(train_log_residuals))
print(f"INFO: Calculated Duan Smearing Factor (Train Only): {smearing_factor:.4f}")

holdout_pred_log = ridge_log.predict(holdout_pos_df[numeric_features + categorical_features])
pred_C = np.maximum(0, (np.exp(holdout_pred_log) * smearing_factor) - 1)

def eval_metrics(actual, pred):
    wape = (np.sum(np.abs(actual - pred)) / np.sum(np.abs(actual))) * 100
    return {
        'MAE': mean_absolute_error(actual, pred),
        'Median AE': np.median(np.abs(actual - pred)),
        'RMSE': np.sqrt(mean_squared_error(actual, pred)),
        'R2': r2_score(actual, pred),
        'WAPE (%)': wape
    }

metrics_A = eval_metrics(y_holdout, pred_A)
metrics_B = eval_metrics(y_holdout, pred_B)
metrics_C = eval_metrics(y_holdout, pred_C)

print("\n--- BASELINE EVALUATION (CLEAN POSITIVE HOLDOUT) ---")
print(pd.DataFrame({'Peer benchmark': metrics_A, 'Ridge raw': metrics_B, 'Ridge log + smearing': metrics_C}).T)

# =====================================================================
# 4. SELECT THE BEST BASELINE
# =====================================================================
wape_vals = {'peer_benchmark': metrics_A['WAPE (%)'], 'ridge_raw': metrics_B['WAPE (%)'], 'ridge_log_smearing': metrics_C['WAPE (%)']}
min_wape = min(wape_vals.values())

# Selection logic (within 1% prefer simpler: Peer > Raw > Log)
if wape_vals['peer_benchmark'] <= min_wape + 1.0:
    SELECTED_EXPECTED_PMT_METHOD = 'peer_benchmark'
elif wape_vals['ridge_raw'] <= min_wape + 1.0:
    SELECTED_EXPECTED_PMT_METHOD = 'ridge_raw'
else:
    SELECTED_EXPECTED_PMT_METHOD = 'ridge_log_smearing'

print(f"\nINFO: Selected Expected-Payment Method: {SELECTED_EXPECTED_PMT_METHOD}")

# =====================================================================
# 5. REFIT ON ALL CLEAN POSITIVE-PAYMENT DATA
# =====================================================================
all_clean_pos_mask = clean_original_mask & (df['ORIGINAL_PMT_AMT'] > 0)
all_clean_pos_df = df[all_clean_pos_mask].copy()

# Rebuild peer mappings from ALL clean positive records
all_clean_pos_df = compute_leakage_free_peer_benchmarks(all_clean_pos_df, all_clean_pos_df)
df_scored = compute_leakage_free_peer_benchmarks(all_clean_pos_df, df.copy())

train_group_set = set(df.loc[train_clean_indices, "CLM_ID"])
holdout_group_set = set(df.loc[holdout_clean_indices, "CLM_ID"])

df_scored["MODEL_PARTITION"] = np.select(
    [
        df_scored["CLM_ID"].isin(train_group_set),
        df_scored["CLM_ID"].isin(holdout_group_set)
    ],
    [
        "train",
        "holdout"
    ],
    default="unassigned"
)

for c in categorical_features:
    all_clean_pos_df[c] = all_clean_pos_df[c].astype(str)
    df_scored[c] = df_scored[c].astype(str)

X_all = all_clean_pos_df[numeric_features + categorical_features]
y_all_raw = all_clean_pos_df['ORIGINAL_PMT_AMT'].values
y_all_log = np.log1p(y_all_raw)

ridge_raw.fit(X_all, y_all_raw)
ridge_log.fit(X_all, y_all_log)

all_train_pred_log = ridge_log.predict(X_all)
all_train_log_residuals = y_all_log - all_train_pred_log
final_smearing_factor = np.mean(np.exp(all_train_log_residuals))

X_full = df_scored[numeric_features + categorical_features]
df_scored['EXPECTED_PMT_PEER'] = df_scored['MODEL_PEER_EXPECTED_PMT']
df_scored['EXPECTED_PMT_RIDGE_RAW'] = np.maximum(0, ridge_raw.predict(X_full))
df_scored['EXPECTED_PMT_RIDGE_LOG_SMEAR'] = np.maximum(0, (np.exp(ridge_log.predict(X_full)) * final_smearing_factor) - 1)

if SELECTED_EXPECTED_PMT_METHOD == 'peer_benchmark':
    df_scored['EXPECTED_PMT_FINAL'] = df_scored['EXPECTED_PMT_PEER']
elif SELECTED_EXPECTED_PMT_METHOD == 'ridge_raw':
    df_scored['EXPECTED_PMT_FINAL'] = df_scored['EXPECTED_PMT_RIDGE_RAW']
else:
    df_scored['EXPECTED_PMT_FINAL'] = df_scored['EXPECTED_PMT_RIDGE_LOG_SMEAR']

# =====================================================================
# 6. CREATE CALIBRATED PRICE-DEVIATION SIGNALS
# =====================================================================
df_scored['PRICE_SIGNAL_ELIGIBLE'] = (df_scored['WORKING_PMT_AMT'] > 0) & (df_scored['EXPECTED_PMT_FINAL'] > 0) & df_scored['EXPECTED_PMT_FINAL'].notna()

df_scored['PAYMENT_RESIDUAL_FINAL'] = df_scored['WORKING_PMT_AMT'] - df_scored['EXPECTED_PMT_FINAL']
df_scored['ABS_PAYMENT_RESIDUAL_FINAL'] = np.abs(df_scored['PAYMENT_RESIDUAL_FINAL'])

df_scored['RELATIVE_SURGE_RATIO_FINAL'] = np.where(
    df_scored['PRICE_SIGNAL_ELIGIBLE'],
    (df_scored['WORKING_PMT_AMT'] - df_scored['EXPECTED_PMT_FINAL']) / df_scored['EXPECTED_PMT_FINAL'],
    np.nan
)
df_scored['PAYMENT_RATIO_TO_EXPECTED_FINAL'] = np.where(
    df_scored['PRICE_SIGNAL_ELIGIBLE'],
    df_scored['WORKING_PMT_AMT'] / df_scored['EXPECTED_PMT_FINAL'],
    np.nan
)
df_scored['EXPECTED_PMT_FINAL_ZERO_FLAG'] = (df_scored['EXPECTED_PMT_FINAL'] == 0).astype(int)

# =====================================================================
# 7. HOLDOUT-CALIBRATED THRESHOLDS & TIERING
# =====================================================================
# Use clean positive holdout records to find P95/P99 of surge ratio
holdout_surge_mask = positive_payment_holdout_mask & df_scored['PRICE_SIGNAL_ELIGIBLE']
holdout_surge_ratios = df_scored.loc[holdout_surge_mask, 'RELATIVE_SURGE_RATIO_FINAL'].dropna()

PAYMENT_SURGE_P95_THRESHOLD = np.percentile(holdout_surge_ratios, 95) if len(holdout_surge_ratios) > 0 else 1.0
PAYMENT_SURGE_P99_THRESHOLD = np.percentile(holdout_surge_ratios, 99) if len(holdout_surge_ratios) > 0 else 2.0

print(f"\nINFO: Holdout-Calibrated Thresholds -> P95: {PAYMENT_SURGE_P95_THRESHOLD:.3f}, P99: {PAYMENT_SURGE_P99_THRESHOLD:.3f}")

def assign_tier(row):
    if not row['PRICE_SIGNAL_ELIGIBLE'] or pd.isna(row['RELATIVE_SURGE_RATIO_FINAL']):
        return 'Not eligible'
    if row['RELATIVE_SURGE_RATIO_FINAL'] >= PAYMENT_SURGE_P99_THRESHOLD:
        return 'High'
    elif row['RELATIVE_SURGE_RATIO_FINAL'] >= PAYMENT_SURGE_P95_THRESHOLD:
        return 'Medium'
    else:
        return 'Standard'

df_scored['PAYMENT_DEVIATION_TIER'] = df_scored.apply(assign_tier, axis=1)

# =====================================================================
# 8. DIAGNOSTICS
# =====================================================================
print("\n=== A. Selected-Model Holdout Calibration ===")
actual_h = df_scored.loc[positive_payment_holdout_mask, 'ORIGINAL_PMT_AMT']
pred_h = df_scored.loc[positive_payment_holdout_mask, 'EXPECTED_PMT_FINAL']
resid_h = df_scored.loc[positive_payment_holdout_mask, 'PAYMENT_RESIDUAL_FINAL']
print(f"Actual mean payment:    ${actual_h.mean():.2f}")
print(f"Predicted mean payment: ${pred_h.mean():.2f}")
print(f"Actual median payment:  ${actual_h.median():.2f}")
print(f"Predicted median payment:${pred_h.median():.2f}")
print(f"Mean residual:          ${resid_h.mean():.2f}")
print(f"Median residual:        ${resid_h.median():.2f}")
print(f"MAE:                    ${mean_absolute_error(actual_h, pred_h):.2f}")
print(f"WAPE:                   {(np.sum(np.abs(actual_h - pred_h)) / np.sum(actual_h))*100:.2f}%")

print("\n=== C. Scenario Signal Diagnostics ===")
scenarios = ['clean', 'extreme_payment_deviation', 'moderate_payment_deviation', 'duplicate_like_billing']
for s in scenarios:
    s_df = df_scored[df_scored['SCENARIO_TYPE'] == s]
    p95_pct = (s_df['RELATIVE_SURGE_RATIO_FINAL'] >= PAYMENT_SURGE_P95_THRESHOLD).mean() * 100
    p99_pct = (s_df['RELATIVE_SURGE_RATIO_FINAL'] >= PAYMENT_SURGE_P99_THRESHOLD).mean() * 100
    elig_pct = s_df['PRICE_SIGNAL_ELIGIBLE'].mean() * 100

    print(f"- {s}: N={len(s_df)}")
    print(f"  Median Working Pmt: ${s_df['WORKING_PMT_AMT'].median():.2f}")
    print(f"  Median Expected Pmt: ${s_df['EXPECTED_PMT_FINAL'].median():.2f}")
    print(f"  Median Residual: ${s_df['PAYMENT_RESIDUAL_FINAL'].median():.2f}")
    print(f"  % Above P95: {p95_pct:.1f}%")
    print(f"  % Above P99: {p99_pct:.1f}%")
    print(f"  % Eligible: {elig_pct:.1f}%\n")

print("=== D. Zero-payment Diagnostics ===")
zero_mask = clean_original_mask & (df['ORIGINAL_PMT_AMT'] == 0)
zero_claims = df_scored[zero_mask]
print(f"Total zero-payment clean claims: {len(zero_claims)}")
print(f"Excluded from PRICE_SIGNAL_ELIGIBLE: {(~zero_claims['PRICE_SIGNAL_ELIGIBLE']).sum()}")
print(f"Undefined price ratio (NaN): {zero_claims['RELATIVE_SURGE_RATIO_FINAL'].isna().sum()}")
# Confirm tier assignment for zero-payment
assert (zero_claims['PAYMENT_DEVIATION_TIER'] == 'Not eligible').all(), "Zero-payment claims incorrectly tiered!"

# =====================================================================
# 10. ASSERTIONS
# =====================================================================
assert (df_scored.loc[clean_train_mask, 'IS_ANOMALY_INJECTED'] == 0).all()
assert (df_scored.loc[clean_train_mask, 'SYNTHETIC_RECORD_CREATED'] == 0).all()
assert len(train_groups.intersection(holdout_groups)) == 0
assert not df_scored.loc[df_scored['EXPECTED_PMT_FINAL'] == 0, 'PRICE_SIGNAL_ELIGIBLE'].any()
assert df_scored.loc[~df_scored['PRICE_SIGNAL_ELIGIBLE'], 'RELATIVE_SURGE_RATIO_FINAL'].isna().all()
assert (df_scored['EXPECTED_PMT_FINAL'] >= 0).all()

# Save output
PARQUET_OUT = 'data/processed/cms_claims_stage3_calibrated.parquet'
CSV_OUT = 'data/processed/cms_claims_stage3_calibrated.csv'

df_scored.to_parquet(PARQUET_OUT, index=False)
df_scored.to_csv(CSV_OUT, index=False)
joblib.dump(ridge_raw, 'models/ridge_expected_payment_raw_pipeline.joblib')
joblib.dump(ridge_log, 'models/ridge_expected_payment_log_smearing_pipeline.joblib')

print(f"\nSUCCESS. Saved artifacts:\n{PARQUET_OUT}\n{CSV_OUT}")