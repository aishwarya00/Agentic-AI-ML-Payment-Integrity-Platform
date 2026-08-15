import os
import sys
import warnings
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

warnings.filterwarnings('ignore')

# =========================================================================
# STEP 0: DYNAMIC COLUMN DETECTION & SAFETY CHECKS
# =========================================================================

def detect_dataframe_columns(df: pd.DataFrame) -> dict:
    """
    Dynamically identifies available CMS column equivalents.
    Returns a dictionary of mapped column names and flags missing essential fields.
    """
    candidates = {
        'bene_id': ['DESY_SORT_KEY', 'BENE_ID', 'BENE_IDENTIFIER', 'BENEFICIARY_ID'],
        'claim_id': ['CLM_ID', 'DESY_SORT_KEY', 'CLAIM_ID', 'CLM_NUM'],
        'provider_npi': ['PRF_PHYSN_NPI_1', 'RNDRNG_NPI', 'PRF_NPI', 'PRVDR_NUM', 'PROVIDER_NPI', 'NPI'],
        'hcpcs_cd': ['HCPCS_CD_1', 'ORIGINAL_HCPCS_CD', 'HCPCS_CD', 'CPT_CD', 'HCPCS'],
        'service_date': ['CLM_FROM_DT', 'SERVICE_DT', 'CLAIM_DATE', 'CLM_FROM_DATE'],
        'payment_amt': ['LINE_NCH_PMT_AMT_1', 'CLM_PMT_AMT', 'PAYMENT_AMOUNT', 'LINE_PAYMENT_AMT', 'ORIGINAL_PMT_AMT'],
        'specialty': ['PROVIDER_SPECIALTY', 'PRF_PHYSN_SPEC', 'SPECIALTY_CD'],
        'pos': ['PLACE_OF_SERVICE', 'POS_CD', 'LINE_PLACE_OF_SRVC_CD'],
        'line_num': ['LINE_NUM', 'CLM_LINE_NUM', 'CLAIM_LINE_NUMBER', 'LINE_INDEX']
    }

    mapped = {}
    for key, choices in candidates.items():
        found = next((col for col in choices if col in df.columns), None)
        mapped[key] = found

    # Essential column validation
    essential = ['claim_id', 'provider_npi', 'hcpcs_cd', 'service_date', 'payment_amt']
    missing_essential = [k for k in essential if mapped[k] is None]

    if missing_essential:
        raise KeyError(
            f"CRITICAL ERROR: Missing essential columns for mapping: {missing_essential}. "
            f"Available columns in dataframe: {list(df.columns)}"
        )

    return mapped


def largest_remainder_allocation(target_total: int, eligible_counts: dict) -> dict:
    """
    Allocates target scenario counts proportionally across clusters using the
    largest-remainder method to ensure exact integer summation and proportional representation.
    """
    total_eligible = sum(eligible_counts.values())
    if total_eligible == 0 or target_total == 0:
        return {k: 0 for k in eligible_counts}

    # Calculate exact proportional shares
    exact_shares = {k: (v / total_eligible) * target_total for k, v in eligible_counts.items()}
    allocated = {k: int(np.floor(v)) for k, v in exact_shares.items()}
    remainders = {k: v - allocated[k] for k, v in exact_shares.items()}

    # Distribute remainders
    remaining_needed = target_total - sum(allocated.values())
    sorted_keys = sorted(remainders.keys(), key=lambda k: remainders[k], reverse=True)

    for i in range(remaining_needed):
        allocated[sorted_keys[i % len(sorted_keys)]] += 1

    # Cap allocations by available eligible candidates in each cluster
    excess = 0
    for k in allocated:
        if allocated[k] > eligible_counts[k]:
            excess += allocated[k] - eligible_counts[k]
            allocated[k] = eligible_counts[k]

    # Reallocate excess if possible to other clusters with remaining capacity
    if excess > 0:
        for k in sorted_keys:
            can_take = eligible_counts[k] - allocated[k]
            add = min(excess, can_take)
            allocated[k] += add
            excess -= add
            if excess == 0:
                break

    return allocated


def run_pipeline(merged_cms_df: pd.DataFrame) -> pd.DataFrame:
    print("=" * 80)
    print("STARTING SYNTHETIC PAYMENT INTEGRITY DATA PROCESSING & SCENARIO INJECTION")
    print("=" * 80)

    # Detect mapped columns
    cols = detect_dataframe_columns(merged_cms_df)

    # Reset index safely
    df = merged_cms_df.copy().reset_index(drop=True)

    # =========================================================================
    # TASK 1: PRESERVE IMMUTABLE ORIGINAL & WORKING COLUMNS
    # =========================================================================
    print("\n[Task 1] Preserving immutable original baseline columns & working fields...")

    df['ORIGINAL_PMT_AMT'] = pd.to_numeric(df[cols['payment_amt']], errors='coerce').fillna(0.0)
    df['ORIGINAL_HCPCS_CD'] = df[cols['hcpcs_cd']].astype(str).str.strip()
    df['ORIGINAL_FROM_DT'] = pd.to_datetime(df[cols['service_date']], errors='coerce')
    df['ORIGINAL_PROVIDER_NPI'] = df[cols['provider_npi']].astype(str).str.strip()

    # Working fields for scenario modifications
    df['WORKING_PMT_AMT'] = df['ORIGINAL_PMT_AMT'].copy()
    df['WORKING_HCPCS_CD'] = df['ORIGINAL_HCPCS_CD'].copy()
    df['WORKING_FROM_DT'] = df['ORIGINAL_FROM_DT'].copy()
    df['WORKING_PROVIDER_NPI'] = df['ORIGINAL_PROVIDER_NPI'].copy()

    # =========================================================================
    # TASK 2: LINE-LEVEL IDENTIFIERS, DATA QUALITY & SPARSE PROVIDERS
    # =========================================================================
    print("[Task 2] Constructing line-level identifiers, quality flags & sparse-provider indicators...")

    # Unique Line Identifier
    if cols['line_num'] is not None and cols['line_num'] in df.columns:
        df['CLAIM_LINE_ID'] = df[cols['claim_id']].astype(str) + "_" + df[cols['line_num']].astype(str)
    else:
        df['CLAIM_LINE_ID'] = df[cols['claim_id']].astype(str) + "_L" + (df.index + 1).astype(str)

    # Handle duplicates if present in raw IDs
    if df['CLAIM_LINE_ID'].duplicated().any():
        df['CLAIM_LINE_ID'] = df['CLAIM_LINE_ID'] + "_IDX" + df.index.astype(str)

    # Data Quality Flags
    df['HAS_VALID_HCPCS'] = df['ORIGINAL_HCPCS_CD'].notna() & (df['ORIGINAL_HCPCS_CD'] != '') & (df['ORIGINAL_HCPCS_CD'] != 'nan')
    df['HAS_VALID_PROVIDER'] = df['ORIGINAL_PROVIDER_NPI'].notna() & (df['ORIGINAL_PROVIDER_NPI'] != '') & (df['ORIGINAL_PROVIDER_NPI'] != 'nan')
    df['HAS_VALID_DATE'] = df['ORIGINAL_FROM_DT'].notna()
    df['HAS_POSITIVE_PAYMENT'] = df['ORIGINAL_PMT_AMT'] > 0

    # Clean Analysis Eligibility Flag
    df['CLEAN_ANALYSIS_ELIGIBLE'] = (
        df['HAS_VALID_HCPCS'] &
        df['HAS_VALID_PROVIDER'] &
        df['HAS_VALID_DATE'] &
        (df['ORIGINAL_PMT_AMT'] >= 0)
    )

    # Sparse Provider Identification (< 3 claim lines)
    # Low-volume (sparse) providers have limited individual behavioral history and therefore rely more heavily on peer benchmarks.
    prov_line_counts = df.groupby('ORIGINAL_PROVIDER_NPI')['CLAIM_LINE_ID'].transform('count')
    df['IS_SPARSE_PROVIDER'] = prov_line_counts < 3

    print("\n--- DATA QUALITY & POPULATION SUMMARY ---")
    print(f"Total Claim Lines:           {len(df):,}")
    print(f"Valid HCPCS Lines:           {df['HAS_VALID_HCPCS'].sum():,}")
    print(f"Valid Provider Lines:        {df['HAS_VALID_PROVIDER'].sum():,}")
    print(f"Positive Payment Lines:      {df['HAS_POSITIVE_PAYMENT'].sum():,}")
    print(f"Zero Payment Lines:          {(df['ORIGINAL_PMT_AMT'] == 0).sum():,}")
    print(f"Missing Date Lines:          {(~df['HAS_VALID_DATE']).sum():,}")
    print(f"Duplicate CLAIM_LINE_IDs:    {df['CLAIM_LINE_ID'].duplicated().sum():,}")
    print(f"Sparse Provider Lines (<3):  {df['IS_SPARSE_PROVIDER'].sum():,} ({df['IS_SPARSE_PROVIDER'].mean()*100:.1f}%)")

    # =========================================================================
    # TASK 1 & 2 (CONT.): PROVIDER FEATURES & FIXED K=3 CLUSTERING
    # =========================================================================
    print("\n[Task 1 & 2] Building provider features & fitting K-Means (k=3)...")

    clean_records = df[df['CLEAN_ANALYSIS_ELIGIBLE']]
    payment_p75 = clean_records['ORIGINAL_PMT_AMT'].quantile(0.75)

    prov_features = clean_records.groupby('ORIGINAL_PROVIDER_NPI').agg(
        prov_median_pmt=('ORIGINAL_PMT_AMT', 'median'),
        prov_mean_pmt=('ORIGINAL_PMT_AMT', 'mean'),
        prov_high_pmt_share=('ORIGINAL_PMT_AMT', lambda x: (x > payment_p75).mean()),
        prov_total_lines=('CLAIM_LINE_ID', 'count'),
        prov_hcpcs_diversity=('ORIGINAL_HCPCS_CD', 'nunique')
    ).reset_index()

    # Log transformations for heavily skewed positive features
    prov_features['log_prov_mean_pmt'] = np.log1p(prov_features['prov_mean_pmt'].clip(lower=0))
    prov_features['log_prov_total_lines'] = np.log1p(prov_features['prov_total_lines'].clip(lower=0))
    prov_features['log_prov_hcpcs_diversity'] = np.log1p(prov_features['prov_hcpcs_diversity'].clip(lower=0))

    feature_cols_clustering = ['log_prov_mean_pmt', 'log_prov_total_lines', 'log_prov_hcpcs_diversity', 'prov_high_pmt_share']
    for c in feature_cols_clustering:
        prov_features[c] = prov_features[c].replace([np.inf, -np.inf], np.nan).fillna(prov_features[c].median())

    X_prov = prov_features[feature_cols_clustering].values
    scaler = StandardScaler()
    X_prov_scaled = scaler.fit_transform(X_prov)

    # Calculate Silhouette Scores for k=2..6 for documentation
    print("Calculating silhouette scores for documentation:")
    for k in range(2, 7):
        if len(prov_features) > k:
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            score = silhouette_score(X_prov_scaled, km.fit_predict(X_prov_scaled))
            print(f"  k={k}: Silhouette Score = {score:.4f}")

    # Explicitly select k=3 for business interpretability and adequate peer-group support
    print("INFO: Selected k=3 provider billing-behavior segments for interpretability and adequate peer-group support.")

    kmeans_3 = KMeans(n_clusters=3, random_state=42, n_init=10)
    prov_features['kmeans_cluster'] = kmeans_3.fit_predict(X_prov_scaled)

    # Map cluster assignments back to claim lines
    cluster_map = prov_features.set_index('ORIGINAL_PROVIDER_NPI')['kmeans_cluster'].to_dict()
    df['kmeans_cluster'] = df['ORIGINAL_PROVIDER_NPI'].map(cluster_map).fillna(0).astype(int)

    # Generate Cluster Profile Table (Provider Billing-Behavior Segments)
    profile_rows = []
    for cid in sorted(df['kmeans_cluster'].unique()):
        c_lines = df[df['kmeans_cluster'] == cid]
        c_provs = c_lines['ORIGINAL_PROVIDER_NPI'].unique()
        sparse_prov_cnt = (c_lines.groupby('ORIGINAL_PROVIDER_NPI')['CLAIM_LINE_ID'].count() < 3).sum()
        total_prov_cnt = len(c_provs)

        profile_rows.append({
            'Cluster ID': f"Segment {cid}",
            'Unique Provider Count': total_prov_cnt,
            'Sparse Provider Count': sparse_prov_cnt,
            'Sparse Provider Share (%)': f"{(sparse_prov_cnt / total_prov_cnt * 100):.1f}%" if total_prov_cnt > 0 else "0.0%",
            'Claim-Line Count': len(c_lines),
            'Median Payment ($)': round(c_lines['ORIGINAL_PMT_AMT'].median(), 2),
            'Mean Payment ($)': round(c_lines['ORIGINAL_PMT_AMT'].mean(), 2),
            'HCPCS Diversity (Avg)': round(c_lines.groupby('ORIGINAL_PROVIDER_NPI')['ORIGINAL_HCPCS_CD'].nunique().mean(), 1),
            'Total Volume Share (%)': f"{(len(c_lines)/len(df))*100:.1f}%"
        })
    provider_cluster_profile_df = pd.DataFrame(profile_rows)
    print("\n=== PROVIDER BILLING-BEHAVIOR SEGMENTS PROFILE ===")
    print(provider_cluster_profile_df.to_string(index=False))

    # =========================================================================
    # TASK 3: REFINED HIERARCHICAL EXPECTED-PAYMENT BENCHMARKS
    # =========================================================================
    print("\n[Task 3] Calculating hierarchical peer-group payment benchmarks...")

    # Calculate statistics on clean original positive-payment claims
    clean_pos_df = df[df['CLEAN_ANALYSIS_ELIGIBLE'] & df['HAS_POSITIVE_PAYMENT']].copy()

    # 1. Cluster + HCPCS statistics
    cl_hcpcs = clean_pos_df.groupby(['kmeans_cluster', 'ORIGINAL_HCPCS_CD'])['ORIGINAL_PMT_AMT'].agg(
        cluster_hcpcs_count='count',
        cluster_hcpcs_median_payment='median',
        cluster_hcpcs_mean_payment='mean',
        cluster_hcpcs_p95_pmt=lambda x: x.quantile(0.95),
        cluster_hcpcs_iqr=lambda x: x.quantile(0.75) - x.quantile(0.25)
    ).reset_index()

    # 2. HCPCS Global statistics
    gl_hcpcs = clean_pos_df.groupby('ORIGINAL_HCPCS_CD')['ORIGINAL_PMT_AMT'].agg(
        hcpcs_count='count',
        hcpcs_median_payment='median'
    ).reset_index()

    # 3. Cluster Global statistics
    cl_gl = clean_pos_df.groupby('kmeans_cluster')['ORIGINAL_PMT_AMT'].agg(
        cluster_count='count',
        cluster_median_payment='median'
    ).reset_index()

    # 4. Global Median Payment
    global_median_payment = clean_pos_df['ORIGINAL_PMT_AMT'].median()

    # Merge statistics onto main dataframe
    df = df.merge(cl_hcpcs, on=['kmeans_cluster', 'ORIGINAL_HCPCS_CD'], how='left')
    df = df.merge(gl_hcpcs, on='ORIGINAL_HCPCS_CD', how='left')
    df = df.merge(cl_gl, on='kmeans_cluster', how='left')

    # Apply Refined Fallback Hierarchy
    def assign_peer_benchmark(row):
        if row['cluster_hcpcs_count'] >= 5:
            return pd.Series([
                row['cluster_hcpcs_median_payment'],
                'cluster_hcpcs',
                row['cluster_hcpcs_count'],
                row['cluster_hcpcs_p95_pmt'],
                row['cluster_hcpcs_iqr']
            ])
        elif row['hcpcs_count'] >= 10:
            return pd.Series([
                row['hcpcs_median_payment'],
                'hcpcs_only',
                row['hcpcs_count'],
                np.nan,
                np.nan
            ])
        elif row['cluster_count'] >= 30:
            return pd.Series([
                row['cluster_median_payment'],
                'cluster_only',
                row['cluster_count'],
                np.nan,
                np.nan
            ])
        else:
            return pd.Series([
                global_median_payment,
                'global_fallback',
                len(clean_pos_df),
                np.nan,
                np.nan
            ])

    benchmark_cols = ['PEER_EXPECTED_PMT', 'PEER_GROUP_LEVEL', 'PEER_GROUP_COUNT', 'PEER_P95_PMT', 'PEER_PAYMENT_IQR']
    df[benchmark_cols] = df.apply(assign_peer_benchmark, axis=1)

    peer_summary_df = df['PEER_GROUP_LEVEL'].value_counts().reset_index()
    peer_summary_df.columns = ['Peer Group Level', 'Claim Lines']
    peer_summary_df['Percentage (%)'] = (peer_summary_df['Claim Lines'] / len(df) * 100).round(2)
    print("\n=== PEER BENCHMARK HIERARCHY COVERAGE ===")
    print(peer_summary_df.to_string(index=False))

    # =========================================================================
    # TASK 4 & 6: SCENARIO ELIGIBILITY & FEASIBLE ALLOCATION
    # =========================================================================
    print("\n[Task 4 & 6] Evaluating scenario eligibility flags & checking weights...")

    bene_col = cols['bene_id'] if cols['bene_id'] in df.columns else 'ORIGINAL_PROVIDER_NPI'

    df['ELIGIBLE_PRICE_DEVIATION'] = (
        (df['ORIGINAL_PMT_AMT'] > 0) &
        (df['PEER_EXPECTED_PMT'] > 0) &
        df['CLEAN_ANALYSIS_ELIGIBLE'] &
        df['PEER_GROUP_LEVEL'].notna()
    )

    df['ELIGIBLE_MODERATE_PAYMENT'] = (
        (df['ORIGINAL_PMT_AMT'] > 0) &
        (df['PEER_EXPECTED_PMT'] > 0) &
        df['CLEAN_ANALYSIS_ELIGIBLE'] &
        df['PEER_GROUP_LEVEL'].notna()
    )

    df['ELIGIBLE_DUPLICATE'] = (
        df['CLEAN_ANALYSIS_ELIGIBLE'] &
        df[bene_col].notna()
    )

    # Validate weights for 3 feasible scenarios
    scenario_weights = {
        "extreme_payment_deviation": 0.45,
        "duplicate_like_billing": 0.30,
        "moderate_payment_deviation": 0.25
    }
    assert np.isclose(sum(scenario_weights.values()), 1.0), "Scenario allocation weights must sum to 1.0"

    # =========================================================================
    # TASK 5, 7, 8, 9: PROPORTIONAL STRATIFIED ANOMALY INJECTION
    # =========================================================================
    print("\n[Task 5, 7, 8, 9] Executing proportional stratified anomaly injection engine...")

    rng = np.random.default_rng(42)

    df['IS_ANOMALY_INJECTED'] = 0
    df['SCENARIO_TYPE'] = "clean"
    df['INJECTION_MULTIPLIER'] = np.nan
    df['INJECTION_SEED'] = 42
    df['SOURCE_CLAIM_LINE_ID'] = df['CLAIM_LINE_ID']
    df['SYNTHETIC_RECORD_CREATED'] = 0

    target_rate = 0.035
    total_target_injections = int(np.round(len(df) * target_rate))

    allocated_indices = set()
    synthetic_duplicate_rows = []

    # Proportional Allocation per Scenario across Clusters
    for scenario_name, weight in scenario_weights.items():
        scenario_target_count = int(np.round(total_target_injections * weight))

        if scenario_name == 'extreme_payment_deviation':
            eligible_mask = df['ELIGIBLE_PRICE_DEVIATION']
        elif scenario_name == 'moderate_payment_deviation':
            eligible_mask = df['ELIGIBLE_MODERATE_PAYMENT']
        elif scenario_name == 'duplicate_like_billing':
            eligible_mask = df['ELIGIBLE_DUPLICATE']

        # Determine eligible count per cluster among unallocated original claim lines
        candidate_pool = df[eligible_mask & (~df.index.isin(allocated_indices))]
        cluster_eligible_counts = candidate_pool.groupby('kmeans_cluster').size().to_dict()

        # Ensure all clusters are present in dictionary
        for cid in df['kmeans_cluster'].unique():
            if cid not in cluster_eligible_counts:
                cluster_eligible_counts[cid] = 0

        # Calculate exact proportional allocation per cluster
        cluster_allocations = largest_remainder_allocation(scenario_target_count, cluster_eligible_counts)

        # Sample within each cluster
        selected_indices = []
        for cid, alloc_count in cluster_allocations.items():
            if alloc_count > 0:
                c_candidates = candidate_pool[candidate_pool['kmeans_cluster'] == cid].index.values
                sampled = rng.choice(c_candidates, size=alloc_count, replace=False).tolist()
                selected_indices.extend(sampled)

        # Track allocated claim lines for ALL scenarios to ensure non-overlapping selections
        allocated_indices.update(selected_indices)

        # Apply specific scenario modifications
        if scenario_name == 'extreme_payment_deviation':
            multipliers = rng.uniform(2.5, 4.0, size=len(selected_indices))
            for idx, mult in zip(selected_indices, multipliers):
                df.loc[idx, 'WORKING_PMT_AMT'] = df.loc[idx, 'PEER_EXPECTED_PMT'] * mult
                df.loc[idx, 'IS_ANOMALY_INJECTED'] = 1
                df.loc[idx, 'SCENARIO_TYPE'] = 'extreme_payment_deviation'
                df.loc[idx, 'INJECTION_MULTIPLIER'] = mult

        elif scenario_name == 'moderate_payment_deviation':
            multipliers = rng.uniform(1.4, 1.9, size=len(selected_indices))
            for idx, mult in zip(selected_indices, multipliers):
                df.loc[idx, 'WORKING_PMT_AMT'] = df.loc[idx, 'PEER_EXPECTED_PMT'] * mult
                df.loc[idx, 'IS_ANOMALY_INJECTED'] = 1
                df.loc[idx, 'SCENARIO_TYPE'] = 'moderate_payment_deviation'
                df.loc[idx, 'INJECTION_MULTIPLIER'] = mult

        elif scenario_name == 'duplicate_like_billing':
            for dup_count, idx in enumerate(selected_indices):
                source_row = df.loc[idx].copy()
                dup_row = source_row.copy()

                # New unique claim line identifier
                dup_row['CLAIM_LINE_ID'] = f"{source_row['CLAIM_LINE_ID']}_DUP{dup_count+1}"
                dup_row['SOURCE_CLAIM_LINE_ID'] = source_row['CLAIM_LINE_ID']

                # Service date offset (0 to 7 days)
                days_offset = rng.integers(0, 8)
                dup_row['WORKING_FROM_DT'] = source_row['ORIGINAL_FROM_DT'] + pd.Timedelta(days=int(days_offset))

                dup_row['IS_ANOMALY_INJECTED'] = 1
                dup_row['SCENARIO_TYPE'] = 'duplicate_like_billing'
                dup_row['SYNTHETIC_RECORD_CREATED'] = 1

                synthetic_duplicate_rows.append(dup_row)

    # Append synthetic duplicate rows to dataset
    if synthetic_duplicate_rows:
        synth_df = pd.DataFrame(synthetic_duplicate_rows)
        df = pd.concat([df, synth_df], ignore_index=True)

    print(f"INFO: Successfully injected synthetic scenarios across {df['IS_ANOMALY_INJECTED'].sum():,} total records.")

    # =========================================================================
    # TASK 10: RECALCULATE POST-INJECTION ANALYTICAL SIGNALS
    # =========================================================================
    print("\n[Task 10] Calculating post-injection analytical signals and ratios...")

    df['PAYMENT_RESIDUAL'] = df['WORKING_PMT_AMT'] - df['PEER_EXPECTED_PMT']
    df['PAYMENT_RATIO_TO_PEER'] = np.where(df['PEER_EXPECTED_PMT'] > 0, df['WORKING_PMT_AMT'] / df['PEER_EXPECTED_PMT'], np.nan)
    df['PAYMENT_DEVIATION_PERCENT'] = np.where(df['PEER_EXPECTED_PMT'] > 0, ((df['WORKING_PMT_AMT'] - df['PEER_EXPECTED_PMT']) / df['PEER_EXPECTED_PMT']) * 100.0, np.nan)

    # Candidate duplicate flag (beneficiary + provider + HCPCS + date window)
    dup_keys = [bene_col, 'WORKING_PROVIDER_NPI', 'WORKING_HCPCS_CD']
    df['DUPLICATE_CANDIDATE_FLAG'] = df.duplicated(subset=dup_keys, keep=False).astype(int)

    # Provider-HCPCS concentration feature
    prov_hcpcs_counts = df.groupby(['WORKING_PROVIDER_NPI', 'WORKING_HCPCS_CD'])['CLAIM_LINE_ID'].transform('count')
    prov_tot_counts = df.groupby('WORKING_PROVIDER_NPI')['CLAIM_LINE_ID'].transform('count')
    df['PROVIDER_HCPCS_CONCENTRATION'] = prov_hcpcs_counts / prov_tot_counts

    # =========================================================================
    # TASK 11: VALIDATION REPORT & ASSERTIONS
    # =========================================================================
    print("\n[Task 11] Running comprehensive validation report & safety assertions...")

    orig_len = len(merged_cms_df)
    final_len = len(df)
    total_inj = df['IS_ANOMALY_INJECTED'].sum()

    print(f"\n1. Population Count: Original={orig_len:,}, Final={final_len:,} (Synthetic Duplicates Added={final_len - orig_len:,})")
    print(f"2. Total Anomaly Rate vs Original Population: {total_inj / orig_len * 100:.2f}% ({total_inj:,} / {orig_len:,})")
    print(f"   Total Anomaly Rate vs Final Population:    {total_inj / final_len * 100:.2f}% ({total_inj:,} / {final_len:,})")

    # 3. Injection Count by Scenario
    print("\n3. Injection Count by Scenario:")
    scenario_report = df['SCENARIO_TYPE'].value_counts().reset_index()
    scenario_report.columns = ['Scenario Type', 'Count']
    print(scenario_report.to_string(index=False))

    # 4. Proportional Injection Rate by Cluster
    print("\n4. Cluster Injection Validation Table:")
    cluster_val = []
    for cid in sorted(df['kmeans_cluster'].unique()):
        sub_all = df[df['kmeans_cluster'] == cid]
        sub_orig = df[(df['kmeans_cluster'] == cid) & (df['SYNTHETIC_RECORD_CREATED'] == 0)]

        ext_cnt = (sub_all['SCENARIO_TYPE'] == 'extreme_payment_deviation').sum()
        mod_cnt = (sub_all['SCENARIO_TYPE'] == 'moderate_payment_deviation').sum()
        dup_cnt = (sub_all['SCENARIO_TYPE'] == 'duplicate_like_billing').sum()
        tot_inj_c = sub_all['IS_ANOMALY_INJECTED'].sum()

        cluster_val.append({
            'Cluster ID': f"Segment {cid}",
            'Total Records': len(sub_all),
            'Eligible Records': sub_orig['ELIGIBLE_PRICE_DEVIATION'].sum(),
            'Injected Records': tot_inj_c,
            'Injection Rate (%)': f"{(tot_inj_c / len(sub_all) * 100):.2f}%",
            'Extreme Price': ext_cnt,
            'Moderate Price': mod_cnt,
            'Duplicate': dup_cnt
        })
    cluster_val_df = pd.DataFrame(cluster_val)
    print(cluster_val_df.to_string(index=False))

    # 5. Injection Rate by Peer Group Level
    print("\n5. Injection Rate by Peer Group Level:")
    peer_val = df.groupby('PEER_GROUP_LEVEL')['IS_ANOMALY_INJECTED'].agg(Total='count', Injected='sum').reset_index()
    peer_val['Injection Rate (%)'] = (peer_val['Injected'] / peer_val['Total'] * 100).round(2)
    print(peer_val.to_string(index=False))

    # 6. Median Original vs Working Payment by Scenario
    print("\n6. Median Original vs Working Payment by Scenario:")
    pmt_val = df.groupby('SCENARIO_TYPE')[['ORIGINAL_PMT_AMT', 'WORKING_PMT_AMT']].median().reset_index()
    print(pmt_val.to_string(index=False))

    # Zero Payment & Overlap Validation Checks
    zero_pmt_injected = df[df['SCENARIO_TYPE'].isin(['extreme_payment_deviation', 'moderate_payment_deviation']) & (df['ORIGINAL_PMT_AMT'] == 0)].shape[0]
    overlapping_scenarios = (df.groupby('SOURCE_CLAIM_LINE_ID')['SCENARIO_TYPE'].transform(lambda x: (x != 'clean').sum()) > 1).sum()
    duplicate_line_ids = df['CLAIM_LINE_ID'].duplicated().sum()
    global_fallback_pct = (df['PEER_GROUP_LEVEL'] == 'global_fallback').mean() * 100

    print(f"\n7. Zero-Payment Claims Injected into Price Scenarios: {zero_pmt_injected} (Expected: 0)")
    print(f"8. Original Lines Assigned Multiple Scenarios:         {overlapping_scenarios} (Expected: 0)")
    print(f"9. Duplicate CLAIM_LINE_IDs in Final Dataset:            {duplicate_line_ids} (Expected: 0)")
    print(f"10. Global-Fallback Benchmark Coverage Percentage:       {global_fallback_pct:.2f}%")

    # --- ASSERTIONS ---
    assert zero_pmt_injected == 0, "Assertion Error: Zero-payment claim was selected for price deviation!"
    assert overlapping_scenarios == 0, "Assertion Error: Claim line assigned multiple scenarios!"
    assert duplicate_line_ids == 0, "Assertion Error: Duplicate CLAIM_LINE_ID found!"
    assert set(df['SCENARIO_TYPE'].unique()).issubset({
        'clean', 'extreme_payment_deviation', 'duplicate_like_billing', 'moderate_payment_deviation'
    }), "Assertion Error: Unrecognized scenario type present!"

    print("\n>>> ALL PIPELINE VALIDATION ASSERTIONS PASSED SUCCESSFULLY! <<<")

    # =========================================================================
    # TASK 12: SAVE REQUIRED OUTPUTS
    # =========================================================================
    print("\n[Task 12] Exporting final injected analytical dataset...")

    os.makedirs("data/processed", exist_ok=True)

    parquet_path = "data/processed/cms_claims_injected.parquet"
    csv_path = "data/processed/cms_claims_injected.csv"

    # String conversion for datetime columns before parquet export
    export_df = df.copy()
    for dt_col in export_df.select_dtypes(include=['datetime64', 'datetime']).columns:
        export_df[dt_col] = export_df[dt_col].astype(str)

    export_df.to_parquet(parquet_path, index=False)
    export_df.to_csv(csv_path, index=False)

    print(f"\nSaved Output Dataset:")
    print(f"  - Parquet: {parquet_path}")
    print(f"  - CSV:     {csv_path}")

    return df


# Demonstration execution harness if run directly
if __name__ == "__main__":
    if 'merged_cms_df' not in locals():
        print("Creating synthetic CMS sample dataset for pipeline execution...")
        np.random.seed(42)
        n_samples = 3000

        mock_df = pd.DataFrame({
            'DESY_SORT_KEY': [f"BENE_{np.random.randint(1000, 2000)}" for _ in range(n_samples)],
            'CLM_ID': [f"CLM_{10000 + i}" for i in range(n_samples)],
            'LINE_NUM': np.random.randint(1, 4, size=n_samples),
            'PRF_PHYSN_NPI_1': [f"NPI_{np.random.randint(100, 250)}" for _ in range(n_samples)],
            'HCPCS_CD_1': np.random.choice(['99213', '99214', '71045', '93000', '36415'], size=n_samples),
            'CLM_FROM_DT': pd.date_range(start='2025-01-01', periods=n_samples, freq='h'),
            'LINE_NCH_PMT_AMT_1': np.random.choice([0.0, 45.0, 85.0, 150.0, 320.0], size=n_samples, p=[0.05, 0.40, 0.30, 0.20, 0.05]),
            'PROVIDER_SPECIALTY': np.random.choice(['Radiology', 'Cardiology', 'General Practice'], size=n_samples)
        })
        merged_cms_df = mock_df

    processed_df = run_pipeline(merged_cms_df)