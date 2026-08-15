%%writefile app.py
import os
import pandas as pd
import streamlit as st
from src.llm_copilot import generate_claim_review_memo

# =========================================================================
# 0. PAGE CONFIGURATION
# =========================================================================
st.set_page_config(
    page_title="ClaimSignal | Payment-Integrity Review Queue",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# =========================================================================
# 1. DATA LOADING & HELPERS
# =========================================================================
@st.cache_data
def load_stage6_data():
    return pd.read_parquet("data/processed/cms_claims_stage6_ensemble_scored.parquet")

def get_identifier(row: pd.Series) -> str:
    for col in ['CLM_ID', 'CLAIM_LINE_ID', 'DESYNPUF_ID']:
        if col in row.index and not pd.isna(row[col]):
            return str(row[col])
    return "UNKNOWN"

def get_hcpcs(row: pd.Series) -> str:
    for col in ['HCPCS_CD_1', 'ORIGINAL_HCPCS_CD']:
        if col in row.index and not pd.isna(row[col]):
            return str(row[col])
    return "UNKNOWN"

def format_currency(val) -> str:
    if pd.isna(val): return "$0.00"
    return f"${float(val):,.2f}"

def format_ratio(val) -> str:
    if pd.isna(val): return "N/A"
    return f"{float(val):.2f}x"

df = load_stage6_data()

# =========================================================================
# 2. HEADER & GUARDRAIL STATEMENT
# =========================================================================
st.title("ClaimSignal | Payment-Integrity Review Queue")
st.markdown("##### Human-in-the-loop prioritization prototype for synthetic CMS claims")

st.info(
    "This prototype prioritizes claims for human review using payment deviation, "
    "price-risk, and multivariate anomaly signals. It does not determine fraud, "
    "billing error, medical necessity, overpayment, or payment action.",
    icon="⚠️"
)

st.markdown("---")

# =========================================================================
# 3. REVIEW-QUEUE SUMMARY METRICS
# =========================================================================
flagged_for_review = df[df["STAGE6_AUDIT_TIER"].isin(["elevated", "extreme"])]
extreme_claims = df[df["STAGE6_AUDIT_TIER"] == "extreme"]
elevated_claims = df[df["STAGE6_AUDIT_TIER"] == "elevated"]

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Total Claims", f"{len(df):,}")
with col2:
    st.metric(
        "Claims Flagged for Review",
        f"{len(flagged_for_review):,}",
        help="Elevated or extreme audit-priority tier; analyst review queue only."
    )
    st.caption("Elevated or extreme audit-priority tier; analyst review queue only.")
with col3:
    st.metric("Extreme Priority Claims", f"{len(extreme_claims):,}")
with col4:
    st.metric("Elevated Priority Claims", f"{len(elevated_claims):,}")

st.markdown("---")

# =========================================================================
# 4. FLAGGED CLAIMS REVIEW QUEUE
# =========================================================================
st.subheader("Flagged Claims Review Queue")

# Sort descending by priority score
queue_df = flagged_for_review.sort_values("AUDIT_PRIORITY_SCORE", ascending=False).copy()

# Build display dataframe
display_data = []
for _, row in queue_df.iterrows():
    display_data.append({
        "Claim ID": get_identifier(row),
        "Procedure Code": get_hcpcs(row),
        "Audit Priority Score": round(float(row.get("AUDIT_PRIORITY_SCORE", 0.0)), 2),
        "Audit Tier": str(row.get("STAGE6_AUDIT_TIER", "")).capitalize(),
        "Working Payment": format_currency(row.get("WORKING_PMT_AMT")),
        "Expected Payment": format_currency(row.get("EXPECTED_PMT_FINAL")),
        "Payment Ratio": format_ratio(row.get("PAYMENT_RATIO_TO_EXPECTED_FINAL")),
        "Data Quality Review": "Yes" if row.get("STAGE6_DATA_QUALITY_REVIEW") else "No"
    })

display_df = pd.DataFrame(display_data)

st.dataframe(
    display_df,
    use_container_width=True,
    hide_index=True
)

st.markdown("---")

# =========================================================================
# 5. CLAIM SELECTION
# =========================================================================
st.subheader("Claim Selection")

def build_label(row: pd.Series) -> str:
    cid = get_identifier(row)
    hcpcs = get_hcpcs(row)
    score = round(float(row.get("AUDIT_PRIORITY_SCORE", 0.0)), 1)
    tier = str(row.get("STAGE6_AUDIT_TIER", "")).capitalize()
    return f"Claim {cid} | HCPCS {hcpcs} | Priority {score} | {tier}"

claim_options = {build_label(row): i for i, row in queue_df.iterrows()}
selected_label = st.selectbox("Select a flagged claim to review:", list(claim_options.keys()))

selected_idx = claim_options[selected_label]
selected_claim_row = queue_df.loc[selected_idx]
current_claim_id = get_identifier(selected_claim_row)

# Reset state if a new claim is selected
if "last_claim_id" not in st.session_state or st.session_state.last_claim_id != current_claim_id:
    st.session_state.last_claim_id = current_claim_id
    st.session_state.memo_result = None

# =========================================================================
# 6. SELECTED CLAIM EVIDENCE PANEL
# =========================================================================
col_left, col_right = st.columns(2)

with col_left:
    st.markdown("**Selected Claim Context**")
    context_data = {
        "Claim Identifier": current_claim_id,
        "Procedure Code": get_hcpcs(selected_claim_row),
        "Cluster Segment": str(selected_claim_row.get("STAGE5_CLUSTER_KEY", "N/A")),
        "Working Payment": format_currency(selected_claim_row.get("WORKING_PMT_AMT")),
        "Expected Payment": format_currency(selected_claim_row.get("EXPECTED_PMT_FINAL")),
        "Payment Residual": format_currency(selected_claim_row.get("PAYMENT_RESIDUAL_FINAL")),
        "Payment Ratio to Expected": format_ratio(selected_claim_row.get("PAYMENT_RATIO_TO_EXPECTED_FINAL")),
        "Audit Priority Score": round(float(selected_claim_row.get("AUDIT_PRIORITY_SCORE", 0.0)), 2),
        "Audit Priority Tier": str(selected_claim_row.get("STAGE6_AUDIT_TIER", "")).capitalize()
    }
    st.dataframe(pd.DataFrame(list(context_data.items()), columns=["Attribute", "Value"]), hide_index=True, use_container_width=True)

with col_right:
    st.markdown("**Deterministic Review Signals**")

    pmt_resid = format_currency(selected_claim_row.get("PAYMENT_RESIDUAL_FINAL"))
    pmt_ratio = format_ratio(selected_claim_row.get("PAYMENT_RATIO_TO_EXPECTED_FINAL"))
    stage4_score = selected_claim_row.get("STAGE4_PRICE_RISK_SCORE")
    stage5_score = selected_claim_row.get("STAGE5_IF_RISK_SCORE")
    dq_flag = selected_claim_row.get("STAGE6_DATA_QUALITY_REVIEW")

    signals_data = [
        {"Signal": "Payment Deviation", "Value": f"{pmt_resid} | {pmt_ratio}", "Tier / Status": "Not applicable"},
        {"Signal": "Stage 4 Price Risk", "Value": f"{stage4_score:.2f}" if pd.notna(stage4_score) else "None", "Tier / Status": str(selected_claim_row.get("STAGE4_PRICE_TIER", "N/A")).capitalize()},
        {"Signal": "Stage 5 Multivariate Risk", "Value": f"{stage5_score:.2f}" if pd.notna(stage5_score) else "None", "Tier / Status": str(selected_claim_row.get("STAGE5_IF_TIER", "N/A")).capitalize()},
        {"Signal": "Audit Priority Score", "Value": f"{selected_claim_row.get('AUDIT_PRIORITY_SCORE', 0.0):.2f}", "Tier / Status": str(selected_claim_row.get("STAGE6_AUDIT_TIER", "")).capitalize()},
        {"Signal": "Data Quality Status", "Value": "Routed to review" if dq_flag else "Complete context", "Tier / Status": "Review" if dq_flag else "No routing flag"}
    ]
    st.dataframe(pd.DataFrame(signals_data), hide_index=True, use_container_width=True)

st.markdown("---")

# =========================================================================
# 7. STAGE 7 GROUNDED MEMO ACTION
# =========================================================================
if not os.environ.get("GROQ_API_KEY"):
    st.error("Groq API key is not configured. Set GROQ_API_KEY before generating a memo.")
else:
    if st.button("Generate Grounded Review Memo", type="primary"):
        with st.spinner("Generating and validating grounded review memo..."):
            try:
                result = generate_claim_review_memo(selected_claim_row)
                st.session_state.memo_result = result
            except Exception as e:
                st.error("An error occurred during memo generation. Please try again.")
                with st.expander("Technical details"):
                    st.write(str(e))

    st.caption("Generates one evidence-grounded summary for the selected claim.")

# =========================================================================
# 8. VALIDATION & MEMO PANEL
# =========================================================================
if st.session_state.get("memo_result"):
    result = st.session_state.memo_result

    st.subheader("Grounding & Safety Validation")

    if result.get("validation_passed") is True:
        st.success("✅ Grounding validation passed")

        st.markdown("### Analyst-Facing Review Memo")
        memo = result.get("memo", {})

        # Display Memo Sections
        st.markdown(f"**Review Recommendation:**\n> {memo.get('review_recommendation', '')}")
        st.markdown(f"**Priority Summary:**\n{memo.get('priority_summary', '')}")

        st.markdown("**Evidence:**")
        evidence_list = memo.get('evidence', [])
        if evidence_list:
            ev_df = pd.DataFrame([{
                "Approved Signal": e.get("signal", ""),
                "Finding": e.get("finding", ""),
                "Source Fields": ", ".join(e.get("source_fields", []))
            } for e in evidence_list])
            st.table(ev_df)

        st.markdown("**Recommended Verification Checks:**")
        for check in memo.get('recommended_checks', []):
            st.markdown(f"- {check}")

        st.markdown(f"**Data Quality Status:** {memo.get('data_quality_status', '')}")

        with st.expander("Limitations"):
            st.write(memo.get('limitations', ''))

    else:
        st.error("⚠ Memo withheld: grounding or safety validation failed.")
        with st.expander("Validation details"):
            for err in result.get("validation_errors", []):
                st.write(f"- {err}")

# =========================================================================
# 9. PERSISTENT LIMITATIONS FOOTER
# =========================================================================
st.markdown("---")
st.caption(
    "Synthetic/de-identified CMS claims POC data. Scores prioritize claims for "
    "human review and do not establish billing error, fraud, medical necessity, "
    "overpayment, or payment action."
)