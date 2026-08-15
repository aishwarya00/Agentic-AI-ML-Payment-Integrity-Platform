import os
import json
import pandas as pd
# from src.llm_copilot import generate_claim_review_memo

def run_stage7_demo():
    print("=" * 80)
    print("=== STAGE 7: REVIEW ENGINE DEMO (SINGLE SELECTED CLAIM) ===")
    print("=" * 80)

    data_path = "data/processed/cms_claims_stage6_ensemble_scored.parquet"

    if not os.path.exists(data_path):
        print(f"Error: Dataset not found at '{data_path}'. Check your workspace path.")
        return

    # Load Stage 6 scored dataset
    df = pd.read_parquet(data_path)

    # Filter for real elevated or extreme claims with complete data context
    demo_candidates = df[
        (df["STAGE6_AUDIT_TIER"].isin(["elevated", "extreme"])) &
        (df["STAGE6_DATA_QUALITY_REVIEW"] == False)
    ].sort_values("AUDIT_PRIORITY_SCORE", ascending=False)

    if demo_candidates.empty:
        print("Warning: No elevated/extreme claims found without data quality flags. Falling back to top score.")
        demo_claim = df.sort_values("AUDIT_PRIORITY_SCORE", ascending=False).iloc[0]
    else:
        demo_claim = demo_candidates.iloc[0]

    claim_id = demo_claim.get("CLM_ID", demo_claim.get("CLAIM_LINE_ID", "UNKNOWN"))
    print(f"\nSelected High-Priority Claim ID: {claim_id}\n")

    # Generate memo using core reusable function
    result = generate_claim_review_memo(demo_claim)

    print("1. STRUCTURED EVIDENCE PACKET:")
    print(json.dumps(result["evidence_packet"], indent=2))

    print("\n2. VALIDATION & GROUNDING STATUS:")
    if result["validation_passed"]:
        print(" [PASS] Output complies with schema, grounding rules, and safety filters.")
    else:
        print(" [FAIL] Grounding/Validation Errors:")
        for err in result["validation_errors"]:
            print(f"  - {err}")

    print("\n3. FINAL ANALYST-FACING REVIEW MEMO:")
    if result["validation_passed"] and result["memo"]:
        print(json.dumps(result["memo"], indent=2))
    else:
        print("Memo generation requires analyst review because output grounding validation did not pass.")

if __name__ == "__main__":
    run_stage7_demo()