import os
import json
from typing import TypedDict, Dict, Any
from google.colab import userdata
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END
import logging
import pandas as pd
from typing import List, Dict, Any, Tuple, Optional
from pydantic import BaseModel, Field, ValidationError

from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.exceptions import OutputParserException

# 1. Initialize Groq API Key from Environment
try:
    groq_key = userdata.get('GROQ_API_KEY')
    os.environ["GROQ_API_KEY"] = groq_key
    print("Groq API Key loaded successfully!")
except Exception:
    # Prompt user if secret is not set in Colab sidebar
    os.environ["GROQ_API_KEY"] = userdata.get("GROQ_API_KEY")

# Initialize Llama 3.3 70B via Groq
llm = ChatGroq(
    model_name="llama-3.3-70b-versatile",
    temperature=0.1 # Low temperature for reliable audit reasoning
)


"""
Stage 7: ClaimSignal Evidence-Grounded Review Engine
Converts Stage 4, Stage 5, and Stage 6 model evidence into a concise,
grounded, human-reviewable payment-integrity memo using Llama-3.3-70b-versatile via Groq.
"""



# Configure logging
logger = logging.getLogger(__name__)

# =========================================================================
# 1. CONSTANTS & APPROVED SIGNALS
# =========================================================================

APPROVED_SIGNALS = [
    "Audit Priority Score",
    "Payment Deviation",
    "Stage 4 Price Risk",
    "Stage 5 Multivariate Risk",
    "Data Quality Status"
]

PROHIBITED_TERMS = [
    "fraud", "fraudulent", "improper", "overcharged", "deny",
    "denial", "reject", "illegal", "guilty", "overpayment confirmed",
    "auto-clear", "payment action"
]

# =========================================================================
# 2. PYDANTIC SCHEMAS (STRUCTURED OUTPUT)
# =========================================================================

class EvidenceItem(BaseModel):
    signal: str = Field(
        description="Must be exactly one approved human-readable signal name."
    )
    finding: str = Field(
        description="Concise factual finding using only supplied evidence values."
    )
    source_fields: List[str] = Field(
        description="Exact CLAIM_EVIDENCE field names supporting the finding."
    )

class ClaimReviewMemo(BaseModel):
    review_recommendation: str = Field(
        description="Must exactly match required_review_recommendation from the evidence packet."
    )
    priority_summary: str = Field(
        description="One or two concise factual sentences."
    )
    evidence: List[EvidenceItem] = Field(
        description="Two to four evidence items in descending importance."
    )
    recommended_checks: List[str] = Field(
        description="Two to four human verification checks only."
    )
    data_quality_status: str = Field(
        description="State whether data context is complete or needs review."
    )
    limitations: str = Field(
        description="Mandatory POC and human-review limitation statement."
    )


# =========================================================================
# 3. SYSTEM PROMPT
# =========================================================================

SYSTEM_PROMPT = """You are ClaimSignal Review Assistant, assisting a healthcare
payment-integrity analyst.

Your only task is to convert supplied CLAIM_EVIDENCE into a concise,
auditable analyst review memo.

You do not make payment decisions and do not determine fraud, billing error,
medical necessity, coding correctness, contract compliance, overpayment, or
payment action.

Rules:
1. Use only facts explicitly present in CLAIM_EVIDENCE.
2. Do not calculate, estimate, infer, or invent missing values.
3. Set review_recommendation exactly equal to required_review_recommendation.
4. Use only these exact evidence signal names:
   - Audit Priority Score
   - Payment Deviation
   - Stage 4 Price Risk
   - Stage 5 Multivariate Risk
   - Data Quality Status
5. Use machine-readable evidence-packet field names only in source_fields.
6. If policy_text_available is false, do not name or claim violation of any
   policy, contract, fee schedule, or documentation requirement.
7. Recommend verification steps only; never payment actions.
8. If data_quality_review is true, prioritize data-quality review and do not
   recommend payment-integrity escalation.
9. Do not use prohibited or accusatory language.
10. Do not provide chain-of-thought, hidden reasoning, or internal
    step-by-step reasoning.
11. Return only the requested Pydantic structured output."""


# =========================================================================
# 4. EVIDENCE PACKET BUILDER & DETERMINISTIC ROUTING
# =========================================================================

def _safe_get(row: pd.Series, col: str, default: Any = None) -> Any:
    """Safely retrieve a value from a pandas Series without NaN issues."""
    if col not in row.index or pd.isna(row[col]):
        return default
    val = row[col]
    if hasattr(val, "item"):
        return val.item()
    return val

def get_required_recommendation(packet: dict) -> str:
    """Deterministic routing rules for the review recommendation."""
    if packet.get("data_quality_review") is True or packet.get("missing_hcpcs_flag") is True:
        return "Route to data-quality review before payment-integrity analysis."

    tier = str(packet.get("audit_priority_tier", "")).lower()
    if tier in ["elevated", "extreme"]:
        return "Prioritize for payment-integrity analyst review."

    return "Not prioritized for payment-integrity review; continue standard workflow."

def build_evidence_packet(row: pd.Series) -> Dict[str, Any]:
    """Constructs the structured evidence packet strictly using Stage 6 output fields."""
    claim_id = _safe_get(row, 'CLM_ID', _safe_get(row, 'CLAIM_LINE_ID'))
    proc_code = _safe_get(row, 'HCPCS_CD_1', _safe_get(row, 'ORIGINAL_HCPCS_CD'))
    stage4_score = _safe_get(row, 'STAGE4_PRICE_RISK_SCORE')
    stage5_score = _safe_get(row, 'STAGE5_IF_RISK_SCORE')

    dq_flag = bool(_safe_get(row, 'STAGE6_DATA_QUALITY_REVIEW', False))
    missing_hcpcs_flag = bool(_safe_get(row, 'STAGE5_MISSING_HCPCS_FLAG', False))

    is_dq_review = (
        dq_flag or
        missing_hcpcs_flag or
        proc_code is None or
        (stage4_score is None and stage5_score is None)
    )

    packet = {
        "claim_identifier": str(claim_id) if claim_id is not None else None,
        "procedure_code": str(proc_code) if proc_code is not None else None,
        "cluster_segment": _safe_get(row, 'STAGE5_CLUSTER_KEY'),

        "working_payment_amount": _safe_get(row, 'WORKING_PMT_AMT'),
        "expected_payment_amount": _safe_get(row, 'EXPECTED_PMT_FINAL'),
        "payment_residual_amount": _safe_get(row, 'PAYMENT_RESIDUAL_FINAL'),
        "relative_surge_ratio": _safe_get(row, 'RELATIVE_SURGE_RATIO_FINAL'),
        "payment_ratio_to_expected": _safe_get(row, 'PAYMENT_RATIO_TO_EXPECTED_FINAL'),

        "stage4_price_risk_score": stage4_score,
        "stage4_price_tier": _safe_get(row, 'STAGE4_PRICE_TIER'),

        "stage5_multivariate_risk_score": stage5_score,
        "stage5_anomaly_tier": _safe_get(row, 'STAGE5_IF_TIER'),

        "ensemble_risk_score": _safe_get(row, 'STAGE6_ENSEMBLE_RISK_SCORE'),
        "audit_priority_score": _safe_get(row, 'AUDIT_PRIORITY_SCORE'),
        "audit_priority_tier": _safe_get(row, 'STAGE6_AUDIT_TIER'),
        "fusion_source": _safe_get(row, 'STAGE6_FUSION_SOURCE'),

        "data_quality_review": is_dq_review,
        "missing_hcpcs_flag": missing_hcpcs_flag,

        "policy_text_available": False,
        "policy_text": None,

        "required_review_recommendation": None,

        "limitations": [
            "Synthetic/de-identified CMS claims POC data",
            "Scores prioritize claims for human review and do not establish billing error, fraud, medical necessity, overpayment, or payment action"
        ]
    }

    packet["required_review_recommendation"] = get_required_recommendation(packet)
    return packet


# =========================================================================
# 5. SAFETY & GROUNDING GUARDRAIL VALIDATOR
# =========================================================================

def validate_memo_output(evidence_packet: Dict[str, Any], memo_obj: ClaimReviewMemo) -> Tuple[bool, List[str], Dict[str, Any]]:
    """
    Validates LLM output against strict grounding, safety, and routing guardrails.
    Returns (is_valid, validation_errors, memo_dict).
    """
    errors = []
    memo_dict = memo_obj.model_dump()

    # 1. Prohibited Terms Check (excluding the standard limitations string)
    check_dict = {k: v for k, v in memo_dict.items() if k != 'limitations'}
    memo_text_block = json.dumps(check_dict).lower()

    found_prohibited = [term for term in PROHIBITED_TERMS if term in memo_text_block]
    if found_prohibited:
        errors.append(f"Safety Violation: Prohibited terms detected: {found_prohibited}")

    # 2. Recommendation Match Check
    required_rec = evidence_packet.get("required_review_recommendation")
    if memo_obj.review_recommendation != required_rec:
        errors.append(
            f"Routing Error: review_recommendation '{memo_obj.review_recommendation}' "
            f"does not exactly match required_review_recommendation '{required_rec}'."
        )

    # 3. Evidence Items Count and Signal Validation
    if len(memo_obj.evidence) > 4:
        errors.append(f"Schema Violation: Maximum 4 evidence items allowed, found {len(memo_obj.evidence)}.")

    packet_keys = set(evidence_packet.keys())
    for ev in memo_obj.evidence:
        # Validate approved signal name
        if ev.signal not in APPROVED_SIGNALS:
            errors.append(f"Grounding Error: Signal '{ev.signal}' is not in APPROVED_SIGNALS.")

        # Validate source fields existence and null values
        for sf in ev.source_fields:
            if sf not in packet_keys:
                errors.append(f"Grounding Error: Source field '{sf}' does not exist in the evidence packet.")
            elif evidence_packet.get(sf) is None:
                errors.append(f"Grounding Error: Source field '{sf}' is cited but is None in the evidence packet.")

    # 4. Data Quality Routing Constraints
    if evidence_packet.get("data_quality_review") is True:
        if "data-quality review" not in memo_obj.review_recommendation.lower():
            errors.append("Routing Error: data_quality_review is True but recommendation missing 'data-quality review'.")
        if "analyst review" in memo_obj.review_recommendation.lower() and "data-quality" not in memo_obj.review_recommendation.lower():
            errors.append("Routing Error: Cannot recommend payment-integrity analyst review for data-quality claims.")

    # 5. Policy Hallucination Check
    if evidence_packet.get("policy_text_available") is False:
        disallowed_policy_terms = ["violated policy", "policy violation", "contract violation", "fee schedule violation"]
        for term in disallowed_policy_terms:
            if term in memo_text_block:
                errors.append(f"Grounding Error: Referenced '{term}' when policy_text_available is False.")

    is_valid = len(errors) == 0
    return is_valid, errors, memo_dict


# =========================================================================
# 6. CORE REUSABLE ENGINE FUNCTION
# =========================================================================

def generate_claim_review_memo(selected_claim_row: pd.Series) -> Dict[str, Any]:
    """
    Main entry point for generating a single claim review memo.
    Constructs packet, calls Groq LLM, validates output, and returns structured result.
    """
    model_name = "llama-3.3-70b-versatile"

    # Check for API key securely without exposing environment secrets
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        return {
            "evidence_packet": {},
            "memo": None,
            "validation_passed": False,
            "validation_errors": ["System Error: GROQ_API_KEY environment variable is not set."],
            "model_name": model_name
        }

    # Build Evidence Packet
    packet = build_evidence_packet(selected_claim_row)

    # Initialize LLM with Graceful Exception Handling
    try:
        llm = ChatGroq(
            model_name=model_name,
            temperature=0.1,
            max_retries=2
        )
        structured_llm = llm.with_structured_output(ClaimReviewMemo)

        prompt_content = f"CLAIM_EVIDENCE:\n{json.dumps(packet, indent=2)}"
        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=prompt_content)
        ]

        # Single claim invocation
        response_memo = structured_llm.invoke(messages)

    except OutputParserException as e:
        return {
            "evidence_packet": packet,
            "memo": None,
            "validation_passed": False,
            "validation_errors": [f"Parsing Error: Model output failed structured Pydantic parsing."],
            "model_name": model_name
        }
    except Exception as e:
        error_msg = str(e)
        if "rate limit" in error_msg.lower():
            safe_error = "API Error: Groq rate limit exceeded. Please try again later."
        else:
            safe_error = "API Error: Internal API call failure."
            logger.error(f"Groq API Error: {error_msg}")

        return {
            "evidence_packet": packet,
            "memo": None,
            "validation_passed": False,
            "validation_errors": [safe_error],
            "model_name": model_name
        }

    # Validate output grounding & guardrails
    is_valid, validation_errors, clean_memo = validate_memo_output(packet, response_memo)

    return {
        "evidence_packet": packet,
        "memo": clean_memo if response_memo else None,
        "validation_passed": is_valid,
        "validation_errors": validation_errors,
        "model_name": model_name
    }