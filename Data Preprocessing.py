import pandas as pd


BENE_COLS = [
    "DESYNPUF_ID",
    "BENE_BIRTH_DT",
    "BENE_SEX_IDENT_CD",
    "SP_STATE_CODE",
    "SP_ALZHDMTA",
    "SP_CHF",
    "SP_CHRNKIDN",
    "SP_CNCR",
    "SP_COPD",
    "SP_DEPRESSN",
    "SP_DIABETES",
    "SP_ISCHMCHT",
    "SP_OSTEOPRS",
    "SP_RA_OA",
    "SP_STRKETIA",
]

CARRIER_COLS = [
    "DESYNPUF_ID",
    "CLM_ID",
    "CLM_FROM_DT",
    "PRF_PHYSN_NPI_1",
    "HCPCS_CD_1",
    "LINE_NCH_PMT_AMT_1",
]


def validate_required_columns(
    df: pd.DataFrame,
    required_columns: list[str],
    dataframe_name: str,
) -> None:
    """Raise a clear error if a required CMS field is missing."""
    missing_columns = sorted(set(required_columns) - set(df.columns))

    if missing_columns:
        raise KeyError(
            f"{dataframe_name} is missing required columns: {missing_columns}"
        )


def preprocess_claims_data(
    bene_df: pd.DataFrame,
    carrier_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Select CMS fields, merge beneficiary and carrier claims data,
    parse dates, and create a preserved original-payment column.
    """
    validate_required_columns(bene_df, BENE_COLS, "Beneficiary dataframe")
    validate_required_columns(carrier_df, CARRIER_COLS, "Carrier dataframe")

    merged_cms_df = pd.merge(
        carrier_df[CARRIER_COLS].copy(),
        bene_df[BENE_COLS].copy(),
        on="DESYNPUF_ID",
        how="inner",
    )

    merged_cms_df["ORIGINAL_PMT_AMT"] = pd.to_numeric(
        merged_cms_df["LINE_NCH_PMT_AMT_1"],
        errors="coerce",
    ).fillna(0.0)

    merged_cms_df["CLM_FROM_DT"] = pd.to_datetime(
        merged_cms_df["CLM_FROM_DT"].astype(str),
        format="%Y%m%d",
        errors="coerce",
    )

    merged_cms_df["LINE_NCH_PMT_AMT_1"] = merged_cms_df[
        "ORIGINAL_PMT_AMT"
    ].copy()

    return merged_cms_df