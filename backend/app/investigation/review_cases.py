from pathlib import Path

import pandas as pd


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]

RESULTS_DIR = PROJECT_ROOT / "data" / "results"

INPUT_FILE = RESULTS_DIR / "investigation_cases.csv"
OUTPUT_FILE = RESULTS_DIR / "case_reviews.csv"


# ============================================================
# LOAD INVESTIGATION CASES
# ============================================================

def load_cases():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Investigation cases file not found:\n{INPUT_FILE}"
        )

    df = pd.read_csv(INPUT_FILE)

    required_columns = [
        "case_id",
        "alert_id",
        "transaction_id",
        "timestamp",
        "sender_account",
        "receiver_account",
        "amount",
        "risk_score",
        "risk_category",
        "case_priority",
        "case_status",
        "rf_probability",
        "tgn_probability",
        "rule_score",
        "triggered_rules",
        "large_transaction",
        "structuring",
        "rapid_movement",
        "fan_out",
        "circular_movement",
        "evidence_summary",
        "is_fraud",
        "typology",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "Missing required investigation columns:\n"
            + "\n".join(missing_columns)
        )

    return df


# ============================================================
# INITIAL REVIEW STATUS
# ============================================================

def determine_review_status(row):
    """
    Initial review workflow state.

    Cases are not automatically declared fraudulent.
    Every generated case starts as requiring review.
    """

    return "PENDING_REVIEW"


# ============================================================
# BUILD INVESTIGATOR SUMMARY
# ============================================================

def build_investigator_summary(row):
    summary_parts = []

    summary_parts.append(
        f"Transaction {row['transaction_id']}"
    )

    summary_parts.append(
        f"risk category {row['risk_category']}"
    )

    summary_parts.append(
        f"risk score {row['risk_score']:.4f}"
    )

    if row["triggered_rules"]:
        summary_parts.append(
            f"triggered rules: {row['triggered_rules']}"
        )
    else:
        summary_parts.append(
            "no AML rules triggered"
        )

    return "; ".join(summary_parts)


# ============================================================
# BUILD REVIEW TABLE
# ============================================================

def build_review_table(cases):
    reviews = cases.copy()

    reviews["review_status"] = reviews.apply(
        determine_review_status,
        axis=1,
    )

    reviews["investigator_decision"] = "UNDECIDED"

    reviews["investigator_notes"] = ""

    reviews["investigator_summary"] = reviews.apply(
        build_investigator_summary,
        axis=1,
    )

    review_columns = [
        "case_id",
        "alert_id",
        "transaction_id",
        "timestamp",
        "sender_account",
        "receiver_account",
        "amount",
        "risk_score",
        "risk_category",
        "case_priority",
        "case_status",
        "review_status",
        "investigator_decision",
        "investigator_notes",
        "investigator_summary",
        "rf_probability",
        "tgn_probability",
        "rule_score",
        "triggered_rules",
        "large_transaction",
        "structuring",
        "rapid_movement",
        "fan_out",
        "circular_movement",
        "evidence_summary",
        "is_fraud",
        "typology",
    ]

    return reviews[review_columns]


# ============================================================
# VALIDATE REVIEW TABLE
# ============================================================

def validate_review_table(review_df):
    if review_df["case_id"].duplicated().any():
        raise ValueError(
            "Duplicate case IDs detected."
        )

    if review_df["alert_id"].duplicated().any():
        raise ValueError(
            "Duplicate alert IDs detected."
        )

    if review_df["transaction_id"].duplicated().any():
        raise ValueError(
            "Duplicate transaction IDs detected."
        )

    invalid_statuses = set(
        review_df["review_status"].dropna()
    ) - {"PENDING_REVIEW"}

    if invalid_statuses:
        raise ValueError(
            f"Unexpected review statuses: {invalid_statuses}"
        )

    invalid_decisions = set(
        review_df["investigator_decision"].dropna()
    ) - {"UNDECIDED"}

    if invalid_decisions:
        raise ValueError(
            f"Unexpected investigator decisions: {invalid_decisions}"
        )


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 70)
    print("AEGISGRAPH CASE REVIEW LAYER")
    print("=" * 70)

    cases = load_cases()

    print(f"Investigation cases loaded: {len(cases)}")

    review_df = build_review_table(cases)

    validate_review_table(review_df)

    print()
    print("Review summary:")
    print(f"  Total cases: {len(review_df)}")

    print()
    print("Review status distribution:")
    print(
        review_df["review_status"]
        .value_counts()
        .sort_index()
    )

    print()
    print("Investigator decision distribution:")
    print(
        review_df["investigator_decision"]
        .value_counts()
        .sort_index()
    )

    print()
    print("Validation:")
    print(
        f"  Duplicate case IDs: "
        f"{review_df['case_id'].duplicated().sum()}"
    )
    print(
        f"  Duplicate alert IDs: "
        f"{review_df['alert_id'].duplicated().sum()}"
    )
    print(
        f"  Duplicate transaction IDs: "
        f"{review_df['transaction_id'].duplicated().sum()}"
    )

    # --------------------------------------------------------
    # SAVE REVIEW TABLE
    # --------------------------------------------------------

    review_df.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print()
    print("Saved to:")
    print(OUTPUT_FILE)

    print("=" * 70)


if __name__ == "__main__":
    main()