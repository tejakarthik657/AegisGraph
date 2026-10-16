from __future__ import annotations

from pathlib import Path

import pandas as pd


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]

DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
RESULTS_DIR = DATA_DIR / "results"

TRANSACTIONS_FILE = (
    PROCESSED_DIR / "transactions_clean.csv"
)

RF_PREDICTIONS_FILE = (
    RESULTS_DIR / "baseline_predictions.csv"
)

TGN_PREDICTIONS_FILE = (
    RESULTS_DIR / "temporal_tgn_predictions.csv"
)

RULE_PREDICTIONS_FILE = (
    RESULTS_DIR / "aml_rule_test_predictions.csv"
)

OUTPUT_FILE = (
    RESULTS_DIR / "risk_fusion_predictions.csv"
)


# ============================================================
# RISK FUSION CONFIGURATION
# ============================================================

# Initial equal weighting.
# These weights are deliberately NOT optimized yet.
RF_WEIGHT = 1.0 / 3.0
TGN_WEIGHT = 1.0 / 3.0
RULE_WEIGHT = 1.0 / 3.0


# Initial risk categories.
# These are prototype thresholds and will be calibrated later.
LOW_RISK_THRESHOLD = 0.25
MEDIUM_RISK_THRESHOLD = 0.50
HIGH_RISK_THRESHOLD = 0.75


# ============================================================
# LOAD TRANSACTION ALIGNMENT
# ============================================================

def load_test_transactions() -> pd.DataFrame:
    """
    Load the chronological transaction dataset and reconstruct
    the same 80/20 test split used by the RF and TGN pipelines.

    Important:
    Random Forest predictions use the ORIGINAL chronological
    dataframe index as transaction_index.

    Therefore the test rows must retain their original index
    values, which are expected to be 8158 through 10197.
    """

    transactions = pd.read_csv(
        TRANSACTIONS_FILE
    )

    transactions["timestamp"] = pd.to_datetime(
        transactions["timestamp"]
    )

    # Sort chronologically.
    # Do NOT reset the index here.
    transactions = (
        transactions
        .sort_values("timestamp")
    )

    split_index = int(
        len(transactions) * 0.80
    )

    # Preserve the original chronological dataframe index.
    test_transactions = (
        transactions
        .iloc[split_index:]
        .copy()
    )

    # RF predictions use this original dataframe index.
    test_transactions["transaction_index"] = (
        test_transactions.index
    )

    return test_transactions


# ============================================================
# LOAD RF PREDICTIONS
# ============================================================

def load_rf_predictions() -> pd.DataFrame:
    """
    Load Random Forest predictions.
    """

    rf = pd.read_csv(
        RF_PREDICTIONS_FILE
    )

    required_columns = {
        "transaction_index",
        "fraud_probability",
    }

    missing = (
        required_columns
        - set(rf.columns)
    )

    if missing:
        raise ValueError(
            "Random Forest predictions are missing "
            f"columns: {sorted(missing)}"
        )

    return rf[
        [
            "transaction_index",
            "fraud_probability",
        ]
    ].copy().rename(
        columns={
            "fraud_probability": "rf_probability"
        }
    )


# ============================================================
# LOAD TGN PREDICTIONS
# ============================================================

def load_tgn_predictions() -> pd.DataFrame:
    """
    Load TemporalMemoryTGN predictions.
    """

    tgn = pd.read_csv(
        TGN_PREDICTIONS_FILE
    )

    required_columns = {
        "transaction_id",
        "fraud_probability",
    }

    missing = (
        required_columns
        - set(tgn.columns)
    )

    if missing:
        raise ValueError(
            "TGN predictions are missing "
            f"columns: {sorted(missing)}"
        )

    return tgn[
        [
            "transaction_id",
            "fraud_probability",
        ]
    ].copy().rename(
        columns={
            "fraud_probability": "tgn_probability"
        }
    )


# ============================================================
# LOAD AML RULE PREDICTIONS
# ============================================================

def load_rule_predictions() -> pd.DataFrame:
    """
    Load transaction-level AML rule predictions.
    """

    rules = pd.read_csv(
        RULE_PREDICTIONS_FILE
    )

    required_columns = {
        "transaction_id",
        "large_transaction",
        "structuring",
        "rapid_movement",
        "fan_out",
        "circular_movement",
        "rule_prediction",
    }

    missing = (
        required_columns
        - set(rules.columns)
    )

    if missing:
        raise ValueError(
            "AML rule predictions are missing "
            f"columns: {sorted(missing)}"
        )

    return rules[
        [
            "transaction_id",
            "large_transaction",
            "structuring",
            "rapid_movement",
            "fan_out",
            "circular_movement",
            "rule_prediction",
        ]
    ].copy()


# ============================================================
# BUILD RULE SCORE
# ============================================================

def calculate_rule_score(
    rules: pd.DataFrame,
) -> pd.DataFrame:
    """
    Convert the binary AML rule signals into an initial
    normalized rule score.

    Each triggered rule contributes equally.

    This is intentionally simple and will be improved later.
    """

    rule_columns = [
        "large_transaction",
        "structuring",
        "rapid_movement",
        "fan_out",
        "circular_movement",
    ]

    rules = rules.copy()

    for column in rule_columns:
        rules[column] = (
            rules[column]
            .astype(int)
        )

    rules["rule_score"] = (
        rules[rule_columns]
        .sum(axis=1)
        / len(rule_columns)
    )

    return rules


# ============================================================
# RISK CATEGORY
# ============================================================

def assign_risk_category(
    risk_score: float,
) -> str:
    """
    Convert the combined risk score into a simple
    risk category.
    """

    if risk_score >= HIGH_RISK_THRESHOLD:
        return "HIGH"

    if risk_score >= MEDIUM_RISK_THRESHOLD:
        return "MEDIUM"

    if risk_score >= LOW_RISK_THRESHOLD:
        return "LOW"

    return "VERY_LOW"


# ============================================================
# BUILD RISK FUSION
# ============================================================

def build_risk_fusion() -> pd.DataFrame:
    """
    Combine Random Forest, TemporalMemoryTGN, and AML rule
    signals into one transaction-level risk table.
    """

    print(
        "=" * 70
    )

    print(
        "AEGISGRAPH RISK FUSION"
    )

    print(
        "=" * 70
    )

    # --------------------------------------------------------
    # Load transaction alignment
    # --------------------------------------------------------

    transactions = (
        load_test_transactions()
    )

    print(
        f"Test transactions: {len(transactions)}"
    )

    print(
        "Test transaction index range: "
        f"{transactions['transaction_index'].min()} "
        f"to "
        f"{transactions['transaction_index'].max()}"
    )

    # --------------------------------------------------------
    # Load model/rule predictions
    # --------------------------------------------------------

    rf = load_rf_predictions()

    tgn = load_tgn_predictions()

    rules = load_rule_predictions()

    print(
        f"RF predictions: {len(rf)}"
    )

    print(
        f"TGN predictions: {len(tgn)}"
    )

    print(
        f"Rule predictions: {len(rules)}"
    )

    # --------------------------------------------------------
    # Validate RF alignment before merge
    # --------------------------------------------------------

    if len(rf) != len(transactions):
        raise ValueError(
            "RF prediction count does not match "
            "the chronological test set."
        )

    if (
        rf["transaction_index"].nunique()
        != len(rf)
    ):
        raise ValueError(
            "RF transaction_index contains duplicates."
        )

    if (
        transactions["transaction_index"].nunique()
        != len(transactions)
    ):
        raise ValueError(
            "Test transaction_index contains duplicates."
        )

    # --------------------------------------------------------
    # Merge RF predictions
    # --------------------------------------------------------

    transactions = transactions.merge(
        rf,
        on="transaction_index",
        how="left",
        validate="one_to_one",
    )

    # --------------------------------------------------------
    # Merge TGN predictions
    # --------------------------------------------------------

    transactions = transactions.merge(
        tgn,
        on="transaction_id",
        how="left",
        validate="one_to_one",
    )

    # --------------------------------------------------------
    # Merge AML rule predictions
    # --------------------------------------------------------

    rules = calculate_rule_score(
        rules
    )

    transactions = transactions.merge(
        rules,
        on="transaction_id",
        how="left",
        validate="one_to_one",
    )

    # --------------------------------------------------------
    # Validate complete alignment
    # --------------------------------------------------------

    probability_columns = [
        "rf_probability",
        "tgn_probability",
        "rule_score",
    ]

    for column in probability_columns:

        missing_count = int(
            transactions[column]
            .isna()
            .sum()
        )

        if missing_count > 0:
            raise ValueError(
                f"{column} contains "
                f"{missing_count} missing values "
                "after alignment."
            )

    # --------------------------------------------------------
    # Validate transaction IDs
    # --------------------------------------------------------

    if transactions["transaction_id"].isna().any():
        raise ValueError(
            "Some transactions are missing transaction_id."
        )

    if (
        transactions["transaction_id"].nunique()
        != len(transactions)
    ):
        raise ValueError(
            "Duplicate transaction IDs detected "
            "after risk-fusion merge."
        )

    # --------------------------------------------------------
    # Combined risk score
    # --------------------------------------------------------

    transactions["risk_score"] = (
        RF_WEIGHT
        * transactions["rf_probability"]
        +
        TGN_WEIGHT
        * transactions["tgn_probability"]
        +
        RULE_WEIGHT
        * transactions["rule_score"]
    )

    # --------------------------------------------------------
    # Risk category
    # --------------------------------------------------------

    transactions["risk_category"] = (
        transactions["risk_score"]
        .apply(assign_risk_category)
    )

    # --------------------------------------------------------
    # Final selected columns
    # --------------------------------------------------------

    output_columns = [
        "transaction_id",
        "timestamp",
        "sender_account",
        "receiver_account",
        "amount",
        "is_fraud",
        "typology",
        "rf_probability",
        "tgn_probability",
        "rule_score",
        "large_transaction",
        "structuring",
        "rapid_movement",
        "fan_out",
        "circular_movement",
        "risk_score",
        "risk_category",
    ]

    output = transactions[
        output_columns
    ].copy()

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print(
        "\nAlignment validation:"
    )

    print(
        f"  RF missing probabilities: "
        f"{output['rf_probability'].isna().sum()}"
    )

    print(
        f"  TGN missing probabilities: "
        f"{output['tgn_probability'].isna().sum()}"
    )

    print(
        f"  Rule missing scores: "
        f"{output['rule_score'].isna().sum()}"
    )

    print(
        f"\nRows generated: {len(output)}"
    )

    print(
        "\nRisk category distribution:"
    )

    print(
        output["risk_category"]
        .value_counts()
        .sort_index()
    )

    print(
        "\nRisk score summary:"
    )

    print(
        output["risk_score"]
        .describe()
        .to_string()
    )

    print(
        f"\nSaved to:\n{OUTPUT_FILE}"
    )

    print(
        "=" * 70
    )

    return output


# ============================================================
# SCRIPT ENTRY POINT
# ============================================================

if __name__ == "__main__":
    build_risk_fusion()