from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


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

THGNN_PREDICTIONS_FILE = (
    RESULTS_DIR / "thgnn_predictions.csv"
)

RULE_PREDICTIONS_FILE = (
    RESULTS_DIR / "aml_rule_test_predictions.csv"
)

OUTPUT_FILE = (
    RESULTS_DIR / "risk_fusion_predictions.csv"
)

METRICS_FILE = (
    RESULTS_DIR / "risk_fusion_metrics.csv"
)


# ============================================================
# RISK FUSION CONFIGURATION
# ============================================================

# Provisional equal weights; these are not optimized yet.
RF_WEIGHT = 0.25
TGN_WEIGHT = 0.25
THGNN_WEIGHT = 0.25
RULE_WEIGHT = 0.25

CLASSIFICATION_THRESHOLD = 0.5


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
# LOAD THGNN PREDICTIONS
# ============================================================

def load_thgnn_predictions() -> pd.DataFrame:
    """
    Load THGNN transaction-level fraud probabilities.
    """

    thgnn = pd.read_csv(
        THGNN_PREDICTIONS_FILE
    )

    required_columns = {
        "transaction_id",
        "fraud_probability",
    }

    missing = (
        required_columns
        - set(thgnn.columns)
    )

    if missing:
        raise ValueError(
            "THGNN predictions are missing "
            f"columns: {sorted(missing)}"
        )

    return thgnn[
        [
            "transaction_id",
            "fraud_probability",
        ]
    ].copy().rename(
        columns={
            "fraud_probability": "thgnn_probability"
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


def validate_prediction_alignment(
    source_name: str,
    predictions: pd.DataFrame,
    key_column: str,
    expected_keys: set,
) -> None:
    """Require prediction keys to match the expected test keys exactly."""

    keys = predictions[key_column]

    if keys.isna().any():
        raise ValueError(
            f"{source_name} predictions contain missing "
            f"{key_column} values."
        )

    duplicates = keys[keys.duplicated()].drop_duplicates().tolist()
    if duplicates:
        raise ValueError(
            f"{source_name} predictions contain duplicate "
            f"{key_column} values: {sorted(duplicates, key=str)}"
        )

    actual_keys = set(keys.tolist())
    missing_keys = expected_keys - actual_keys
    unexpected_keys = actual_keys - expected_keys

    if missing_keys or unexpected_keys:
        raise ValueError(
            f"{source_name} prediction {key_column} mismatch: "
            f"missing {key_column} values "
            f"{sorted(missing_keys, key=str)}; unexpected "
            f"{key_column} values {sorted(unexpected_keys, key=str)}."
        )


# ============================================================
# BUILD RISK FUSION
# ============================================================

def build_risk_fusion() -> pd.DataFrame:
    """
    Combine Random Forest, TemporalMemoryTGN, THGNN, and AML
    rule signals into one transaction-level risk table.
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

    thgnn = load_thgnn_predictions()

    rules = load_rule_predictions()

    print(
        f"RF predictions: {len(rf)}"
    )

    print(
        f"TGN predictions: {len(tgn)}"
    )

    print(
        f"THGNN predictions: {len(thgnn)}"
    )

    print(
        f"Rule predictions: {len(rules)}"
    )

    if transactions["transaction_id"].isna().any():
        raise ValueError(
            "Chronological test set contains missing transaction_id values."
        )

    if transactions["transaction_id"].duplicated().any():
        duplicates = (
            transactions.loc[
                transactions["transaction_id"].duplicated(keep=False),
                "transaction_id",
            ]
            .drop_duplicates()
            .tolist()
        )
        raise ValueError(
            "Chronological test set contains duplicate transaction_id "
            f"values: {sorted(duplicates, key=str)}"
        )

    expected_transaction_ids = set(
        transactions["transaction_id"].tolist()
    )
    expected_transaction_indices = set(
        transactions["transaction_index"].tolist()
    )

    validate_prediction_alignment(
        "RF",
        rf,
        "transaction_index",
        expected_transaction_indices,
    )
    validate_prediction_alignment(
        "TGN",
        tgn,
        "transaction_id",
        expected_transaction_ids,
    )
    validate_prediction_alignment(
        "THGNN",
        thgnn,
        "transaction_id",
        expected_transaction_ids,
    )
    validate_prediction_alignment(
        "AML rule",
        rules,
        "transaction_id",
        expected_transaction_ids,
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
    # Merge THGNN predictions
    # --------------------------------------------------------

    transactions = transactions.merge(
        thgnn,
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
        "thgnn_probability",
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
        THGNN_WEIGHT
        * transactions["thgnn_probability"]
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
        "thgnn_probability",
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
    # Evaluate fused predictions on the chronological test set
    # --------------------------------------------------------

    y_true = output["is_fraud"].to_numpy(dtype=int)
    risk_scores = output["risk_score"].to_numpy(dtype=float)
    y_pred = (
        risk_scores >= CLASSIFICATION_THRESHOLD
    ).astype(int)

    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    )
    true_negatives, false_positives, false_negatives, true_positives = (
        matrix.ravel()
    )

    if y_true.size == 0:
        precision = recall = f1 = average_precision = float("nan")
        roc_auc = float("nan")
    else:
        precision = float(
            precision_score(y_true, y_pred, zero_division=0)
        )
        recall = float(
            recall_score(y_true, y_pred, zero_division=0)
        )
        f1 = float(
            f1_score(y_true, y_pred, zero_division=0)
        )
        if np.unique(y_true).size == 2:
            average_precision = float(
                average_precision_score(y_true, risk_scores)
            )
            roc_auc = float(
                roc_auc_score(y_true, risk_scores)
            )
        else:
            average_precision = float(y_true[0])
            roc_auc = float("nan")

    metrics = {
        "threshold": CLASSIFICATION_THRESHOLD,
        "test_transactions": int(len(output)),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
        "pr_auc": average_precision,
        "true_positives": int(true_positives),
        "false_positives": int(false_positives),
        "true_negatives": int(true_negatives),
        "false_negatives": int(false_negatives),
    }

    METRICS_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    pd.DataFrame([metrics]).to_csv(
        METRICS_FILE,
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
        f"  THGNN missing probabilities: "
        f"{output['thgnn_probability'].isna().sum()}"
    )

    print(
        f"  Rule missing scores: "
        f"{output['rule_score'].isna().sum()}"
    )

    print(
        f"\nRows generated: {len(output)}"
    )

    print(
        f"\nRisk-fusion metrics (threshold={CLASSIFICATION_THRESHOLD}):"
    )
    print(
        f"  Precision: {precision}"
    )
    print(
        f"  Recall: {recall}"
    )
    print(
        f"  F1-score: {f1}"
    )
    print(
        f"  ROC-AUC: {roc_auc}"
    )
    print(
        f"  Average precision (PR-AUC): {average_precision}"
    )
    print(
        "  Confusion matrix "
        "(rows=true [0, 1], columns=predicted [0, 1]):"
    )
    print(matrix)
    print(
        f"  True positives: {true_positives}"
    )
    print(
        f"  False positives: {false_positives}"
    )
    print(
        f"  True negatives: {true_negatives}"
    )
    print(
        f"  False negatives: {false_negatives}"
    )
    print(
        f"  Metrics saved to: {METRICS_FILE}"
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
