"""
AegisGraph - AML Rule Evaluation

Evaluates AML rules against the chronological test set.

IMPORTANT:
    is_fraud and typology are used ONLY after rule execution,
    for evaluation. They are never passed into the rule engine.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from app.rules.aml_rules import (
    score_transactions_chronologically,
)


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[3]

TRANSACTIONS_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "transactions_clean.csv"
)

RESULTS_DIR = (
    PROJECT_ROOT
    / "data"
    / "results"
)

RULE_EVALUATION_FILE = (
    RESULTS_DIR
    / "aml_rule_evaluation.csv"
)

RULE_TYPOLOGY_FILE = (
    RESULTS_DIR
    / "aml_rule_typology_evaluation.csv"
)


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

TRAIN_RATIO = 0.80


# ---------------------------------------------------------------------
# Load chronological test set
# ---------------------------------------------------------------------

def load_transactions() -> pd.DataFrame:
    """
    Load all cleaned transactions in chronological order.

    Ground-truth columns are retained here only for evaluation.
    """

    if not TRANSACTIONS_FILE.exists():
        raise FileNotFoundError(
            f"Transaction file not found: "
            f"{TRANSACTIONS_FILE}"
        )

    transactions = pd.read_csv(
        TRANSACTIONS_FILE
    )

    required_columns = [
        "transaction_id",
        "timestamp",
        "sender_account",
        "receiver_account",
        "amount",
        "is_fraud",
        "typology",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in transactions.columns
    ]

    if missing_columns:
        raise ValueError(
            "Transactions are missing required evaluation "
            "columns: "
            + ", ".join(missing_columns)
        )

    transactions["timestamp"] = pd.to_datetime(
        transactions["timestamp"],
        errors="coerce",
    )

    if transactions["timestamp"].isna().any():
        raise ValueError(
            "Invalid timestamps detected."
        )

    transactions = (
        transactions
        .sort_values("timestamp", kind="mergesort")
        .reset_index(drop=True)
    )

    return transactions


def load_test_transactions() -> pd.DataFrame:
    """Return the chronological 20% test partition."""

    transactions = load_transactions()
    split_index = int(len(transactions) * TRAIN_RATIO)
    return transactions.iloc[split_index:].copy().reset_index(drop=True)


# ---------------------------------------------------------------------
# Run rules against test history
# ---------------------------------------------------------------------

def run_rules_on_test_history(
    test_transactions: pd.DataFrame,
    prior_history: pd.DataFrame | None = None,
) -> dict:
    """
    Run all history-dependent rules using only transaction
    attributes.

    Returns the transaction IDs identified by each rule.
    """

    rule_input_columns = [
        "transaction_id",
        "timestamp",
        "sender_account",
        "receiver_account",
        "amount",
    ]

    test_rule_data = test_transactions[rule_input_columns].copy()
    if prior_history is not None:
        history_rule_data = prior_history[rule_input_columns].copy()
        rule_data = pd.concat(
            [history_rule_data, test_rule_data],
            ignore_index=True,
        )
    else:
        rule_data = test_rule_data

    if test_rule_data["transaction_id"].duplicated().any():
        raise ValueError(
            "Test transactions must have unique transaction IDs."
        )

    scored = score_transactions_chronologically(rule_data)
    test_ids = set(test_rule_data["transaction_id"])
    scored = scored[scored["transaction_id"].isin(test_ids)]

    return {
        rule_name: set(
            scored.loc[
                scored[rule_name] == 1,
                "transaction_id",
            ]
        )
        for rule_name in (
            "large_transaction",
            "structuring",
            "rapid_movement",
            "fan_out",
            "circular_movement",
        )
    }


# ---------------------------------------------------------------------
# Build transaction-level predictions
# ---------------------------------------------------------------------

def build_predictions(
    test_transactions: pd.DataFrame,
    rule_matches: dict,
) -> pd.DataFrame:
    """
    Convert rule transaction-ID matches into transaction-level
    binary predictions.
    """

    predictions = test_transactions[
        [
            "transaction_id",
            "timestamp",
            "is_fraud",
            "typology",
        ]
    ].copy()

    predictions["large_transaction"] = (
        predictions["transaction_id"]
        .isin(
            rule_matches[
                "large_transaction"
            ]
        )
        .astype(int)
    )

    predictions["structuring"] = (
        predictions["transaction_id"]
        .isin(
            rule_matches[
                "structuring"
            ]
        )
        .astype(int)
    )

    predictions["rapid_movement"] = (
        predictions["transaction_id"]
        .isin(
            rule_matches[
                "rapid_movement"
            ]
        )
        .astype(int)
    )

    predictions["fan_out"] = (
        predictions["transaction_id"]
        .isin(
            rule_matches[
                "fan_out"
            ]
        )
        .astype(int)
    )

    predictions["circular_movement"] = (
        predictions["transaction_id"]
        .isin(
            rule_matches[
                "circular_movement"
            ]
        )
        .astype(int)
    )

    # A transaction is suspicious if at least one rule
    # identifies that transaction.
    predictions["rule_prediction"] = (
        predictions[
            [
                "large_transaction",
                "structuring",
                "rapid_movement",
                "fan_out",
                "circular_movement",
            ]
        ]
        .max(axis=1)
    )

    return predictions


# ---------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------

def calculate_metrics(
    actual: pd.Series,
    predicted: pd.Series,
) -> dict:
    """
    Calculate binary classification metrics.
    """

    tn, fp, fn, tp = confusion_matrix(
        actual,
        predicted,
        labels=[0, 1],
    ).ravel()

    return {
        "test_transactions": int(
            len(actual)
        ),
        "test_fraud": int(
            actual.sum()
        ),
        "accuracy": accuracy_score(
            actual,
            predicted,
        ),
        "precision": precision_score(
            actual,
            predicted,
            zero_division=0,
        ),
        "recall": recall_score(
            actual,
            predicted,
            zero_division=0,
        ),
        "f1": f1_score(
            actual,
            predicted,
            zero_division=0,
        ),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
    }


# ---------------------------------------------------------------------
# Rule-level metrics
# ---------------------------------------------------------------------

def calculate_rule_metrics(
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate precision, recall and F1 for each individual rule.
    """

    actual = predictions["is_fraud"]

    rule_columns = [
        "large_transaction",
        "structuring",
        "rapid_movement",
        "fan_out",
        "circular_movement",
        "rule_prediction",
    ]

    rows = []

    for rule_name in rule_columns:

        predicted = predictions[
            rule_name
        ]

        metrics = calculate_metrics(
            actual,
            predicted,
        )

        rows.append(
            {
                "rule": rule_name,
                **metrics,
            }
        )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# Typology evaluation
# ---------------------------------------------------------------------

def calculate_typology_evaluation(
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate rule detection by fraud typology.

    Detection means transaction-level recall within that typology.
    """

    fraud = predictions[
        predictions["is_fraud"] == 1
    ].copy()

    rows = []

    for typology in sorted(
        fraud["typology"].unique()
    ):

        typology_rows = fraud[
            fraud["typology"] == typology
        ]

        total = len(
            typology_rows
        )

        detected = int(
            typology_rows[
                "rule_prediction"
            ].sum()
        )

        missed = (
            total - detected
        )

        recall = (
            detected / total
            if total > 0
            else 0.0
        )

        rows.append(
            {
                "typology": typology,
                "total": total,
                "rule_detected": detected,
                "rule_recall": recall,
                "rule_missed": missed,
            }
        )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------

def evaluate_rules() -> None:

    print("=" * 70)
    print("AEGISGRAPH AML RULE EVALUATION")
    print("=" * 70)

    # -------------------------------------------------------------
    # Load test set
    # -------------------------------------------------------------

    all_transactions = load_transactions()
    split_index = int(len(all_transactions) * TRAIN_RATIO)
    prior_history = all_transactions.iloc[:split_index].copy()
    test_transactions = all_transactions.iloc[split_index:].copy().reset_index(
        drop=True
    )

    print()
    print(
        f"Test transactions: "
        f"{len(test_transactions)}"
    )

    print(
        f"Test fraud cases: "
        f"{int(test_transactions['is_fraud'].sum())}"
    )

    # -------------------------------------------------------------
    # Run rules
    # -------------------------------------------------------------

    print()
    print(
        "Running AML rules against test history..."
    )

    rule_matches = run_rules_on_test_history(
        test_transactions,
        prior_history=prior_history,
    )

    # -------------------------------------------------------------
    # Build transaction-level predictions
    # -------------------------------------------------------------

    predictions = build_predictions(
        test_transactions,
        rule_matches,
    )

    # -------------------------------------------------------------
    # Combined AML metrics
    # -------------------------------------------------------------

    metrics = calculate_metrics(
        actual=predictions["is_fraud"],
        predicted=predictions[
            "rule_prediction"
        ],
    )

    print()
    print("Combined AML rule metrics:")
    print(
        f"  Accuracy:  "
        f"{metrics['accuracy']:.4f}"
    )
    print(
        f"  Precision: "
        f"{metrics['precision']:.4f}"
    )
    print(
        f"  Recall:    "
        f"{metrics['recall']:.4f}"
    )
    print(
        f"  F1:        "
        f"{metrics['f1']:.4f}"
    )

    print()
    print("Confusion matrix:")
    print(
        f"  True negatives:  "
        f"{metrics['true_negatives']}"
    )
    print(
        f"  False positives: "
        f"{metrics['false_positives']}"
    )
    print(
        f"  False negatives: "
        f"{metrics['false_negatives']}"
    )
    print(
        f"  True positives:  "
        f"{metrics['true_positives']}"
    )

    # -------------------------------------------------------------
    # Individual rule metrics
    # -------------------------------------------------------------

    rule_metrics = calculate_rule_metrics(
        predictions
    )

    print()
    print("Individual rule metrics:")
    print(
        rule_metrics[
            [
                "rule",
                "precision",
                "recall",
                "f1",
                "false_positives",
                "false_negatives",
            ]
        ].to_string(index=False)
    )

    # -------------------------------------------------------------
    # Typology evaluation
    # -------------------------------------------------------------

    typology_results = (
        calculate_typology_evaluation(
            predictions
        )
    )

    print()
    print("Fraud detection by typology:")
    print(
        typology_results.to_string(
            index=False
        )
    )

    # -------------------------------------------------------------
    # Rule match counts
    # -------------------------------------------------------------

    print()
    print("Transaction matches by rule:")

    for rule_name, transaction_ids in (
        rule_matches.items()
    ):
        print(
            f"  {rule_name}: "
            f"{len(transaction_ids)}"
        )

    # -------------------------------------------------------------
    # Save combined metrics
    # -------------------------------------------------------------

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    combined_metrics = pd.DataFrame(
        [
            {
                "model": "AML Rules",
                **metrics,
            }
        ]
    )

    combined_metrics.to_csv(
        RULE_EVALUATION_FILE,
        index=False,
    )

    # -------------------------------------------------------------
    # Save individual rule metrics
    # -------------------------------------------------------------

    rule_metrics.to_csv(
        RESULTS_DIR
        / "aml_individual_rule_metrics.csv",
        index=False,
    )

    # -------------------------------------------------------------
    # Save typology results
    # -------------------------------------------------------------

    typology_results.to_csv(
        RULE_TYPOLOGY_FILE,
        index=False,
    )

    # -------------------------------------------------------------
    # Save transaction-level rule predictions
    # -------------------------------------------------------------

    predictions.to_csv(
        RESULTS_DIR
        / "aml_rule_test_predictions.csv",
        index=False,
    )

    print()
    print(
        "Saved combined metrics:"
    )
    print(
        RULE_EVALUATION_FILE
    )

    print()
    print(
        "Saved individual rule metrics:"
    )
    print(
        RESULTS_DIR
        / "aml_individual_rule_metrics.csv"
    )

    print()
    print(
        "Saved typology evaluation:"
    )
    print(
        RULE_TYPOLOGY_FILE
    )

    print()
    print(
        "Saved transaction-level predictions:"
    )
    print(
        RESULTS_DIR
        / "aml_rule_test_predictions.csv"
    )

    print()
    print(
        "AML rule evaluation complete."
    )

    print("=" * 70)


# ---------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------

if __name__ == "__main__":
    evaluate_rules()