"""
AegisGraph - AML Rule Runner

Runs the deterministic AML rules against the cleaned transaction
dataset and saves explainable rule-detection results.

Important:
    is_fraud and typology are ground-truth fields only.
    They are NOT used as rule inputs.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from app.rules.aml_rules import (
    check_large_transaction,
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

RULE_RESULTS_FILE = (
    RESULTS_DIR
    / "aml_rule_results.csv"
)


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

RULE_INPUT_COLUMNS = [
    "transaction_id",
    "timestamp",
    "sender_account",
    "receiver_account",
    "amount",
]


# ---------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------

def load_transactions() -> pd.DataFrame:
    """
    Load cleaned transactions and validate the columns required
    by the AML rules.
    """

    if not TRANSACTIONS_FILE.exists():
        raise FileNotFoundError(
            f"Transaction file not found: "
            f"{TRANSACTIONS_FILE}"
        )

    transactions = pd.read_csv(
        TRANSACTIONS_FILE
    )

    missing_columns = [
        column
        for column in RULE_INPUT_COLUMNS
        if column not in transactions.columns
    ]

    if missing_columns:
        raise ValueError(
            "Transactions are missing required rule-input "
            "columns: "
            + ", ".join(missing_columns)
        )

    return transactions


# ---------------------------------------------------------------------
# Transaction-level rules
# ---------------------------------------------------------------------

def run_large_transaction_rule(
    transactions: pd.DataFrame,
) -> pd.DataFrame:
    """
    Evaluate the large-transaction rule for every transaction.
    """

    results = []

    for _, transaction in transactions.iterrows():

        result = check_large_transaction(
            transaction
        )

        results.append(
            {
                "transaction_id": (
                    transaction["transaction_id"]
                ),
                "rule": result["rule"],
                "triggered": result["triggered"],
                "reason": result["reason"],
            }
        )

    return pd.DataFrame(results)


# ---------------------------------------------------------------------
# History-dependent rules
# ---------------------------------------------------------------------

def run_history_rules(
    transactions: pd.DataFrame,
) -> list[dict]:
    """
    Score history-dependent rules causally for each transaction.
    """

    scored = score_transactions_chronologically(transactions)
    results = []

    for rule_name in (
        "structuring",
        "rapid_movement",
        "fan_out",
        "circular_movement",
    ):
        results.extend(
            {
                "transaction_id": row["transaction_id"],
                "rule": rule_name,
                "triggered": bool(row[rule_name]),
                "reason": (
                    "Causal transaction-level rule evaluation"
                ),
            }
            for _, row in scored.iterrows()
        )

    return results


# ---------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------

def run_rules() -> pd.DataFrame:
    """
    Execute all AML rules and save the results.
    """

    print("=" * 70)
    print("AEGISGRAPH AML RULE RUNNER")
    print("=" * 70)

    # -------------------------------------------------------------
    # Load data
    # -------------------------------------------------------------

    transactions = load_transactions()

    print()
    print(
        f"Loaded transactions: "
        f"{len(transactions)}"
    )

    print(
        "Rule input columns:"
    )

    for column in RULE_INPUT_COLUMNS:
        print(
            f"  - {column}"
        )

    # -------------------------------------------------------------
    # Explicitly prevent ground-truth leakage
    # -------------------------------------------------------------

    forbidden_columns = [
        "is_fraud",
        "typology",
    ]

    print()
    print(
        "Ground-truth columns are excluded "
        "from rule inputs:"
    )

    for column in forbidden_columns:
        print(
            f"  - {column}"
        )

    # Only pass the required rule columns.
    rule_data = transactions[
        RULE_INPUT_COLUMNS
    ].copy()

    # -------------------------------------------------------------
    # Run large transaction rule
    # -------------------------------------------------------------

    print()
    print(
        "Running large transaction rule..."
    )

    large_results = run_large_transaction_rule(
        rule_data
    )

    # -------------------------------------------------------------
    # Run history-dependent rules
    # -------------------------------------------------------------

    print(
        "Running history-dependent rules..."
    )

    history_results = run_history_rules(rule_data)

    history_results_df = pd.DataFrame(
        history_results
    )

    # -------------------------------------------------------------
    # Combine results
    # -------------------------------------------------------------

    results = pd.concat(
        [
            large_results,
            history_results_df,
        ],
        ignore_index=True,
    )

    # -------------------------------------------------------------
    # Save results
    # -------------------------------------------------------------

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    results.to_csv(
        RULE_RESULTS_FILE,
        index=False,
    )

    # -------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------

    print()
    print(
        "Rule results:"
    )

    summary = (
        results
        .groupby("rule")["triggered"]
        .max()
        .reset_index()
    )

    for _, row in summary.iterrows():
        print(
            f"  {row['rule']}: "
            f"{bool(row['triggered'])}"
        )

    large_trigger_count = int(
        large_results["triggered"].sum()
    )

    print()
    print(
        "Large transaction detections: "
        f"{large_trigger_count}"
    )

    print()
    print(
        "Results saved to:"
    )
    print(
        RULE_RESULTS_FILE
    )

    print()
    print(
        "AML rule runner complete."
    )

    print("=" * 70)

    return results


# ---------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------

if __name__ == "__main__":
    run_rules()