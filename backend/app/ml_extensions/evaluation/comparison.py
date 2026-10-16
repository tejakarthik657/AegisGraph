"""
Compare Random Forest and TemporalMemoryTGN results.

The comparison is performed on the exact same chronological test
transactions. Predictions are aligned using transaction_id before
any metrics are calculated.
"""

from pathlib import Path

import pandas as pd

from app.ml_extensions.evaluation.metrics import (
    calculate_binary_metrics,
    calculate_confusion_matrix,
    calculate_typology_detection,
)


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[4]

RESULTS_DIR = PROJECT_ROOT / "data" / "results"

RF_PREDICTIONS_FILE = RESULTS_DIR / "baseline_predictions.csv"
TGN_PREDICTIONS_FILE = RESULTS_DIR / "temporal_tgn_predictions.csv"

COMPARISON_METRICS_FILE = (
    RESULTS_DIR / "model_comparison_metrics.csv"
)

TYPOLOGY_FILE = (
    RESULTS_DIR / "model_typology_comparison.csv"
)


# ---------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------

def load_predictions():
    """Load Random Forest and TemporalMemoryTGN predictions."""

    if not RF_PREDICTIONS_FILE.exists():
        raise FileNotFoundError(
            f"Random Forest predictions not found:\n"
            f"{RF_PREDICTIONS_FILE}"
        )

    if not TGN_PREDICTIONS_FILE.exists():
        raise FileNotFoundError(
            f"TemporalMemoryTGN predictions not found:\n"
            f"{TGN_PREDICTIONS_FILE}"
        )

    rf = pd.read_csv(RF_PREDICTIONS_FILE)
    tgn = pd.read_csv(TGN_PREDICTIONS_FILE)

    return rf, tgn


# ---------------------------------------------------------------------
# Random Forest normalization
# ---------------------------------------------------------------------

def normalize_rf_predictions(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Normalize the existing Random Forest prediction schema.

    The RF prediction file stores the chronological test-row
    position in `transaction_index`, not the actual transaction ID.

    We therefore map transaction_index back to transaction_id using
    the chronologically sorted cleaned transaction dataset.
    """

    df = df.copy()

    required_columns = {
        "transaction_index",
        "actual_is_fraud",
        "actual_typology",
        "predicted_is_fraud",
        "fraud_probability",
    }

    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(
            "Random Forest predictions are missing columns: "
            f"{sorted(missing)}"
        )

    # -------------------------------------------------------------
    # Load the same cleaned transaction dataset used by the models
    # -------------------------------------------------------------

    transactions_file = (
        PROJECT_ROOT
        / "data"
        / "processed"
        / "transactions_clean.csv"
    )

    if not transactions_file.exists():
        raise FileNotFoundError(
            "Clean transaction dataset not found:\n"
            f"{transactions_file}"
        )

    transactions = pd.read_csv(
        transactions_file
    )

    if "timestamp" not in transactions.columns:
        raise ValueError(
            "transactions_clean.csv does not contain "
            "the required timestamp column."
        )

    if "transaction_id" not in transactions.columns:
        raise ValueError(
            "transactions_clean.csv does not contain "
            "the required transaction_id column."
        )

    # -------------------------------------------------------------
    # Reproduce the chronological ordering used by the models
    # -------------------------------------------------------------

    transactions["timestamp"] = pd.to_datetime(
        transactions["timestamp"],
        errors="coerce",
    )

    if transactions["timestamp"].isna().any():
        raise ValueError(
            "Invalid timestamps found in transactions_clean.csv."
        )

    transactions = (
        transactions
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    # -------------------------------------------------------------
    # Validate RF indices
    # -------------------------------------------------------------

    rf_indices = (
        df["transaction_index"]
        .astype(int)
    )

    if rf_indices.min() < 0:
        raise ValueError(
            "Random Forest transaction_index contains "
            "negative values."
        )

    if rf_indices.max() >= len(transactions):
        raise ValueError(
            "Random Forest transaction_index contains values "
            "outside the transaction dataset."
        )

    # -------------------------------------------------------------
    # Map chronological row index -> actual transaction ID
    # -------------------------------------------------------------

    df["transaction_id"] = (
        rf_indices
        .map(
            transactions["transaction_id"]
        )
    )

    if df["transaction_id"].isna().any():
        raise ValueError(
            "Failed to map one or more Random Forest "
            "transaction indices to transaction IDs."
        )

    # -------------------------------------------------------------
    # Rename prediction columns
    # -------------------------------------------------------------

    df = df.rename(
        columns={
            "actual_is_fraud": "is_fraud",
            "actual_typology": "typology",
            "predicted_is_fraud": "prediction",
        }
    )

    return df


# ---------------------------------------------------------------------
# TGN normalization
# ---------------------------------------------------------------------

def normalize_tgn_predictions(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Normalize TemporalMemoryTGN prediction columns.

    Expected TGN columns include:

        transaction_id
        timestamp
        sender_account
        receiver_account
        amount
        is_fraud
        fraud_probability
        prediction
    """

    df = df.copy()

    required_columns = {
        "transaction_id",
        "prediction",
        "fraud_probability",
    }

    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(
            "TemporalMemoryTGN predictions are missing columns: "
            f"{sorted(missing)}"
        )

    return df


# ---------------------------------------------------------------------
# Test-set verification
# ---------------------------------------------------------------------

def verify_same_test_set(
    rf: pd.DataFrame,
    tgn: pd.DataFrame,
) -> None:
    """
    Verify that both models were evaluated on exactly the same
    transactions.
    """

    rf_ids = (
        rf["transaction_id"]
        .astype(str)
        .tolist()
    )

    tgn_ids = (
        tgn["transaction_id"]
        .astype(str)
        .tolist()
    )

    if len(rf_ids) != len(tgn_ids):
        raise ValueError(
            "Random Forest and TGN contain different numbers "
            "of test transactions.\n"
            f"RF: {len(rf_ids)}\n"
            f"TGN: {len(tgn_ids)}"
        )

    if set(rf_ids) != set(tgn_ids):

        rf_only = sorted(
            set(rf_ids) - set(tgn_ids)
        )

        tgn_only = sorted(
            set(tgn_ids) - set(rf_ids)
        )

        raise ValueError(
            "The Random Forest and TGN test sets do not match.\n"
            f"RF-only transactions: {rf_only[:10]}\n"
            f"TGN-only transactions: {tgn_only[:10]}"
        )

    print()
    print("Test-set verification:")
    print(
        f"  RF test transactions:  {len(rf_ids)}"
    )
    print(
        f"  TGN test transactions: {len(tgn_ids)}"
    )
    print("  Same transaction IDs:  True")


# ---------------------------------------------------------------------
# Prediction alignment
# ---------------------------------------------------------------------

def align_predictions(
    rf: pd.DataFrame,
    tgn: pd.DataFrame,
) -> pd.DataFrame:
    """
    Align Random Forest and TGN predictions by transaction_id.

    The merge is one-to-one so duplicate transaction IDs are rejected.
    """

    rf_columns = [
        "transaction_id",
        "prediction",
        "fraud_probability",
        "is_fraud",
        "typology",
    ]

    rf_aligned = rf[rf_columns].copy()

    rf_aligned = rf_aligned.rename(
        columns={
            "prediction": "rf_prediction",
            "fraud_probability": "rf_probability",
            "is_fraud": "rf_is_fraud",
            "typology": "rf_typology",
        }
    )

    tgn_columns = [
        "transaction_id",
        "prediction",
        "fraud_probability",
    ]

    if "is_fraud" in tgn.columns:
        tgn_columns.append("is_fraud")

    if "typology" in tgn.columns:
        tgn_columns.append("typology")

    tgn_aligned = tgn[tgn_columns].copy()

    tgn_aligned = tgn_aligned.rename(
        columns={
            "prediction": "tgn_prediction",
            "fraud_probability": "tgn_probability",
            "is_fraud": "tgn_is_fraud",
            "typology": "tgn_typology",
        }
    )

    merged = pd.merge(
        rf_aligned,
        tgn_aligned,
        on="transaction_id",
        how="inner",
        validate="one_to_one",
    )

    if len(merged) != len(rf):
        raise ValueError(
            "Alignment failed: merged test set contains "
            "a different number of transactions."
        )

    return merged


# ---------------------------------------------------------------------
# Ground truth
# ---------------------------------------------------------------------

def add_ground_truth(
    merged: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add a single ground-truth label and typology column.

    The Random Forest prediction file already contains the
    ground truth, so those values are used as the primary source.
    """

    merged = merged.copy()

    if "rf_is_fraud" in merged.columns:
        merged["is_fraud"] = (
            merged["rf_is_fraud"]
            .astype(int)
        )

    elif "tgn_is_fraud" in merged.columns:
        merged["is_fraud"] = (
            merged["tgn_is_fraud"]
            .astype(int)
        )

    else:
        raise ValueError(
            "Ground-truth is_fraud column is missing."
        )

    if "rf_typology" in merged.columns:
        merged["typology"] = (
            merged["rf_typology"]
            .astype(str)
        )

    elif "tgn_typology" in merged.columns:
        merged["typology"] = (
            merged["tgn_typology"]
            .astype(str)
        )

    else:
        merged["typology"] = "unknown"

    return merged


# ---------------------------------------------------------------------
# Model metrics
# ---------------------------------------------------------------------

def calculate_model_results(
    df: pd.DataFrame,
    model_name: str,
    prediction_column: str,
    probability_column: str,
):
    """
    Calculate standard binary classification metrics
    for one model.
    """

    y_true = (
        df["is_fraud"]
        .astype(int)
    )

    y_pred = (
        df[prediction_column]
        .astype(int)
    )

    y_probability = (
        df[probability_column]
        .astype(float)
    )

    metrics = calculate_binary_metrics(
        y_true=y_true,
        y_pred=y_pred,
        y_probability=y_probability,
    )

    confusion = calculate_confusion_matrix(
        y_true=y_true,
        y_pred=y_pred,
    )

    result = {
        "model": model_name,
        "test_transactions": len(df),
        "test_fraud": int(y_true.sum()),
        **metrics,
        **confusion,
    }

    return result


# ---------------------------------------------------------------------
# Typology comparison
# ---------------------------------------------------------------------

def calculate_typology_comparison(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compare fraud detection by typology.

    This reports detection/recall within each fraud typology.
    It is not a one-vs-rest precision/F1 calculation.
    """

    # -------------------------------------------------------------
    # Random Forest
    # -------------------------------------------------------------

    rf_input = df[
        [
            "is_fraud",
            "typology",
            "rf_prediction",
        ]
    ].copy()

    rf_input = rf_input.rename(
        columns={
            "rf_prediction": "prediction"
        }
    )

    rf_typology = calculate_typology_detection(
        rf_input,
        label_column="is_fraud",
        typology_column="typology",
        prediction_column="prediction",
    )

    rf_typology = rf_typology.rename(
        columns={
            "detected": "rf_detected",
            "recall": "rf_recall",
        }
    )

    rf_typology = rf_typology[
        [
            "typology",
            "total",
            "rf_detected",
            "rf_recall",
        ]
    ]

    # -------------------------------------------------------------
    # TemporalMemoryTGN
    # -------------------------------------------------------------

    tgn_input = df[
        [
            "is_fraud",
            "typology",
            "tgn_prediction",
        ]
    ].copy()

    tgn_input = tgn_input.rename(
        columns={
            "tgn_prediction": "prediction"
        }
    )

    tgn_typology = calculate_typology_detection(
        tgn_input,
        label_column="is_fraud",
        typology_column="typology",
        prediction_column="prediction",
    )

    tgn_typology = tgn_typology.rename(
        columns={
            "detected": "tgn_detected",
            "recall": "tgn_recall",
        }
    )

    tgn_typology = tgn_typology[
        [
            "typology",
            "tgn_detected",
            "tgn_recall",
        ]
    ]

    # -------------------------------------------------------------
    # Merge
    # -------------------------------------------------------------

    result = pd.merge(
        rf_typology,
        tgn_typology,
        on="typology",
        how="outer",
    )

    result["rf_missed"] = (
        result["total"]
        - result["rf_detected"]
    )

    result["tgn_missed"] = (
        result["total"]
        - result["tgn_detected"]
    )

    result = result.sort_values(
        "typology"
    ).reset_index(drop=True)

    return result


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    """Run the complete model comparison."""

    print("=" * 70)
    print("AEGISGRAPH MODEL COMPARISON")
    print("=" * 70)

    # -------------------------------------------------------------
    # Load
    # -------------------------------------------------------------

    rf, tgn = load_predictions()

    # -------------------------------------------------------------
    # Normalize schemas
    # -------------------------------------------------------------

    rf = normalize_rf_predictions(rf)

    tgn = normalize_tgn_predictions(tgn)

    # -------------------------------------------------------------
    # Verify exact test-set identity
    # -------------------------------------------------------------

    verify_same_test_set(
        rf,
        tgn,
    )

    # -------------------------------------------------------------
    # Align predictions
    # -------------------------------------------------------------

    comparison_df = align_predictions(
        rf,
        tgn,
    )

    comparison_df = add_ground_truth(
        comparison_df
    )

    print()
    print("Aligned test set:")
    print(
        f"  Transactions: "
        f"{len(comparison_df)}"
    )

    print(
        f"  Fraud cases:  "
        f"{int(comparison_df['is_fraud'].sum())}"
    )

    # -------------------------------------------------------------
    # Calculate standard metrics
    # -------------------------------------------------------------

    rf_results = calculate_model_results(
        comparison_df,
        model_name="Random Forest",
        prediction_column="rf_prediction",
        probability_column="rf_probability",
    )

    tgn_results = calculate_model_results(
        comparison_df,
        model_name="TemporalMemoryTGN",
        prediction_column="tgn_prediction",
        probability_column="tgn_probability",
    )

    metrics_df = pd.DataFrame(
        [
            rf_results,
            tgn_results,
        ]
    )

    # -------------------------------------------------------------
    # Save standard comparison metrics
    # -------------------------------------------------------------

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    metrics_df.to_csv(
        COMPARISON_METRICS_FILE,
        index=False,
    )

    # -------------------------------------------------------------
    # Calculate typology comparison
    # -------------------------------------------------------------

    typology_df = calculate_typology_comparison(
        comparison_df
    )

    typology_df.to_csv(
        TYPOLOGY_FILE,
        index=False,
    )

    # -------------------------------------------------------------
    # Display standard metrics
    # -------------------------------------------------------------

    print()
    print("Model comparison:")

    display_columns = [
        "model",
        "test_transactions",
        "test_fraud",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "pr_auc",
        "false_positives",
        "false_negatives",
    ]

    print(
        metrics_df[
            display_columns
        ].to_string(index=False)
    )

    # -------------------------------------------------------------
    # Display typology results
    # -------------------------------------------------------------

    print()
    print("Fraud detection by typology:")

    print(
        typology_df.to_string(
            index=False
        )
    )

    # -------------------------------------------------------------
    # Saved files
    # -------------------------------------------------------------

    print()
    print("Saved:")
    print(
        f"  {COMPARISON_METRICS_FILE}"
    )
    print(
        f"  {TYPOLOGY_FILE}"
    )

    print()
    print("=" * 70)
    print("MODEL COMPARISON COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()