import sys
from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------
# Add backend directory to Python path
# ---------------------------------------------------------
BACKEND_DIR = Path(__file__).resolve().parents[3]
sys.path.append(str(BACKEND_DIR))


# ---------------------------------------------------------
# Project configuration
# ---------------------------------------------------------
from app.utils.config import (
    CLEAN_TRANSACTIONS_FILE,
    RESULTS_DATA_DIR,
    NODE_FEATURES_FILE,
    create_directories,
)


# ---------------------------------------------------------
# Output file
# ---------------------------------------------------------
OUTPUT_FILE = (
    RESULTS_DATA_DIR / "baseline_features.csv"
)

NON_MODEL_COLUMNS = {
    "transaction_id",
    "timestamp",
    "sender_account",
    "receiver_account",
    "sender_country",
    "receiver_country",
    "status",
    "typology",
    "is_fraud",
    "sender_balance_after",
    "receiver_balance_after",
}


# ---------------------------------------------------------
# Load data
# ---------------------------------------------------------
def load_data():

    print("Loading transaction data...")

    transactions = pd.read_csv(
        CLEAN_TRANSACTIONS_FILE
    )

    transactions["timestamp"] = pd.to_datetime(
        transactions["timestamp"],
        errors="coerce",
    )

    print("Loading graph features...")

    node_features = pd.read_csv(
        NODE_FEATURES_FILE
    )

    return transactions, node_features


# ---------------------------------------------------------
# Create temporal account features
# ---------------------------------------------------------
def create_temporal_account_features(df):
    """
    Create account activity features from strictly earlier timestamps.

    Transactions at the same timestamp are scored from the same prior
    history snapshot; the batch is added to history only after scoring.
    """

    print(
        "\nCreating temporal account features..."
    )

    df = df.sort_values(
        "timestamp",
        kind="mergesort",
    ).reset_index(drop=True).copy()
    amounts = pd.to_numeric(
        df["amount"],
        errors="coerce",
    ).fillna(0)

    sender_counts: dict[object, int] = {}
    sender_amounts: dict[object, float] = {}
    sender_receivers: dict[object, set] = {}
    receiver_counts: dict[object, int] = {}
    receiver_amounts: dict[object, float] = {}

    sender_out_degree = [0] * len(df)
    sender_total_outgoing = [0.0] * len(df)
    sender_unique_neighbors = [0] * len(df)
    receiver_in_degree = [0] * len(df)
    receiver_total_incoming = [0.0] * len(df)

    for positions in df.groupby(
        "timestamp",
        sort=False,
    ).indices.values():
        batch = df.iloc[positions]

        for position, row in zip(positions, batch.itertuples(index=False)):
            sender = getattr(row, "sender_account")
            receiver = getattr(row, "receiver_account")

            if pd.notna(sender):
                sender_out_degree[position] = sender_counts.get(
                    sender,
                    0,
                )
                sender_total_outgoing[position] = sender_amounts.get(
                    sender,
                    0.0,
                )
                sender_unique_neighbors[position] = len(
                    sender_receivers.get(sender, set())
                )

            if pd.notna(receiver):
                receiver_in_degree[position] = receiver_counts.get(
                    receiver,
                    0,
                )
                receiver_total_incoming[position] = receiver_amounts.get(
                    receiver,
                    0.0,
                )

        for position, row in zip(positions, batch.itertuples(index=False)):
            sender = getattr(row, "sender_account")
            receiver = getattr(row, "receiver_account")
            amount = float(amounts.iloc[position])

            if pd.notna(sender):
                sender_counts[sender] = sender_counts.get(sender, 0) + 1
                sender_amounts[sender] = (
                    sender_amounts.get(sender, 0.0) + amount
                )
                if pd.notna(receiver):
                    sender_receivers.setdefault(sender, set()).add(receiver)

            if pd.notna(receiver):
                receiver_counts[receiver] = (
                    receiver_counts.get(receiver, 0) + 1
                )
                receiver_amounts[receiver] = (
                    receiver_amounts.get(receiver, 0.0) + amount
                )

    temporal_columns = [
        "sender_out_degree",
        "sender_total_outgoing",
        "sender_unique_neighbors",
        "receiver_in_degree",
        "receiver_total_incoming",
    ]
    temporal_values = [
        sender_out_degree,
        sender_total_outgoing,
        sender_unique_neighbors,
        receiver_in_degree,
        receiver_total_incoming,
    ]
    for column, values in zip(temporal_columns, temporal_values):
        df[column] = values

    return df


def chronological_split_index(
    sorted_transactions: pd.DataFrame,
    train_ratio: float = 0.80,
) -> int:
    """Return the nearest chronological split that keeps timestamp ties intact."""

    if not 0 < train_ratio < 1:
        raise ValueError("train_ratio must be between 0 and 1.")
    if "timestamp" not in sorted_transactions:
        raise ValueError("Chronological split requires a timestamp column.")

    timestamps = pd.to_datetime(
        sorted_transactions["timestamp"],
        errors="raise",
    )
    if timestamps.isna().any():
        raise ValueError("Chronological split cannot include missing timestamps.")
    if not timestamps.is_monotonic_increasing:
        raise ValueError("Transactions must be sorted by timestamp before splitting.")

    target = int(len(sorted_transactions) * train_ratio)
    valid_boundaries = [
        index
        for index in range(1, len(sorted_transactions))
        if timestamps.iloc[index - 1] < timestamps.iloc[index]
    ]
    if not valid_boundaries:
        raise ValueError(
            "Cannot split transactions without dividing a timestamp batch."
        )

    return min(
        valid_boundaries,
        key=lambda index: (
            abs(index - target),
            -index,
        ),
    )


def build_feature_dataset(
    transactions: pd.DataFrame,
    feature_columns: list[str] | None = None,
) -> pd.DataFrame:
    """Create labeled, chronologically aligned transaction features."""

    df = transactions.copy()

    if "is_fraud" not in df.columns:
        df["is_fraud"] = 0
    if "typology" not in df.columns:
        df["typology"] = "unknown"

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        errors="raise",
    )
    df["amount"] = pd.to_numeric(
        df["amount"],
        errors="coerce",
    )

    df = (
        df.sort_values("timestamp", kind="mergesort")
        .reset_index(drop=True)
    )

    df["hour"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["day_of_month"] = df["timestamp"].dt.day
    df["month"] = df["timestamp"].dt.month
    df["is_weekend"] = (
        df["timestamp"].dt.dayofweek >= 5
    ).astype(int)
    df["log_amount"] = np.log1p(
        df["amount"].clip(lower=0)
    )

    balance_columns = [
        "sender_balance_before",
        "receiver_balance_before",
    ]
    for column in balance_columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    df["amount_to_sender_balance"] = (
        df["amount"]
        / df["sender_balance_before"].replace(0, np.nan)
    )
    df["amount_to_receiver_balance"] = (
        df["amount"]
        / df["receiver_balance_before"].replace(0, np.nan)
    )

    df = create_temporal_account_features(df)

    df["sender_activity_ratio"] = (
        df["sender_total_outgoing"]
        / (df["sender_out_degree"] + 1)
    )
    df["receiver_activity_ratio"] = (
        df["receiver_total_incoming"]
        / (df["receiver_in_degree"] + 1)
    )
    df["sender_receiver_degree_ratio"] = (
        (df["sender_out_degree"] + 1)
        / (df["receiver_in_degree"] + 1)
    )

    categorical_columns = [
        "channel",
        "transaction_type",
        "currency",
    ]
    df = pd.get_dummies(
        df,
        columns=categorical_columns,
        prefix=categorical_columns,
        dtype=int,
    )

    numeric_columns = df.select_dtypes(
        include=["number"]
    ).columns
    df[numeric_columns] = (
        df[numeric_columns]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0)
    )

    model_columns = [
        column
        for column in df.columns
        if column not in NON_MODEL_COLUMNS
    ]
    if feature_columns is None:
        feature_columns = model_columns
    else:
        forbidden = NON_MODEL_COLUMNS.intersection(feature_columns)
        if forbidden:
            raise ValueError(
                "RF feature schema contains excluded columns: "
                f"{sorted(forbidden)}"
            )

    alignment_columns = [
        column
        for column in ("transaction_id", "timestamp")
        if column in df.columns
    ]
    target_columns = [
        column
        for column in ("is_fraud", "typology")
        if column in df.columns
    ]
    feature_df = df[
        alignment_columns
        + feature_columns
        + target_columns
    ].copy()

    for column in feature_df.columns:
        if feature_df[column].dtype == "bool":
            feature_df[column] = feature_df[column].astype(int)

    return feature_df


def build_features(
    transactions: pd.DataFrame,
    feature_columns: list[str] | None = None,
) -> pd.DataFrame:
    """Build RF inputs, optionally aligned to a saved training schema."""

    feature_df = build_feature_dataset(transactions)
    model_columns = [
        column
        for column in feature_df.columns
        if column not in NON_MODEL_COLUMNS
    ]

    if feature_columns is None:
        feature_columns = model_columns

    forbidden = NON_MODEL_COLUMNS.intersection(feature_columns)
    if forbidden:
        raise ValueError(
            "RF feature schema contains excluded columns: "
            f"{sorted(forbidden)}"
        )

    return feature_df.reindex(
        columns=feature_columns,
        fill_value=0,
    ).copy()


# ---------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------
def engineer_features():

    create_directories()

    transactions, node_features = (
        load_data()
    )

    print(
        "\n========== FEATURE ENGINEERING =========="
    )

    transactions["timestamp"] = pd.to_datetime(
        transactions["timestamp"],
        errors="raise",
    )
    transactions = (
        transactions.sort_values("timestamp", kind="mergesort")
        .reset_index(drop=True)
    )
    split_index = chronological_split_index(transactions)
    training_feature_df = build_feature_dataset(
        transactions.iloc[:split_index]
    )
    feature_columns = [
        column
        for column in training_feature_df.columns
        if column not in NON_MODEL_COLUMNS
    ]
    feature_df = build_feature_dataset(
        transactions,
        feature_columns=feature_columns,
    )

    # -----------------------------------------------------
    # Save
    # -----------------------------------------------------

    feature_df.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    # -----------------------------------------------------
    # Summary
    # -----------------------------------------------------

    print(
        f"\nTransactions processed: "
        f"{len(feature_df)}"
    )

    print(
        f"Total columns generated: "
        f"{len(feature_df.columns)}"
    )

    print(
        f"Model input features: "
        f"{len(feature_columns)}"
    )

    print(
        "\nAlignment columns retained "
        "for evaluation:"
    )

    print("  transaction_id")
    print("  timestamp")

    print(
        "\nGround-truth columns kept "
        "only for evaluation:"
    )

    print("  is_fraud")
    print("  typology")

    print(
        f"\nFeature dataset saved to:"
        f"\n{OUTPUT_FILE}"
    )

    print(
        "\n========== FEATURE ENGINEERING COMPLETE =========="
    )

    return feature_df


# ---------------------------------------------------------
# Script entry point
# ---------------------------------------------------------
if __name__ == "__main__":
    engineer_features()