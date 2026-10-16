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
    Create account activity features using ONLY transactions
    that occurred before the current transaction.

    This prevents future/test information from leaking into
    the current transaction's feature vector.
    """

    print(
        "\nCreating temporal account features..."
    )

    # Ensure chronological order
    df = df.sort_values(
        "timestamp"
    ).reset_index(drop=True)

    # -----------------------------------------------------
    # Sender historical activity
    # -----------------------------------------------------

    # Number of previous transactions sent
    df["sender_out_degree"] = (
        df.groupby("sender_account")
        .cumcount()
    )

    # Total amount previously sent
    df["sender_total_outgoing"] = (
        df.groupby("sender_account")["amount"]
        .cumsum()
        .sub(df["amount"])
    )

    # Number of unique receivers previously used
    df["sender_unique_neighbors"] = (
        df.groupby("sender_account")[
            "receiver_account"
        ]
        .transform(
            lambda x:
            x.shift()
            .groupby(x.shift())
            .transform("nunique")
        )
    )

    # -----------------------------------------------------
    # Receiver historical activity
    # -----------------------------------------------------

    # Number of previous transactions received
    df["receiver_in_degree"] = (
        df.groupby("receiver_account")
        .cumcount()
    )

    # Total amount previously received
    df["receiver_total_incoming"] = (
        df.groupby("receiver_account")["amount"]
        .cumsum()
        .sub(df["amount"])
    )

    # -----------------------------------------------------
    # Fill missing historical values
    # -----------------------------------------------------

    temporal_columns = [
        "sender_out_degree",
        "sender_total_outgoing",
        "sender_unique_neighbors",
        "receiver_in_degree",
        "receiver_total_incoming",
    ]

    for column in temporal_columns:
        df[column] = (
            pd.to_numeric(
                df[column],
                errors="coerce",
            )
            .fillna(0)
        )

    return df


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

    # -----------------------------------------------------
    # Basic cleanup
    # -----------------------------------------------------

    df = transactions.copy()

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        errors="coerce",
    )

    df["amount"] = pd.to_numeric(
        df["amount"],
        errors="coerce",
    )

    # -----------------------------------------------------
    # Sort chronologically
    # -----------------------------------------------------

    df = (
        df.sort_values("timestamp")
        .reset_index(drop=True)
    )

    # -----------------------------------------------------
    # Time features
    # -----------------------------------------------------

    df["hour"] = (
        df["timestamp"].dt.hour
    )

    df["day_of_week"] = (
        df["timestamp"].dt.dayofweek
    )

    df["day_of_month"] = (
        df["timestamp"].dt.day
    )

    df["month"] = (
        df["timestamp"].dt.month
    )

    df["is_weekend"] = (
        df["timestamp"].dt.dayofweek >= 5
    ).astype(int)

    # -----------------------------------------------------
    # Amount features
    # -----------------------------------------------------

    df["log_amount"] = np.log1p(
        df["amount"].clip(lower=0)
    )

    # -----------------------------------------------------
    # Balance features
    # -----------------------------------------------------

    balance_columns = [
        "sender_balance_before",
        "sender_balance_after",
        "receiver_balance_before",
        "receiver_balance_after",
    ]

    for column in balance_columns:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    # -----------------------------------------------------
    # Amount-to-balance ratios
    # -----------------------------------------------------

    df["amount_to_sender_balance"] = (
        df["amount"]
        / df["sender_balance_before"]
        .replace(0, np.nan)
    )

    df["amount_to_receiver_balance"] = (
        df["amount"]
        / df["receiver_balance_before"]
        .replace(0, np.nan)
    )

    # -----------------------------------------------------
    # Temporal account features
    # -----------------------------------------------------

    df = create_temporal_account_features(
        df
    )

    # -----------------------------------------------------
    # Historical interaction features
    # -----------------------------------------------------

    df["sender_activity_ratio"] = (
        df["sender_total_outgoing"]
        / (
            df["sender_out_degree"] + 1
        )
    )

    df["receiver_activity_ratio"] = (
        df["receiver_total_incoming"]
        / (
            df["receiver_in_degree"] + 1
        )
    )

    df["sender_receiver_degree_ratio"] = (
        (
            df["sender_out_degree"] + 1
        )
        /
        (
            df["receiver_in_degree"] + 1
        )
    )

    # -----------------------------------------------------
    # Transaction categorical features
    # -----------------------------------------------------

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

    # -----------------------------------------------------
    # Fill numeric missing values
    # -----------------------------------------------------

    numeric_columns = (
        df.select_dtypes(
            include=["number"]
        ).columns
    )

    df[numeric_columns] = (
        df[numeric_columns]
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
        .fillna(0)
    )

    # -----------------------------------------------------
    # Separate model inputs from alignment/evaluation data
    # -----------------------------------------------------
    #
    # transaction_id and timestamp are retained in the
    # output file so that Random Forest predictions can be
    # aligned exactly with the TGN test transactions.
    #
    # They are NOT included in feature_columns and therefore
    # must never be used as Random Forest model inputs.
    #
    # The following fields are also excluded from model input.
    # -----------------------------------------------------

    non_model_columns = [
        "transaction_id",
        "timestamp",
        "sender_account",
        "receiver_account",
        "sender_country",
        "receiver_country",
        "status",
        "typology",
        "is_fraud",
    ]

    feature_columns = [
        column
        for column in df.columns
        if column not in non_model_columns
    ]

    # -----------------------------------------------------
    # Build final dataset
    # -----------------------------------------------------
    #
    # Keep transaction_id and timestamp only for alignment
    # and evaluation. They are not model features.
    # -----------------------------------------------------

    feature_df = df[
        [
            "transaction_id",
            "timestamp",
        ]
        + feature_columns
        + [
            "is_fraud",
            "typology",
        ]
    ].copy()

    # -----------------------------------------------------
    # Convert boolean columns if any
    # -----------------------------------------------------

    for column in feature_df.columns:

        if feature_df[column].dtype == "bool":
            feature_df[column] = (
                feature_df[column]
                .astype(int)
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