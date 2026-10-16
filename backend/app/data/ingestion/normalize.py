import sys
from pathlib import Path

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
    TRANSACTIONS_FILE,
    CLEAN_TRANSACTIONS_FILE,
    create_directories,
)


# ---------------------------------------------------------
# Normalize transactions
# ---------------------------------------------------------
def normalize_transactions(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Normalize transaction data before downstream processing.

    Operations:
    - Normalize timestamps
    - Normalize numeric fields
    - Normalize IDs
    - Normalize categorical fields
    - Sort transactions chronologically
    """

    print(
        "\n========== NORMALIZATION =========="
    )

    df = df.copy()

    # -----------------------------------------------------
    # Column names
    # -----------------------------------------------------
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
    )

    # -----------------------------------------------------
    # Timestamp
    # -----------------------------------------------------
    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        errors="coerce",
    )

    # -----------------------------------------------------
    # Transaction amount
    # -----------------------------------------------------
    df["amount"] = pd.to_numeric(
        df["amount"],
        errors="coerce",
    ).round(2)

    # -----------------------------------------------------
    # Balance fields
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
        ).round(2)

    # -----------------------------------------------------
    # Transaction ID
    # -----------------------------------------------------
    df["transaction_id"] = (
        df["transaction_id"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    # -----------------------------------------------------
    # Account IDs
    # -----------------------------------------------------
    df["sender_account"] = (
        df["sender_account"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    df["receiver_account"] = (
        df["receiver_account"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    # -----------------------------------------------------
    # Transaction type
    # -----------------------------------------------------
    df["transaction_type"] = (
        df["transaction_type"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    # -----------------------------------------------------
    # Typology
    #
    # Keep this lowercase because validate.py and
    # inject_typologies.py use lowercase labels.
    # -----------------------------------------------------
    df["typology"] = (
        df["typology"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    # -----------------------------------------------------
    # Currency
    # -----------------------------------------------------
    df["currency"] = (
        df["currency"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    # -----------------------------------------------------
    # Channel
    # -----------------------------------------------------
    df["channel"] = (
        df["channel"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    # -----------------------------------------------------
    # Status
    # -----------------------------------------------------
    df["status"] = (
        df["status"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    # -----------------------------------------------------
    # Countries
    # -----------------------------------------------------
    df["sender_country"] = (
        df["sender_country"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    df["receiver_country"] = (
        df["receiver_country"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    # -----------------------------------------------------
    # Fraud label
    # -----------------------------------------------------
    df["is_fraud"] = pd.to_numeric(
        df["is_fraud"],
        errors="coerce",
    ).astype("Int64")

    # -----------------------------------------------------
    # Sort chronologically
    # -----------------------------------------------------
    df = (
        df.sort_values(
            "timestamp"
        )
        .reset_index(drop=True)
    )

    # -----------------------------------------------------
    # Create required directories
    # -----------------------------------------------------
    create_directories()

    # -----------------------------------------------------
    # Save normalized dataset
    # -----------------------------------------------------
    df.to_csv(
        CLEAN_TRANSACTIONS_FILE,
        index=False,
    )

    # -----------------------------------------------------
    # Summary
    # -----------------------------------------------------
    print(
        f"Normalized {len(df)} transactions."
    )

    print(
        f"Saved to: {CLEAN_TRANSACTIONS_FILE}"
    )

    print(
        f"Fraud transactions: "
        f"{(df['is_fraud'] == 1).sum()}"
    )

    print(
        f"Normal transactions: "
        f"{(df['is_fraud'] == 0).sum()}"
    )

    print(
        "\nTypology distribution:"
    )

    print(
        df["typology"]
        .value_counts()
        .sort_index()
    )

    print(
        "\n========== NORMALIZATION COMPLETE ==========\n"
    )

    return df


# ---------------------------------------------------------
# Script entry point
# ---------------------------------------------------------
if __name__ == "__main__":

    df = pd.read_csv(
        TRANSACTIONS_FILE
    )

    normalized = normalize_transactions(
        df
    )

    print(
        "\nFirst 5 normalized transactions:"
    )

    print(
        normalized.head()
    )