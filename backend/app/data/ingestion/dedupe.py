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
    CLEAN_TRANSACTIONS_FILE,
    create_directories,
)


# ---------------------------------------------------------
# Deduplicate transactions
# ---------------------------------------------------------
def deduplicate_transactions(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Remove duplicate transaction records.

    The primary duplicate key is transaction_id.

    Transactions with the same sender, receiver, amount,
    and timestamp are NOT automatically removed if they
    have different transaction IDs, because they may
    represent legitimate repeated transactions.
    """

    print(
        "\n========== DEDUPLICATION =========="
    )

    df = df.copy()

    # -----------------------------------------------------
    # Record count before deduplication
    # -----------------------------------------------------
    before = len(df)

    # -----------------------------------------------------
    # Check exact duplicate rows
    # -----------------------------------------------------
    exact_duplicates = df.duplicated(
        keep="first"
    ).sum()

    print(
        f"Exact duplicate rows: "
        f"{exact_duplicates}"
    )

    # Remove completely identical rows
    if exact_duplicates > 0:
        df = df.drop_duplicates(
            keep="first"
        )

    # -----------------------------------------------------
    # Remove duplicate transaction IDs
    # -----------------------------------------------------
    duplicate_ids = (
        df["transaction_id"]
        .duplicated(keep="first")
        .sum()
    )

    print(
        f"Duplicate transaction IDs: "
        f"{duplicate_ids}"
    )

    if duplicate_ids > 0:
        df = df.drop_duplicates(
            subset=["transaction_id"],
            keep="first",
        )

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
    # Record count after deduplication
    # -----------------------------------------------------
    after = len(df)

    removed = before - after

    print(
        f"\nRecords before: "
        f"{before}"
    )

    print(
        f"Records after: "
        f"{after}"
    )

    print(
        f"Duplicates removed: "
        f"{removed}"
    )

    # -----------------------------------------------------
    # Create directories
    # -----------------------------------------------------
    create_directories()

    # -----------------------------------------------------
    # Save cleaned data
    # -----------------------------------------------------
    df.to_csv(
        CLEAN_TRANSACTIONS_FILE,
        index=False,
    )

    print(
        f"\nSaved clean data to:"
        f"\n{CLEAN_TRANSACTIONS_FILE}"
    )

    print(
        "\n========== DEDUPLICATION COMPLETE ==========\n"
    )

    return df


# ---------------------------------------------------------
# Script entry point
# ---------------------------------------------------------
if __name__ == "__main__":

    print(
        f"Loading normalized data from:"
        f"\n{CLEAN_TRANSACTIONS_FILE}"
    )

    df = pd.read_csv(
        CLEAN_TRANSACTIONS_FILE
    )

    deduplicate_transactions(df)