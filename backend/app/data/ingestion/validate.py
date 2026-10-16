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
from app.utils.config import TRANSACTIONS_FILE


# ---------------------------------------------------------
# Required transaction columns
# ---------------------------------------------------------
REQUIRED_COLUMNS = [
    "transaction_id",
    "timestamp",
    "sender_account",
    "receiver_account",
    "amount",
    "currency",
    "transaction_type",
    "channel",
    "sender_country",
    "receiver_country",
    "sender_balance_before",
    "sender_balance_after",
    "receiver_balance_before",
    "receiver_balance_after",
    "status",
    "is_fraud",
    "typology",
]


# ---------------------------------------------------------
# Allowed values
# ---------------------------------------------------------
ALLOWED_FRAUD_LABELS = {0, 1}

ALLOWED_TYPOLOGIES = {
    "normal",
    "structuring",
    "rapid_movement",
    "fan_out",
    "circular",
}

ALLOWED_STATUSES = {
    "COMPLETED",
    "FAILED",
}

ALLOWED_CURRENCIES = {
    "INR",
    "USD",
    "GBP",
    "SGD",
    "AED",
}

ALLOWED_TRANSACTION_TYPES = {
    "TRANSFER",
    "UPI",
    "NEFT",
    "IMPS",
}

ALLOWED_CHANNELS = {
    "MOBILE",
    "ONLINE",
    "BRANCH",
    "ATM",
}


# ---------------------------------------------------------
# Validation function
# ---------------------------------------------------------
def validate_transactions(
    df: pd.DataFrame,
) -> bool:
    """
    Validate the transaction dataset.

    Checks:
    - Required columns
    - Missing values
    - Positive transaction amounts
    - Valid fraud labels
    - Valid typologies
    - Fraud/typology consistency
    - Self-transactions
    - Duplicate transaction IDs
    - Valid timestamps
    - Valid categorical values
    - Balance consistency
    """

    print("\n========== TRANSACTION VALIDATION ==========")

    validation_passed = True

    # -----------------------------------------------------
    # 1. Check required columns
    # -----------------------------------------------------
    missing_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing_columns:
        print(
            "\n❌ Missing columns:",
            missing_columns,
        )
        return False

    print("\n✓ Required columns present")

    # -----------------------------------------------------
    # 2. Check missing values
    # -----------------------------------------------------
    missing_values = (
        df[REQUIRED_COLUMNS]
        .isnull()
        .sum()
    )

    total_missing = missing_values.sum()

    print("\nMissing values:")

    if total_missing == 0:
        print("✓ No missing values")
    else:
        print(missing_values[missing_values > 0])
        validation_passed = False

    # -----------------------------------------------------
    # 3. Validate transaction IDs
    # -----------------------------------------------------
    duplicate_ids = (
        df["transaction_id"]
        .duplicated()
        .sum()
    )

    print(
        f"\nDuplicate transaction IDs: "
        f"{duplicate_ids}"
    )

    if duplicate_ids > 0:
        validation_passed = False
    else:
        print("✓ Transaction IDs are unique")

    # -----------------------------------------------------
    # 4. Validate timestamps
    # -----------------------------------------------------
    timestamps = pd.to_datetime(
        df["timestamp"],
        errors="coerce",
    )

    invalid_timestamps = timestamps.isna().sum()

    print(
        f"\nInvalid timestamps: "
        f"{invalid_timestamps}"
    )

    if invalid_timestamps > 0:
        validation_passed = False
    else:
        print("✓ Timestamps are valid")

    # -----------------------------------------------------
    # 5. Validate transaction amounts
    # -----------------------------------------------------
    invalid_amounts = (
        pd.to_numeric(
            df["amount"],
            errors="coerce",
        ).isna()
        | (df["amount"] <= 0)
    ).sum()

    print(
        f"\nInvalid amounts: "
        f"{invalid_amounts}"
    )

    if invalid_amounts > 0:
        validation_passed = False
    else:
        print("✓ Transaction amounts are valid")

    # -----------------------------------------------------
    # 6. Validate fraud labels
    # -----------------------------------------------------
    invalid_labels = (
        ~df["is_fraud"].isin(
            ALLOWED_FRAUD_LABELS
        )
    ).sum()

    print(
        f"\nInvalid fraud labels: "
        f"{invalid_labels}"
    )

    if invalid_labels > 0:
        validation_passed = False
    else:
        print("✓ Fraud labels are valid")

    # -----------------------------------------------------
    # 7. Validate typologies
    # -----------------------------------------------------
    invalid_typologies = (
        ~df["typology"].isin(
            ALLOWED_TYPOLOGIES
        )
    ).sum()

    print(
        f"\nInvalid typologies: "
        f"{invalid_typologies}"
    )

    if invalid_typologies > 0:
        validation_passed = False
    else:
        print("✓ Typologies are valid")

    # -----------------------------------------------------
    # 8. Check fraud / typology consistency
    # -----------------------------------------------------
    invalid_normal_fraud = (
        (df["is_fraud"] == 0)
        & (df["typology"] != "normal")
    ).sum()

    invalid_fraud_normal = (
        (df["is_fraud"] == 1)
        & (df["typology"] == "normal")
    ).sum()

    typology_mismatches = (
        invalid_normal_fraud
        + invalid_fraud_normal
    )

    print(
        f"\nFraud/typology mismatches: "
        f"{typology_mismatches}"
    )

    if typology_mismatches > 0:
        validation_passed = False
    else:
        print(
            "✓ Fraud labels and typologies are consistent"
        )

    # -----------------------------------------------------
    # 9. Check self-transactions
    # -----------------------------------------------------
    self_transactions = (
        df["sender_account"]
        == df["receiver_account"]
    ).sum()

    print(
        f"\nSelf transactions: "
        f"{self_transactions}"
    )

    if self_transactions > 0:
        validation_passed = False
    else:
        print("✓ No self-transactions")

    # -----------------------------------------------------
    # 10. Validate categorical fields
    # -----------------------------------------------------
    invalid_statuses = (
        ~df["status"].isin(
            ALLOWED_STATUSES
        )
    ).sum()

    invalid_currencies = (
        ~df["currency"].isin(
            ALLOWED_CURRENCIES
        )
    ).sum()

    invalid_transaction_types = (
        ~df["transaction_type"].isin(
            ALLOWED_TRANSACTION_TYPES
        )
    ).sum()

    invalid_channels = (
        ~df["channel"].isin(
            ALLOWED_CHANNELS
        )
    ).sum()

    print(
        f"\nInvalid statuses: "
        f"{invalid_statuses}"
    )

    print(
        f"Invalid currencies: "
        f"{invalid_currencies}"
    )

    print(
        f"Invalid transaction types: "
        f"{invalid_transaction_types}"
    )

    print(
        f"Invalid channels: "
        f"{invalid_channels}"
    )

    if (
        invalid_statuses > 0
        or invalid_currencies > 0
        or invalid_transaction_types > 0
        or invalid_channels > 0
    ):
        validation_passed = False
    else:
        print(
            "✓ Categorical transaction fields are valid"
        )

    # -----------------------------------------------------
    # 11. Validate balance fields
    # -----------------------------------------------------
    balance_columns = [
        "sender_balance_before",
        "sender_balance_after",
        "receiver_balance_before",
        "receiver_balance_after",
    ]

    invalid_balances = 0

    for column in balance_columns:

        numeric_values = pd.to_numeric(
            df[column],
            errors="coerce",
        )

        invalid_balances += (
            numeric_values.isna()
            | (numeric_values < 0)
        ).sum()

    print(
        f"\nInvalid balance values: "
        f"{invalid_balances}"
    )

    if invalid_balances > 0:
        validation_passed = False
    else:
        print("✓ Balance values are valid")

    # -----------------------------------------------------
    # 12. Validate account IDs
    # -----------------------------------------------------
    missing_sender_accounts = (
        df["sender_account"]
        .astype(str)
        .str.strip()
        .eq("")
        .sum()
    )

    missing_receiver_accounts = (
        df["receiver_account"]
        .astype(str)
        .str.strip()
        .eq("")
        .sum()
    )

    print(
        f"\nMissing sender accounts: "
        f"{missing_sender_accounts}"
    )

    print(
        f"Missing receiver accounts: "
        f"{missing_receiver_accounts}"
    )

    if (
        missing_sender_accounts > 0
        or missing_receiver_accounts > 0
    ):
        validation_passed = False
    else:
        print("✓ Account references are present")

    # -----------------------------------------------------
    # 13. Dataset summary
    # -----------------------------------------------------
    fraud_count = (
        df["is_fraud"] == 1
    ).sum()

    normal_count = (
        df["is_fraud"] == 0
    ).sum()

    print("\n========== DATASET SUMMARY ==========")

    print(
        f"Total transactions : {len(df)}"
    )

    print(
        f"Normal transactions: {normal_count}"
    )

    print(
        f"Fraud transactions : {fraud_count}"
    )

    if len(df) > 0:
        fraud_percentage = (
            fraud_count / len(df)
        ) * 100

        print(
            f"Fraud percentage   : "
            f"{fraud_percentage:.2f}%"
        )

    # -----------------------------------------------------
    # 14. Typology distribution
    # -----------------------------------------------------
    print("\nTypology distribution:")

    print(
        df["typology"]
        .value_counts()
        .sort_index()
    )

    # -----------------------------------------------------
    # Final result
    # -----------------------------------------------------
    print("\n=====================================")

    if validation_passed:
        print("✅ VALIDATION PASSED")
    else:
        print("❌ VALIDATION FAILED")

    print("=====================================\n")

    return validation_passed


# ---------------------------------------------------------
# Script entry point
# ---------------------------------------------------------
if __name__ == "__main__":

    print(
        f"Loading transactions from:\n"
        f"{TRANSACTIONS_FILE}"
    )

    df = pd.read_csv(
        TRANSACTIONS_FILE
    )

    validate_transactions(df)