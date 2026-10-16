import sys
from pathlib import Path

import numpy as np
import pandas as pd

BACKEND_DIR = Path(__file__).resolve().parents[3]
sys.path.append(str(BACKEND_DIR))

from app.utils.config import (
    ACCOUNTS_FILE,
    TRANSACTIONS_FILE,
    create_directories,
)


def generate_transactions(
    num_transactions: int = 10000,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate synthetic normal banking transactions.

    Fraudulent transactions are NOT generated here.
    They are injected separately by inject_typologies.py.
    """

    np.random.seed(seed)

    accounts = pd.read_csv(
        ACCOUNTS_FILE
    )

    account_ids = (
        accounts["account_id"]
        .tolist()
    )

    account_country = dict(
        zip(
            accounts["account_id"],
            accounts["country"],
        )
    )

    account_balance = dict(
        zip(
            accounts["account_id"],
            accounts["current_balance"],
        )
    )

    records = []

    start_date = pd.Timestamp(
        "2026-01-01"
    )

    end_date = pd.Timestamp(
        "2026-06-30"
    )

    transaction_types = [
        "TRANSFER",
        "UPI",
        "NEFT",
        "IMPS",
    ]

    channels = [
        "MOBILE",
        "ONLINE",
        "BRANCH",
        "ATM",
    ]

    statuses = [
        "COMPLETED",
        "COMPLETED",
        "COMPLETED",
        "FAILED",
    ]

    total_seconds = int(
        (
            end_date - start_date
        ).total_seconds()
    )

    for i in range(
        1,
        num_transactions + 1,
    ):

        sender = np.random.choice(
            account_ids
        )

        receiver = np.random.choice(
            account_ids
        )

        # Prevent self-transfer
        while receiver == sender:

            receiver = np.random.choice(
                account_ids
            )

        amount = round(
            np.random.lognormal(
                mean=7.5,
                sigma=1.0,
            ),
            2,
        )

        amount = min(
            amount,
            100000,
        )

        timestamp = (
            start_date
            + pd.Timedelta(
                seconds=np.random.randint(
                    0,
                    total_seconds,
                )
            )
        )

        sender_balance_before = (
            account_balance[sender]
        )

        receiver_balance_before = (
            account_balance[receiver]
        )

        status = np.random.choice(
            statuses
        )

        # Only completed transactions
        # modify the simulated balances.
        if status == "COMPLETED":

            sender_balance_after = max(
                0,
                sender_balance_before
                - amount,
            )

            receiver_balance_after = (
                receiver_balance_before
                + amount
            )

            account_balance[sender] = (
                sender_balance_after
            )

            account_balance[receiver] = (
                receiver_balance_after
            )

        else:

            sender_balance_after = (
                sender_balance_before
            )

            receiver_balance_after = (
                receiver_balance_before
            )

        records.append({
            "transaction_id":
                f"T{i:07d}",

            "timestamp":
                timestamp,

            "sender_account":
                sender,

            "receiver_account":
                receiver,

            "amount":
                amount,

            "currency":
                "INR",

            "transaction_type":
                np.random.choice(
                    transaction_types
                ),

            "channel":
                np.random.choice(
                    channels
                ),

            "sender_country":
                account_country[sender],

            "receiver_country":
                account_country[receiver],

            "sender_balance_before":
                round(
                    sender_balance_before,
                    2,
                ),

            "sender_balance_after":
                round(
                    sender_balance_after,
                    2,
                ),

            "receiver_balance_before":
                round(
                    receiver_balance_before,
                    2,
                ),

            "receiver_balance_after":
                round(
                    receiver_balance_after,
                    2,
                ),

            "status":
                status,

            # Ground truth
            "is_fraud":
                0,

            "typology":
                "normal",
        })

    df = pd.DataFrame(records)

    create_directories()

    df.to_csv(
        TRANSACTIONS_FILE,
        index=False,
    )

    print(
        f"Generated {len(df)} "
        f"normal transactions."
    )

    print(
        f"Saved to: {TRANSACTIONS_FILE}"
    )

    return df


if __name__ == "__main__":
    generate_transactions()