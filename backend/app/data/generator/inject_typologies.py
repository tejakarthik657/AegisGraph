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
    TRANSACTIONS_FILE,
    create_directories,
)


# ---------------------------------------------------------
# Fraud configuration
# ---------------------------------------------------------
RANDOM_SEED = 42

STRUCTURING_COUNT = 50
RAPID_MOVEMENT_COUNT = 50
FAN_OUT_COUNT = 50

# Keep 48 so circular fraud contains complete 3-event cycles.
# 50 requested events would otherwise produce an incomplete
# A -> B -> C -> A pattern.
CIRCULAR_COUNT = 48


# ---------------------------------------------------------
# Helper: create a fraud transaction
# ---------------------------------------------------------
def create_fraud_transaction(
    transaction_id,
    timestamp,
    sender_account,
    receiver_account,
    amount,
    currency,
    transaction_type,
    channel,
    sender_country,
    receiver_country,
    sender_balance_before,
    sender_balance_after,
    receiver_balance_before,
    receiver_balance_after,
    status,
    typology,
):
    """
    Create one fraud transaction using the same schema
    as the normal transaction dataset.
    """

    return {
        "transaction_id": transaction_id,
        "timestamp": timestamp,
        "sender_account": sender_account,
        "receiver_account": receiver_account,
        "amount": round(float(amount), 2),
        "currency": currency,
        "transaction_type": transaction_type,
        "channel": channel,
        "sender_country": sender_country,
        "receiver_country": receiver_country,
        "sender_balance_before": round(
            float(sender_balance_before), 2
        ),
        "sender_balance_after": round(
            float(sender_balance_after), 2
        ),
        "receiver_balance_before": round(
            float(receiver_balance_before), 2
        ),
        "receiver_balance_after": round(
            float(receiver_balance_after), 2
        ),
        "status": status,
        "is_fraud": 1,
        "typology": typology,
    }


# ---------------------------------------------------------
# Helper: generate timestamps across the full normal period
# ---------------------------------------------------------
def generate_distributed_timestamps(
    count,
    min_time,
    max_time,
    rng,
):
    """
    Generate timestamps distributed across the full
    normal transaction period.

    Unlike the previous implementation, each typology
    calls this function independently. Therefore one
    typology cannot consume the early timestamps while
    another typology receives only the late timestamps.
    """

    if count <= 0:
        return []

    if min_time >= max_time:
        raise ValueError(
            "Normal transaction timestamps do not contain "
            "a valid time range."
        )

    total_seconds = (
        max_time - min_time
    ).total_seconds()

    # Use evenly spaced positions across the complete
    # period and add bounded random jitter.
    #
    # This gives every typology events in both the early
    # and late portions of the chronological dataset.
    base_positions = np.linspace(
        0,
        total_seconds,
        num=count,
    )

    if count > 1:
        spacing = total_seconds / (count - 1)
        jitter_limit = spacing * 0.35

        jitter = rng.uniform(
            -jitter_limit,
            jitter_limit,
            size=count,
        )

        offsets = base_positions + jitter
        offsets[0] = max(0.0, offsets[0])
        offsets[-1] = min(
            total_seconds,
            offsets[-1],
        )
    else:
        offsets = np.array(
            [total_seconds / 2.0]
        )

    offsets = np.clip(
        offsets,
        0,
        total_seconds,
    )

    offsets.sort()

    timestamps = [
        min_time + pd.Timedelta(
            seconds=float(offset)
        )
        for offset in offsets
    ]

    return timestamps


# ---------------------------------------------------------
# Helper: generate circular cycle timestamps
# ---------------------------------------------------------
def generate_circular_timestamps(
    count,
    min_time,
    max_time,
    rng,
):
    """
    Generate timestamps for circular fraud cycles.

    Each cycle contains three consecutive events:

        A -> B
        B -> C
        C -> A

    Cycle start times are distributed across the complete
    normal transaction period, while the three events inside
    each cycle remain close together in time.
    """

    if count <= 0:
        return []

    if count % 3 != 0:
        raise ValueError(
            "Circular transaction count must be divisible "
            "by 3 so every cycle contains A -> B -> C -> A."
        )

    if min_time >= max_time:
        raise ValueError(
            "Normal transaction timestamps do not contain "
            "a valid time range."
        )

    number_of_cycles = count // 3

    total_seconds = (
        max_time - min_time
    ).total_seconds()

    # Keep each cycle comfortably inside the normal
    # transaction period.
    max_cycle_duration = 15 * 60

    usable_seconds = max(
        1.0,
        total_seconds - max_cycle_duration,
    )

    # Spread cycle starts throughout the entire period.
    base_positions = np.linspace(
        0,
        usable_seconds,
        num=number_of_cycles,
    )

    if number_of_cycles > 1:
        spacing = (
            usable_seconds
            / (number_of_cycles - 1)
        )

        jitter_limit = min(
            spacing * 0.25,
            6 * 60,
        )

        jitter = rng.uniform(
            -jitter_limit,
            jitter_limit,
            size=number_of_cycles,
        )

        cycle_starts = (
            base_positions + jitter
        )

        cycle_starts[0] = max(
            0.0,
            cycle_starts[0],
        )

        cycle_starts[-1] = min(
            usable_seconds,
            cycle_starts[-1],
        )
    else:
        cycle_starts = np.array(
            [usable_seconds / 2.0]
        )

    cycle_starts = np.clip(
        cycle_starts,
        0,
        usable_seconds,
    )

    cycle_starts.sort()

    timestamps = []

    for cycle_start in cycle_starts:

        # Three events occur within a short interval.
        first_gap = rng.uniform(
            30,
            120,
        )

        second_gap = rng.uniform(
            30,
            120,
        )

        offsets = [
            cycle_start,
            cycle_start + first_gap,
            cycle_start
            + first_gap
            + second_gap,
        ]

        for offset in offsets:

            offset = min(
                offset,
                total_seconds,
            )

            timestamps.append(
                min_time
                + pd.Timedelta(
                    seconds=float(offset)
                )
            )

    return timestamps


# ---------------------------------------------------------
# Structuring
# ---------------------------------------------------------
def generate_structuring_transactions(
    accounts,
    count,
    start_id,
    rng,
    fraud_timestamps,
):
    """
    Generate a simplified synthetic structuring/smurfing
    pattern.

    Transactions use amounts below the selected threshold
    and are distributed across the overall transaction
    timeline.
    """

    records = []

    selected_accounts = accounts.sample(
        n=min(count, len(accounts)),
        random_state=RANDOM_SEED,
    )

    for index, (_, sender) in enumerate(
        selected_accounts.iterrows()
    ):

        receiver = accounts[
            accounts["account_id"]
            != sender["account_id"]
        ].sample(
            n=1,
            random_state=RANDOM_SEED + index,
        ).iloc[0]

        amount = rng.uniform(
            8000,
            49000,
        )

        sender_balance_before = float(
            sender["current_balance"]
        )

        sender_balance_after = max(
            0,
            sender_balance_before - amount,
        )

        receiver_balance_before = float(
            receiver["current_balance"]
        )

        receiver_balance_after = (
            receiver_balance_before + amount
        )

        timestamp = fraud_timestamps[index]

        records.append(
            create_fraud_transaction(
                transaction_id=f"F{start_id + index:07d}",
                timestamp=timestamp,
                sender_account=sender["account_id"],
                receiver_account=receiver["account_id"],
                amount=amount,
                currency="INR",
                transaction_type="UPI",
                channel="MOBILE",
                sender_country=sender["country"],
                receiver_country=receiver["country"],
                sender_balance_before=sender_balance_before,
                sender_balance_after=sender_balance_after,
                receiver_balance_before=receiver_balance_before,
                receiver_balance_after=receiver_balance_after,
                status="COMPLETED",
                typology="structuring",
            )
        )

    return records


# ---------------------------------------------------------
# Rapid movement
# ---------------------------------------------------------
def generate_rapid_movement_transactions(
    accounts,
    count,
    start_id,
    rng,
    fraud_timestamps,
):
    """
    Generate transactions where money moves rapidly
    between accounts.

    Events are distributed across the overall timeline.
    """

    records = []

    for index in range(count):

        sender = accounts.sample(
            n=1,
            random_state=RANDOM_SEED + 100 + index,
        ).iloc[0]

        receiver = accounts[
            accounts["account_id"]
            != sender["account_id"]
        ].sample(
            n=1,
            random_state=RANDOM_SEED + 200 + index,
        ).iloc[0]

        amount = rng.uniform(
            50000,
            300000,
        )

        sender_balance_before = float(
            sender["current_balance"]
        )

        sender_balance_after = max(
            0,
            sender_balance_before - amount,
        )

        receiver_balance_before = float(
            receiver["current_balance"]
        )

        receiver_balance_after = (
            receiver_balance_before + amount
        )

        timestamp = fraud_timestamps[index]

        records.append(
            create_fraud_transaction(
                transaction_id=f"F{start_id + index:07d}",
                timestamp=timestamp,
                sender_account=sender["account_id"],
                receiver_account=receiver["account_id"],
                amount=amount,
                currency="INR",
                transaction_type="IMPS",
                channel="ONLINE",
                sender_country=sender["country"],
                receiver_country=receiver["country"],
                sender_balance_before=sender_balance_before,
                sender_balance_after=sender_balance_after,
                receiver_balance_before=receiver_balance_before,
                receiver_balance_after=receiver_balance_after,
                status="COMPLETED",
                typology="rapid_movement",
            )
        )

    return records


# ---------------------------------------------------------
# Fan-out
# ---------------------------------------------------------
def generate_fan_out_transactions(
    accounts,
    count,
    start_id,
    rng,
    fraud_timestamps,
):
    """
    Generate a fan-out pattern where one source account
    sends money to many different destination accounts.
    """

    records = []

    source = accounts.sample(
        n=1,
        random_state=RANDOM_SEED + 300,
    ).iloc[0]

    receiver_pool = accounts[
        accounts["account_id"]
        != source["account_id"]
    ].sample(
        n=min(count, len(accounts) - 1),
        random_state=RANDOM_SEED + 301,
    )

    for index, (_, receiver) in enumerate(
        receiver_pool.iterrows()
    ):

        amount = rng.uniform(
            10000,
            75000,
        )

        sender_balance_before = float(
            source["current_balance"]
        )

        sender_balance_after = max(
            0,
            sender_balance_before
            - amount * (index + 1),
        )

        receiver_balance_before = float(
            receiver["current_balance"]
        )

        receiver_balance_after = (
            receiver_balance_before + amount
        )

        timestamp = fraud_timestamps[index]

        records.append(
            create_fraud_transaction(
                transaction_id=f"F{start_id + index:07d}",
                timestamp=timestamp,
                sender_account=source["account_id"],
                receiver_account=receiver["account_id"],
                amount=amount,
                currency="INR",
                transaction_type="NEFT",
                channel="ONLINE",
                sender_country=source["country"],
                receiver_country=receiver["country"],
                sender_balance_before=sender_balance_before,
                sender_balance_after=sender_balance_after,
                receiver_balance_before=receiver_balance_before,
                receiver_balance_after=receiver_balance_after,
                status="COMPLETED",
                typology="fan_out",
            )
        )

    return records


# ---------------------------------------------------------
# Circular transactions
# ---------------------------------------------------------
def generate_circular_transactions(
    accounts,
    count,
    start_id,
    rng,
    fraud_timestamps,
):
    """
    Generate circular transaction patterns:

        A -> B -> C -> A

    Each cycle uses three temporally consecutive
    transactions.
    """

    records = []

    cycle_size = 3

    if count % cycle_size != 0:
        raise ValueError(
            "Circular count must be divisible by 3."
        )

    number_of_cycles = (
        count // cycle_size
    )

    timestamp_index = 0

    for cycle in range(number_of_cycles):

        selected = accounts.sample(
            n=cycle_size,
            random_state=RANDOM_SEED + 400 + cycle,
        )

        selected = selected.reset_index(
            drop=True
        )

        amount = rng.uniform(
            25000,
            150000,
        )

        for step in range(cycle_size):

            sender = selected.iloc[step]

            receiver = selected.iloc[
                (step + 1) % cycle_size
            ]

            sender_balance_before = float(
                sender["current_balance"]
            )

            sender_balance_after = max(
                0,
                sender_balance_before - amount,
            )

            receiver_balance_before = float(
                receiver["current_balance"]
            )

            receiver_balance_after = (
                receiver_balance_before + amount
            )

            timestamp = fraud_timestamps[
                timestamp_index
            ]

            timestamp_index += 1

            transaction_number = (
                cycle * cycle_size + step
            )

            records.append(
                create_fraud_transaction(
                    transaction_id=(
                        f"F"
                        f"{start_id + transaction_number:07d}"
                    ),
                    timestamp=timestamp,
                    sender_account=sender["account_id"],
                    receiver_account=receiver["account_id"],
                    amount=amount,
                    currency="INR",
                    transaction_type="TRANSFER",
                    channel="ONLINE",
                    sender_country=sender["country"],
                    receiver_country=receiver["country"],
                    sender_balance_before=sender_balance_before,
                    sender_balance_after=sender_balance_after,
                    receiver_balance_before=receiver_balance_before,
                    receiver_balance_after=receiver_balance_after,
                    status="COMPLETED",
                    typology="circular",
                )
            )

    return records


# ---------------------------------------------------------
# Main fraud injection function
# ---------------------------------------------------------
def inject_fraud_typologies(
    transaction_file=TRANSACTIONS_FILE,
):
    """
    Load normal transactions and append synthetic
    fraudulent transaction typologies.

    Each typology receives its own timestamps distributed
    across the complete normal transaction period.

    This prevents chronological train/test splits from
    accidentally placing entire typologies into only one
    temporal partition.
    """

    print(
        "\n========== FRAUD INJECTION =========="
    )

    rng = np.random.default_rng(
        RANDOM_SEED
    )

    # -----------------------------------------------------
    # Load transactions
    # -----------------------------------------------------
    transactions = pd.read_csv(
        transaction_file
    )

    print(
        f"Original transactions: "
        f"{len(transactions)}"
    )

    # -----------------------------------------------------
    # Validate timestamps
    # -----------------------------------------------------
    transactions["timestamp"] = pd.to_datetime(
        transactions["timestamp"],
        errors="coerce",
    )

    if transactions["timestamp"].isna().any():
        raise ValueError(
            "Some normal transactions contain invalid "
            "timestamps."
        )

    normal_min_time = transactions[
        "timestamp"
    ].min()

    normal_max_time = transactions[
        "timestamp"
    ].max()

    print(
        f"Normal transaction period: "
        f"{normal_min_time} → {normal_max_time}"
    )

    if normal_min_time >= normal_max_time:
        raise ValueError(
            "Normal transactions must span a valid "
            "time range."
        )

    # -----------------------------------------------------
    # Load account information
    # -----------------------------------------------------
    project_root = (
        Path(__file__)
        .resolve()
        .parents[4]
    )

    accounts_file = (
        project_root
        / "data"
        / "raw"
        / "accounts.csv"
    )

    accounts = pd.read_csv(
        accounts_file
    )

    # -----------------------------------------------------
    # Generate timestamps independently per typology
    # -----------------------------------------------------
    structuring_timestamps = (
        generate_distributed_timestamps(
            count=STRUCTURING_COUNT,
            min_time=normal_min_time,
            max_time=normal_max_time,
            rng=rng,
        )
    )

    rapid_movement_timestamps = (
        generate_distributed_timestamps(
            count=RAPID_MOVEMENT_COUNT,
            min_time=normal_min_time,
            max_time=normal_max_time,
            rng=rng,
        )
    )

    fan_out_timestamps = (
        generate_distributed_timestamps(
            count=FAN_OUT_COUNT,
            min_time=normal_min_time,
            max_time=normal_max_time,
            rng=rng,
        )
    )

    circular_timestamps = (
        generate_circular_timestamps(
            count=CIRCULAR_COUNT,
            min_time=normal_min_time,
            max_time=normal_max_time,
            rng=rng,
        )
    )

    # -----------------------------------------------------
    # Find next fraud transaction ID
    # -----------------------------------------------------
    start_id = 1

    # -----------------------------------------------------
    # Generate fraud typologies
    # -----------------------------------------------------
    structuring = (
        generate_structuring_transactions(
            accounts,
            STRUCTURING_COUNT,
            start_id,
            rng,
            structuring_timestamps,
        )
    )

    start_id += len(structuring)

    rapid_movement = (
        generate_rapid_movement_transactions(
            accounts,
            RAPID_MOVEMENT_COUNT,
            start_id,
            rng,
            rapid_movement_timestamps,
        )
    )

    start_id += len(rapid_movement)

    fan_out = (
        generate_fan_out_transactions(
            accounts,
            FAN_OUT_COUNT,
            start_id,
            rng,
            fan_out_timestamps,
        )
    )

    start_id += len(fan_out)

    circular = (
        generate_circular_transactions(
            accounts,
            CIRCULAR_COUNT,
            start_id,
            rng,
            circular_timestamps,
        )
    )

    # -----------------------------------------------------
    # Combine fraud transactions
    # -----------------------------------------------------
    fraud_records = (
        structuring
        + rapid_movement
        + fan_out
        + circular
    )

    fraud_df = pd.DataFrame(
        fraud_records
    )

    # -----------------------------------------------------
    # Append fraud to normal transactions
    # -----------------------------------------------------
    combined_df = pd.concat(
        [
            transactions,
            fraud_df,
        ],
        ignore_index=True,
    )

    # -----------------------------------------------------
    # Sort by timestamp
    # -----------------------------------------------------
    combined_df["timestamp"] = pd.to_datetime(
        combined_df["timestamp"]
    )

    combined_df = (
        combined_df
        .sort_values(
            "timestamp"
        )
        .reset_index(drop=True)
    )

    # -----------------------------------------------------
    # Create output directory
    # -----------------------------------------------------
    create_directories()

    # -----------------------------------------------------
    # Save updated transactions
    # -----------------------------------------------------
    combined_df.to_csv(
        TRANSACTIONS_FILE,
        index=False,
    )

    # -----------------------------------------------------
    # Print summary
    # -----------------------------------------------------
    print(
        f"\nFraud transactions injected: "
        f"{len(fraud_df)}"
    )

    print(
        "\nFraud distribution:"
    )

    print(
        fraud_df["typology"]
        .value_counts()
        .sort_index()
    )

    print(
        f"\nFinal transaction count: "
        f"{len(combined_df)}"
    )

    print(
        f"\nFraud date range: "
        f"{fraud_df['timestamp'].min()} "
        f"→ "
        f"{fraud_df['timestamp'].max()}"
    )

    # -----------------------------------------------------
    # Chronological 80/20 split check
    # -----------------------------------------------------
    split_index = int(
        len(combined_df) * 0.80
    )

    train_preview = combined_df.iloc[
        :split_index
    ]

    test_preview = combined_df.iloc[
        split_index:
    ]

    print(
        "\n========== CHRONOLOGICAL SPLIT CHECK =========="
    )

    print(
        f"80% split timestamp: "
        f"{combined_df.iloc[split_index]['timestamp']}"
    )

    print(
        f"Training fraud count: "
        f"{int(train_preview['is_fraud'].sum())}"
    )

    print(
        f"Testing fraud count: "
        f"{int(test_preview['is_fraud'].sum())}"
    )

    # -----------------------------------------------------
    # Typology distribution in chronological split
    # -----------------------------------------------------
    train_fraud = train_preview[
        train_preview["is_fraud"] == 1
    ]

    test_fraud = test_preview[
        test_preview["is_fraud"] == 1
    ]

    typologies = [
        "structuring",
        "rapid_movement",
        "fan_out",
        "circular",
    ]

    print(
        "\nTraining fraud by typology:"
    )

    train_typology_counts = (
        train_fraud["typology"]
        .value_counts()
        .reindex(
            typologies,
            fill_value=0,
        )
    )

    print(
        train_typology_counts
    )

    print(
        "\nTesting fraud by typology:"
    )

    test_typology_counts = (
        test_fraud["typology"]
        .value_counts()
        .reindex(
            typologies,
            fill_value=0,
        )
    )

    print(
        test_typology_counts
    )

    # -----------------------------------------------------
    # Warnings
    # -----------------------------------------------------
    missing_train_typologies = [
        typology
        for typology in typologies
        if train_typology_counts[typology] == 0
    ]

    missing_test_typologies = [
        typology
        for typology in typologies
        if test_typology_counts[typology] == 0
    ]

    if missing_train_typologies:
        print(
            "\nWARNING: The following fraud typologies "
            "are absent from the training period:"
        )
        print(
            missing_train_typologies
        )

    if missing_test_typologies:
        print(
            "\nWARNING: The following fraud typologies "
            "are absent from the testing period:"
        )
        print(
            missing_test_typologies
        )

    if (
        not missing_train_typologies
        and not missing_test_typologies
    ):
        print(
            "\nSUCCESS: All four fraud typologies "
            "are represented in both chronological "
            "training and testing periods."
        )

    # -----------------------------------------------------
    # Save message
    # -----------------------------------------------------
    print(
        f"\nSaved to: {TRANSACTIONS_FILE}"
    )

    print(
        "\n========== INJECTION COMPLETE ==========\n"
    )

    return combined_df


# ---------------------------------------------------------
# Script entry point
# ---------------------------------------------------------
if __name__ == "__main__":
    inject_fraud_typologies()