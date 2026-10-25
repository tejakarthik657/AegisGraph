from __future__ import annotations

from typing import Dict, List, Set

import pandas as pd


# ============================================================
# AML RULE CONFIGURATION
# ============================================================

LARGE_TRANSACTION_THRESHOLD = 100_000.0

# Synthetic-data-specific structuring heuristic.
# The generator creates low-value transactions distributed
# across the overall timeline rather than repeated transactions
# from the same sender within a short window.
STRUCTURING_MAX_AMOUNT = 50_000.0
STRUCTURING_MIN_COUNT = 3

# Rapid movement in the current synthetic generator is represented
# by high-value transactions rather than explicit A -> B -> C chains.
RAPID_MOVEMENT_MINIMUM_AMOUNT = 50_000.0

# Fan-out is generated as one source account sending to many
# different receivers across the overall timeline.
FAN_OUT_MINIMUM_RECEIVERS = 3
FAN_OUT_MINIMUM_TOTAL_AMOUNT = 100_000.0

# Circular movement is explicitly generated as:
# A -> B -> C -> A
CIRCULAR_WINDOW_HOURS = 24
CIRCULAR_MINIMUM_AMOUNT = 50_000.0


# ============================================================
# HELPERS
# ============================================================

def _prepare_transactions(
    transactions: pd.DataFrame,
) -> pd.DataFrame:
    """
    Prepare transaction history for rule evaluation.

    Required columns:
        transaction_id
        timestamp
        sender_account
        receiver_account
        amount
    """

    required_columns = [
        "transaction_id",
        "timestamp",
        "sender_account",
        "receiver_account",
        "amount",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in transactions.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    data = transactions[required_columns].copy()

    data["timestamp"] = pd.to_datetime(
        data["timestamp"],
        errors="coerce",
    )

    data["amount"] = pd.to_numeric(
        data["amount"],
        errors="coerce",
    )

    data = data.dropna(
        subset=[
            "transaction_id",
            "timestamp",
            "sender_account",
            "receiver_account",
            "amount",
        ]
    )

    data = (
        data
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    return data


def score_transactions_chronologically(
    transactions: pd.DataFrame,
) -> pd.DataFrame:
    """
    Score each event using only strictly earlier event timestamps.

    Events sharing a timestamp are scored as a batch before any are
    added to the history. Each target event is available to its own
    amount-based and cumulative rule checks.
    """

    data = _prepare_transactions(transactions)
    data = data.sort_values(
        "timestamp",
        kind="mergesort",
    ).reset_index(drop=True)

    score_columns = [
        "large_transaction",
        "structuring",
        "rapid_movement",
        "fan_out",
        "circular_movement",
    ]
    for column in score_columns:
        data[column] = 0

    low_value_counts: Dict[str, int] = {}
    sender_total_amounts: Dict[str, float] = {}
    sender_receivers: Dict[str, Set[str]] = {}
    edges_by_pair: Dict[
        tuple[str, str],
        list[tuple[pd.Timestamp, float]],
    ] = {}
    outgoing_edges: Dict[
        str,
        list[tuple[pd.Timestamp, str, float]],
    ] = {}

    for _, timestamp_group in data.groupby(
        "timestamp",
        sort=False,
    ):
        batch_scores = []

        for row_index, transaction in timestamp_group.iterrows():
            sender = transaction["sender_account"]
            receiver = transaction["receiver_account"]
            amount = float(transaction["amount"])
            timestamp = transaction["timestamp"]

            structuring_count = low_value_counts.get(sender, 0)
            if amount < STRUCTURING_MAX_AMOUNT:
                structuring_count += 1

            receivers = sender_receivers.get(sender, set())
            fan_out_receiver_count = len(receivers | {receiver})
            fan_out_total = (
                sender_total_amounts.get(sender, 0.0)
                + amount
            )

            circular = False
            if sender != receiver:
                for first_time, middle_account, first_amount in (
                    outgoing_edges.get(receiver, ())
                ):
                    if (
                        first_amount < CIRCULAR_MINIMUM_AMOUNT
                        or timestamp - first_time
                        > pd.Timedelta(hours=CIRCULAR_WINDOW_HOURS)
                    ):
                        continue
                    second_edges = edges_by_pair.get(
                        (middle_account, sender),
                        (),
                    )
                    if any(
                        first_time < second_time < timestamp
                        for second_time, _ in second_edges
                    ):
                        circular = True
                        break

            batch_scores.append(
                (
                    row_index,
                    int(amount >= LARGE_TRANSACTION_THRESHOLD),
                    int(
                        amount < STRUCTURING_MAX_AMOUNT
                        and structuring_count >= STRUCTURING_MIN_COUNT
                    ),
                    int(amount >= RAPID_MOVEMENT_MINIMUM_AMOUNT),
                    int(
                        fan_out_receiver_count >= FAN_OUT_MINIMUM_RECEIVERS
                        and fan_out_total >= FAN_OUT_MINIMUM_TOTAL_AMOUNT
                    ),
                    int(circular),
                )
            )

        for (
            row_index,
            large,
            structuring,
            rapid,
            fan_out,
            circular,
        ) in batch_scores:
            data.loc[row_index, score_columns] = [
                large,
                structuring,
                rapid,
                fan_out,
                circular,
            ]

        for _, transaction in timestamp_group.iterrows():
            sender = transaction["sender_account"]
            receiver = transaction["receiver_account"]
            amount = float(transaction["amount"])

            if amount < STRUCTURING_MAX_AMOUNT:
                low_value_counts[sender] = (
                    low_value_counts.get(sender, 0) + 1
                )
            sender_total_amounts[sender] = (
                sender_total_amounts.get(sender, 0.0) + amount
            )
            sender_receivers.setdefault(sender, set()).add(receiver)
            edges_by_pair.setdefault(
                (sender, receiver),
                [],
            ).append(
                (transaction["timestamp"], amount)
            )
            outgoing_edges.setdefault(sender, []).append(
                (
                    transaction["timestamp"],
                    receiver,
                    amount,
                )
            )

    data["rule_prediction"] = data[score_columns].max(axis=1).astype(int)
    return data


# ============================================================
# RULE 1 — LARGE TRANSACTION
# ============================================================

def check_large_transaction(
    transaction: pd.Series,
) -> Dict:
    """
    Detect a single large transaction.

    This rule is independent of transaction history.
    """

    amount = float(transaction["amount"])
    transaction_id = transaction["transaction_id"]

    triggered = (
        amount >= LARGE_TRANSACTION_THRESHOLD
    )

    return {
        "rule": "large_transaction",
        "triggered": triggered,
        "transaction_id": transaction_id,
        "matching_transaction_ids": (
            [transaction_id]
            if triggered
            else []
        ),
        "amount": amount,
        "threshold": LARGE_TRANSACTION_THRESHOLD,
        "reason": (
            f"Transaction amount {amount:.2f} "
            f"meets or exceeds threshold "
            f"{LARGE_TRANSACTION_THRESHOLD:.2f}"
            if triggered
            else "Transaction amount below threshold"
        ),
    }


# ============================================================
# RULE 2 — STRUCTURING
# ============================================================

def check_structuring(
    transactions: pd.DataFrame,
) -> Dict:
    """
    Detect synthetic structuring behavior.

    The current synthetic generator creates individual
    transactions in the range ₹8,000–₹49,000 and distributes
    them across the overall transaction timeline.

    Therefore this rule identifies low-value transactions
    below the large-transaction threshold and looks for
    repeated low-value activity.

    The rule does NOT use the synthetic typology label.
    """

    data = _prepare_transactions(transactions)

    low_value = data[
        data["amount"] < STRUCTURING_MAX_AMOUNT
    ].copy()

    matching_ids: Set[str] = set()
    matching_groups: List[Dict] = []

    if low_value.empty:
        return {
            "rule": "structuring",
            "triggered": False,
            "matching_transaction_ids": [],
            "matching_groups": [],
            "maximum_amount": STRUCTURING_MAX_AMOUNT,
            "minimum_count": STRUCTURING_MIN_COUNT,
            "reason": "No low-value structuring activity detected",
        }

    # Group low-value activity by sender.
    sender_groups = (
        low_value
        .groupby("sender_account", sort=False)
    )

    for sender_account, group in sender_groups:

        if len(group) < STRUCTURING_MIN_COUNT:
            continue

        group = (
            group
            .sort_values("timestamp")
            .reset_index(drop=True)
        )

        transaction_ids = (
            group["transaction_id"]
            .tolist()
        )

        total_amount = float(
            group["amount"].sum()
        )

        matching_ids.update(transaction_ids)

        matching_groups.append(
            {
                "sender_account": sender_account,
                "transaction_ids": transaction_ids,
                "transaction_count": len(group),
                "total_amount": total_amount,
                "start_time": group["timestamp"].iloc[0],
                "end_time": group["timestamp"].iloc[-1],
            }
        )

    matching_ids_list = sorted(
        matching_ids,
        key=str,
    )

    return {
        "rule": "structuring",
        "triggered": bool(matching_ids_list),
        "matching_transaction_ids": matching_ids_list,
        "matching_groups": matching_groups,
        "maximum_amount": STRUCTURING_MAX_AMOUNT,
        "minimum_count": STRUCTURING_MIN_COUNT,
        "reason": (
            f"Detected {len(matching_groups)} sender-level "
            f"low-value activity group(s) covering "
            f"{len(matching_ids_list)} transaction(s)"
            if matching_ids_list
            else "No structuring activity detected"
        ),
    }


# ============================================================
# RULE 3 — RAPID MOVEMENT
# ============================================================

def check_rapid_movement(
    transactions: pd.DataFrame,
) -> Dict:
    """
    Detect the high-value rapid-movement behavior represented
    by the current synthetic generator.

    The current generator creates rapid-movement transactions
    independently across the overall timeline with amounts
    between ₹50,000 and ₹300,000.

    Therefore this rule flags transactions at or above the
    synthetic rapid-movement minimum.
    """

    data = _prepare_transactions(transactions)

    matches = data[
        data["amount"] >= RAPID_MOVEMENT_MINIMUM_AMOUNT
    ].copy()

    matching_ids = (
        matches["transaction_id"]
        .tolist()
    )

    matching_patterns = []

    for _, transaction in matches.iterrows():
        matching_patterns.append(
            {
                "transaction_ids": [
                    transaction["transaction_id"]
                ],
                "sender_account": transaction["sender_account"],
                "receiver_account": transaction["receiver_account"],
                "amount": float(transaction["amount"]),
                "timestamp": transaction["timestamp"],
            }
        )

    return {
        "rule": "rapid_movement",
        "triggered": bool(matching_ids),
        "matching_transaction_ids": matching_ids,
        "matching_patterns": matching_patterns,
        "minimum_amount": RAPID_MOVEMENT_MINIMUM_AMOUNT,
        "reason": (
            f"Detected {len(matching_ids)} high-value "
            f"rapid-movement transaction(s)"
            if matching_ids
            else "No rapid-movement activity detected"
        ),
    }


# ============================================================
# RULE 4 — FAN-OUT
# ============================================================

def check_fan_out(
    transactions: pd.DataFrame,
) -> Dict:
    """
    Detect fan-out behavior.

    Pattern:

        A -> B
        A -> C
        A -> D
        ...

    The current generator uses one source account and many
    different receivers across the overall timeline.

    Therefore the rule does not require the transactions
    to occur within a 24-hour window.
    """

    data = _prepare_transactions(transactions)

    matching_ids: Set[str] = set()
    matching_patterns: List[Dict] = []

    for sender_account, group in data.groupby(
        "sender_account",
        sort=False,
    ):

        distinct_receivers = (
            group["receiver_account"]
            .nunique()
        )

        total_amount = float(
            group["amount"].sum()
        )

        if (
            distinct_receivers >= FAN_OUT_MINIMUM_RECEIVERS
            and total_amount >= FAN_OUT_MINIMUM_TOTAL_AMOUNT
        ):
            transaction_ids = (
                group["transaction_id"]
                .tolist()
            )

            matching_ids.update(
                transaction_ids
            )

            matching_patterns.append(
                {
                    "sender_account": sender_account,
                    "transaction_ids": transaction_ids,
                    "distinct_receivers": distinct_receivers,
                    "total_amount": total_amount,
                    "start_time": group["timestamp"].min(),
                    "end_time": group["timestamp"].max(),
                }
            )

    matching_ids_list = sorted(
        matching_ids,
        key=str,
    )

    return {
        "rule": "fan_out",
        "triggered": bool(matching_ids_list),
        "matching_transaction_ids": matching_ids_list,
        "matching_patterns": matching_patterns,
        "minimum_receivers": FAN_OUT_MINIMUM_RECEIVERS,
        "minimum_total_amount": FAN_OUT_MINIMUM_TOTAL_AMOUNT,
        "reason": (
            f"Detected {len(matching_patterns)} fan-out "
            f"sender pattern(s) covering "
            f"{len(matching_ids_list)} transaction(s)"
            if matching_ids_list
            else "No fan-out pattern detected"
        ),
    }


# ============================================================
# RULE 5 — CIRCULAR MOVEMENT
# ============================================================

def check_circular_movement(
    transactions: pd.DataFrame,
) -> Dict:
    """
    Detect an explicit three-hop circular pattern:

        A -> B
        B -> C
        C -> A

    The three transactions must occur within the configured
    time window.

    The first transaction must meet the minimum amount.
    """

    data = _prepare_transactions(transactions)

    matching_ids: Set[str] = set()
    matching_patterns: List[Dict] = []

    for first_index in range(len(data)):

        first_transaction = data.iloc[
            first_index
        ]

        first_time = first_transaction["timestamp"]

        account_a = (
            first_transaction["sender_account"]
        )

        account_b = (
            first_transaction["receiver_account"]
        )

        first_amount = float(
            first_transaction["amount"]
        )

        if first_amount < CIRCULAR_MINIMUM_AMOUNT:
            continue

        for second_index in range(
            first_index + 1,
            len(data),
        ):

            second_transaction = data.iloc[
                second_index
            ]

            second_time = (
                second_transaction["timestamp"]
            )

            elapsed_second = (
                second_time - first_time
            ).total_seconds() / 3600.0

            if elapsed_second > CIRCULAR_WINDOW_HOURS:
                break

            if (
                second_transaction["sender_account"]
                != account_b
            ):
                continue

            account_c = (
                second_transaction["receiver_account"]
            )

            if account_c == account_a:
                continue

            for third_index in range(
                second_index + 1,
                len(data),
            ):

                third_transaction = data.iloc[
                    third_index
                ]

                third_time = (
                    third_transaction["timestamp"]
                )

                elapsed_third = (
                    third_time - first_time
                ).total_seconds() / 3600.0

                if elapsed_third > CIRCULAR_WINDOW_HOURS:
                    break

                if (
                    third_transaction["sender_account"]
                    == account_c
                    and
                    third_transaction["receiver_account"]
                    == account_a
                ):

                    first_id = (
                        first_transaction["transaction_id"]
                    )

                    second_id = (
                        second_transaction["transaction_id"]
                    )

                    third_id = (
                        third_transaction["transaction_id"]
                    )

                    matching_ids.update(
                        [
                            first_id,
                            second_id,
                            third_id,
                        ]
                    )

                    matching_patterns.append(
                        {
                            "transaction_ids": [
                                first_id,
                                second_id,
                                third_id,
                            ],
                            "accounts": [
                                account_a,
                                account_b,
                                account_c,
                            ],
                            "amounts": [
                                first_amount,
                                float(
                                    second_transaction["amount"]
                                ),
                                float(
                                    third_transaction["amount"]
                                ),
                            ],
                            "start_time": first_time,
                            "end_time": third_time,
                            "elapsed_hours": elapsed_third,
                        }
                    )

    matching_ids_list = sorted(
        matching_ids,
        key=str,
    )

    return {
        "rule": "circular_movement",
        "triggered": bool(matching_ids_list),
        "matching_transaction_ids": matching_ids_list,
        "matching_patterns": matching_patterns,
        "window_hours": CIRCULAR_WINDOW_HOURS,
        "minimum_amount": CIRCULAR_MINIMUM_AMOUNT,
        "reason": (
            f"Detected {len(matching_patterns)} circular "
            f"pattern(s) covering "
            f"{len(matching_ids_list)} transaction(s)"
            if matching_ids_list
            else "No circular movement detected"
        ),
    }


# ============================================================
# SINGLE TRANSACTION RULE EVALUATION
# ============================================================

def evaluate_transaction_rules(
    transaction: pd.Series,
) -> Dict:
    """
    Evaluate rules that can be applied independently to
    a single transaction.

    History-dependent rules are intentionally excluded because
    they require transaction history.
    """

    large_transaction_result = (
        check_large_transaction(transaction)
    )

    return {
        "transaction_id": transaction["transaction_id"],
        "large_transaction": large_transaction_result,
        "triggered_rules": (
            ["large_transaction"]
            if large_transaction_result["triggered"]
            else []
        ),
        "risk_flag": (
            large_transaction_result["triggered"]
        ),
    }


# ============================================================
# ALL HISTORY RULES
# ============================================================

def evaluate_history_rules(
    transactions: pd.DataFrame,
) -> Dict[str, Dict]:
    """
    Evaluate all history-dependent AML rules.
    """

    return {
        "structuring": check_structuring(
            transactions
        ),
        "rapid_movement": check_rapid_movement(
            transactions
        ),
        "fan_out": check_fan_out(
            transactions
        ),
        "circular_movement": check_circular_movement(
            transactions
        ),
    }