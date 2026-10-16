"""
AegisGraph — Smurfing Pattern Detection.

Detects smurfing: breaking large amounts into many
small transactions across multiple accounts to avoid
detection thresholds.
"""

from __future__ import annotations

import pandas as pd


DEFAULT_THRESHOLD = 10000.0
DEFAULT_WINDOW_HOURS = 72
DEFAULT_MIN_RECEIVERS = 3


def detect_smurfing(
    transactions: pd.DataFrame,
    threshold: float = DEFAULT_THRESHOLD,
    window_hours: int = DEFAULT_WINDOW_HOURS,
    min_receivers: int = DEFAULT_MIN_RECEIVERS,
) -> pd.DataFrame:
    """
    Flag transactions that are part of a smurfing pattern.

    A smurfing pattern is identified when a sender sends
    small amounts to many different receivers within a
    time window, where the total amount exceeds the threshold.

    Returns
    -------
    DataFrame with an added `smurfing` boolean column.
    """

    df = transactions.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp")
    df["smurfing"] = 0

    window = pd.Timedelta(hours=window_hours)

    for sender_id, group in df.groupby("sender_account"):
        if len(group) < min_receivers:
            continue

        group = group.sort_values("timestamp")

        for i, (idx, row) in enumerate(group.iterrows()):
            window_end = row["timestamp"] + window

            window_txns = group[
                (group["timestamp"] >= row["timestamp"])
                & (group["timestamp"] <= window_end)
            ]

            unique_receivers = window_txns[
                "receiver_account"
            ].nunique()

            total_amount = window_txns["amount"].sum()

            if (
                unique_receivers >= min_receivers
                and total_amount >= threshold
            ):
                df.loc[
                    window_txns.index, "smurfing"
                ] = 1

    return df
