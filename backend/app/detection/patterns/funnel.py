"""
AegisGraph — Funnel Pattern Detection.

Detects funnel patterns: multiple senders consolidating
funds into a single receiver account.
"""

from __future__ import annotations

import pandas as pd


DEFAULT_WINDOW_HOURS = 48
DEFAULT_MIN_SENDERS = 5
DEFAULT_MIN_TOTAL = 50000.0


def detect_funnel(
    transactions: pd.DataFrame,
    window_hours: int = DEFAULT_WINDOW_HOURS,
    min_senders: int = DEFAULT_MIN_SENDERS,
    min_total: float = DEFAULT_MIN_TOTAL,
) -> pd.DataFrame:
    """
    Flag transactions where many different senders
    send funds to the same receiver within a time window.

    Returns
    -------
    DataFrame with an added `funnel` boolean column.
    """

    df = transactions.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp")
    df["funnel"] = 0

    window = pd.Timedelta(hours=window_hours)

    for receiver_id, group in df.groupby("receiver_account"):
        if len(group) < min_senders:
            continue

        group = group.sort_values("timestamp")

        for _, row in group.iterrows():
            window_start = row["timestamp"] - window
            window_end = row["timestamp"]

            window_txns = group[
                (group["timestamp"] >= window_start)
                & (group["timestamp"] <= window_end)
            ]

            unique_senders = window_txns[
                "sender_account"
            ].nunique()

            total = window_txns["amount"].sum()

            if (
                unique_senders >= min_senders
                and total >= min_total
            ):
                df.loc[window_txns.index, "funnel"] = 1

    return df
