"""
AegisGraph — Fan-Out Pattern Detection.

Detects fan-out patterns: a single sender distributing
funds to many different receivers.
"""

from __future__ import annotations

import pandas as pd


DEFAULT_WINDOW_HOURS = 24
DEFAULT_MIN_RECEIVERS = 5
DEFAULT_MIN_TOTAL = 25000.0


def detect_fan_out(
    transactions: pd.DataFrame,
    window_hours: int = DEFAULT_WINDOW_HOURS,
    min_receivers: int = DEFAULT_MIN_RECEIVERS,
    min_total: float = DEFAULT_MIN_TOTAL,
) -> pd.DataFrame:
    """
    Flag transactions where a single sender sends
    funds to many receivers within a time window.

    Returns
    -------
    DataFrame with an added `fan_out_pattern` boolean column.
    """

    df = transactions.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp")
    df["fan_out_pattern"] = 0

    window = pd.Timedelta(hours=window_hours)

    for sender_id, group in df.groupby("sender_account"):
        if len(group) < min_receivers:
            continue

        group = group.sort_values("timestamp")

        for _, row in group.iterrows():
            window_end = row["timestamp"] + window

            window_txns = group[
                (group["timestamp"] >= row["timestamp"])
                & (group["timestamp"] <= window_end)
            ]

            unique_receivers = window_txns[
                "receiver_account"
            ].nunique()

            total = window_txns["amount"].sum()

            if (
                unique_receivers >= min_receivers
                and total >= min_total
            ):
                df.loc[
                    window_txns.index, "fan_out_pattern"
                ] = 1

    return df
