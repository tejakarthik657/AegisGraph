"""
AegisGraph — Layering Pattern Detection.

Detects layering: complex chains of transactions designed
to obscure the origin of funds through multiple hops.
"""

from __future__ import annotations

import pandas as pd


DEFAULT_MIN_HOPS = 3
DEFAULT_WINDOW_HOURS = 48


def detect_layering(
    transactions: pd.DataFrame,
    min_hops: int = DEFAULT_MIN_HOPS,
    window_hours: int = DEFAULT_WINDOW_HOURS,
) -> pd.DataFrame:
    """
    Flag transactions that form chains longer than
    `min_hops`, suggesting layering behavior.

    Returns
    -------
    DataFrame with an added `layering` boolean column.
    """

    df = transactions.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp")
    df["layering"] = 0

    window = pd.Timedelta(hours=window_hours)

    # Build adjacency: receiver -> next sender
    for _, row in df.iterrows():
        chain = [row.name]
        current_receiver = row["receiver_account"]
        current_time = row["timestamp"]

        for hop in range(min_hops):
            next_txns = df[
                (df["sender_account"] == current_receiver)
                & (df["timestamp"] > current_time)
                & (df["timestamp"] <= current_time + window)
            ]

            if next_txns.empty:
                break

            next_txn = next_txns.iloc[0]
            chain.append(next_txn.name)
            current_receiver = next_txn["receiver_account"]
            current_time = next_txn["timestamp"]

        if len(chain) > min_hops:
            df.loc[chain, "layering"] = 1

    return df
