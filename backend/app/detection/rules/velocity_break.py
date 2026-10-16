"""
AegisGraph — Velocity Break Detection Rule.

Detects abrupt spikes in transaction frequency for an account
over a short time window, which may indicate automated
or rapid manual movement of funds.
"""

from __future__ import annotations

import pandas as pd


DEFAULT_WINDOW_HOURS = 24
DEFAULT_VELOCITY_THRESHOLD = 10


def detect_velocity_break(
    transactions: pd.DataFrame,
    window_hours: int = DEFAULT_WINDOW_HOURS,
    velocity_threshold: int = DEFAULT_VELOCITY_THRESHOLD,
) -> pd.DataFrame:
    """
    Flag transactions where the sender's outgoing transaction
    count within `window_hours` exceeds `velocity_threshold`.

    Parameters
    ----------
    transactions:
        Must contain: sender_account, timestamp.

    window_hours:
        Rolling window in hours.

    velocity_threshold:
        Maximum allowed transactions in the window.

    Returns
    -------
    DataFrame with an added `velocity_break` boolean column.
    """

    df = transactions.copy()

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp")

    window = pd.Timedelta(hours=window_hours)

    velocity_counts = {}

    for account_id, group in df.groupby("sender_account"):
        times = group["timestamp"].values

        counts = []
        for i, current_time in enumerate(times):
            start_time = current_time - window
            count = int(
                ((times >= start_time) & (times <= current_time)).sum()
            )
            counts.append(count)

        velocity_counts.update(
            dict(zip(group.index, counts))
        )

    df["sender_velocity"] = df.index.map(
        lambda idx: velocity_counts.get(idx, 0)
    )

    df["velocity_break"] = (
        df["sender_velocity"] > velocity_threshold
    ).astype(int)

    return df
