"""
AegisGraph — Rapid Holding Detection Rule.

Detects accounts that receive and immediately forward funds
with minimal holding time, suggesting pass-through behavior.
"""

from __future__ import annotations

import pandas as pd


DEFAULT_HOLD_MINUTES = 30
DEFAULT_FORWARD_RATIO = 0.80


def detect_rapid_holding(
    transactions: pd.DataFrame,
    max_hold_minutes: int = DEFAULT_HOLD_MINUTES,
    forward_ratio: float = DEFAULT_FORWARD_RATIO,
) -> pd.DataFrame:
    """
    Flag accounts that receive funds and forward them
    within `max_hold_minutes`, retaining less than
    `(1 - forward_ratio)` of the received amount.

    Parameters
    ----------
    transactions:
        Must contain: sender_account, receiver_account,
        amount, timestamp.

    max_hold_minutes:
        Maximum time between inflow and outflow.

    forward_ratio:
        Minimum ratio of amount forwarded vs received.

    Returns
    -------
    DataFrame with an added `rapid_holding` boolean column.
    """

    df = transactions.copy()

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp")

    hold_window = pd.Timedelta(minutes=max_hold_minutes)

    flagged_indices = set()

    # Get all accounts that appear as both receiver and sender
    receivers = set(df["receiver_account"].unique())
    senders = set(df["sender_account"].unique())
    pass_through_candidates = receivers & senders

    for account_id in pass_through_candidates:
        inflows = df[
            df["receiver_account"] == account_id
        ].copy()

        outflows = df[
            df["sender_account"] == account_id
        ].copy()

        if inflows.empty or outflows.empty:
            continue

        for in_idx, in_row in inflows.iterrows():
            in_time = in_row["timestamp"]
            in_amount = in_row["amount"]

            # Find outflows within hold window
            time_mask = (
                (outflows["timestamp"] >= in_time)
                & (
                    outflows["timestamp"]
                    <= in_time + hold_window
                )
            )

            nearby_outflows = outflows[time_mask]

            if nearby_outflows.empty:
                continue

            total_forwarded = nearby_outflows["amount"].sum()

            if in_amount > 0 and (
                total_forwarded / in_amount
            ) >= forward_ratio:
                flagged_indices.add(in_idx)
                flagged_indices.update(
                    nearby_outflows.index.tolist()
                )

    df["rapid_holding"] = 0
    df.loc[
        df.index.isin(flagged_indices),
        "rapid_holding",
    ] = 1

    return df
