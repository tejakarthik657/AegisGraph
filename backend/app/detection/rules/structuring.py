"""
AegisGraph — Structuring Detection Rule.

Detects transaction structuring (smurfing): multiple
transactions just below reporting thresholds that
together exceed the threshold.
"""

from __future__ import annotations

import pandas as pd


DEFAULT_THRESHOLD = 10000.0
DEFAULT_MARGIN_PCT = 0.20
DEFAULT_WINDOW_HOURS = 48
DEFAULT_MIN_COUNT = 3


def detect_structuring(
    transactions: pd.DataFrame,
    threshold: float = DEFAULT_THRESHOLD,
    margin_pct: float = DEFAULT_MARGIN_PCT,
    window_hours: int = DEFAULT_WINDOW_HOURS,
    min_count: int = DEFAULT_MIN_COUNT,
) -> pd.DataFrame:
    """
    Flag transactions that appear to be structured to
    avoid reporting thresholds.

    A transaction is flagged when:
    - Its amount is below the threshold but within margin_pct.
    - The same sender has at least `min_count` such
      transactions within `window_hours`.

    Parameters
    ----------
    transactions:
        Must contain: sender_account, amount, timestamp.

    threshold:
        The reporting threshold (e.g. $10,000).

    margin_pct:
        Fraction below threshold to consider suspicious.

    window_hours:
        Time window for grouping transactions.

    min_count:
        Minimum number of just-below-threshold transactions
        required to flag.

    Returns
    -------
    DataFrame with an added `structuring` boolean column.
    """

    df = transactions.copy()

    df["timestamp"] = pd.to_datetime(df["timestamp"])

    lower_bound = threshold * (1.0 - margin_pct)

    df["below_threshold"] = (
        (df["amount"] >= lower_bound)
        & (df["amount"] < threshold)
    )

    df["structuring"] = 0

    window = pd.Timedelta(hours=window_hours)

    for account_id, group in df.groupby("sender_account"):
        suspect = group[group["below_threshold"]].copy()

        if len(suspect) < min_count:
            continue

        suspect = suspect.sort_values("timestamp")
        times = suspect["timestamp"].values

        for i, current_time in enumerate(times):
            start_time = current_time - window
            count = int(
                ((times >= start_time) & (times <= current_time)).sum()
            )

            if count >= min_count:
                mask = (
                    (suspect["timestamp"] >= pd.Timestamp(start_time))
                    & (suspect["timestamp"] <= pd.Timestamp(current_time))
                )
                flagged_indices = suspect[mask].index
                df.loc[flagged_indices, "structuring"] = 1

    df = df.drop(columns=["below_threshold"])

    return df
