import numpy as np
import pandas as pd


def chronological_split_index(
    df: pd.DataFrame,
    train_ratio: float = 0.80,
) -> int:
    """Choose the nearest chronological boundary without splitting timestamp ties."""

    if not 0 < train_ratio < 1:
        raise ValueError("train_ratio must be between 0 and 1.")
    if "timestamp" not in df.columns:
        raise ValueError("Chronological split requires a timestamp column.")

    timestamps = pd.to_datetime(df["timestamp"], errors="raise")
    if timestamps.isna().any():
        raise ValueError("Chronological split cannot include missing timestamps.")
    if not timestamps.is_monotonic_increasing:
        raise ValueError("Transactions must be sorted by timestamp before splitting.")

    target = int(len(df) * train_ratio)
    boundaries = [
        index
        for index in range(1, len(df))
        if timestamps.iloc[index - 1] < timestamps.iloc[index]
    ]
    if not boundaries:
        raise ValueError(
            "Cannot split transactions without dividing a timestamp batch."
        )

    return min(
        boundaries,
        key=lambda index: (abs(index - target), -index),
    )


def create_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create event-level temporal features for transactions.

    The dataframe must contain:
        - timestamp
        - sender_account
        - receiver_account

    Temporal features are calculated using only previous
    transactions for the corresponding account.

    Returns:
        DataFrame containing temporal features aligned with
        the original dataframe order.
    """

    required_columns = [
        "timestamp",
        "sender_account",
        "receiver_account",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    result = df.copy()

    # ---------------------------------------------------------
    # Timestamp preparation
    # ---------------------------------------------------------

    result["timestamp"] = pd.to_datetime(
        result["timestamp"],
        errors="coerce",
    )

    if result["timestamp"].isna().any():
        invalid_count = int(
            result["timestamp"].isna().sum()
        )

        raise ValueError(
            f"Found {invalid_count} invalid timestamps."
        )

    # Keep original order so features align with input transactions.
    result["_original_order"] = np.arange(len(result))

    for account_column, previous_column in (
        ("sender_account", "_previous_sender_timestamp"),
        ("receiver_account", "_previous_receiver_timestamp"),
    ):
        account_events = (
            result.loc[
                result[account_column].notna(),
                [account_column, "timestamp"],
            ]
            .drop_duplicates()
            .sort_values(
                [account_column, "timestamp"],
                kind="mergesort",
            )
        )
        account_events[previous_column] = (
            account_events.groupby(
                account_column,
                sort=False,
            )["timestamp"]
            .shift(1)
        )
        result = result.merge(
            account_events,
            on=[account_column, "timestamp"],
            how="left",
            sort=False,
            validate="many_to_one",
        )

    # ---------------------------------------------------------
    # Time since previous sender transaction
    # ---------------------------------------------------------

    result["sender_time_delta_seconds"] = (
        result["timestamp"]
        - result["_previous_sender_timestamp"]
    ).dt.total_seconds()

    # ---------------------------------------------------------
    # Time since previous receiver transaction
    # ---------------------------------------------------------

    result["receiver_time_delta_seconds"] = (
        result["timestamp"]
        - result["_previous_receiver_timestamp"]
    ).dt.total_seconds()

    # First transaction for an account has no previous event.
    # We represent this as zero elapsed time.
    result["sender_time_delta_seconds"] = (
        result["sender_time_delta_seconds"]
        .fillna(0.0)
        .clip(lower=0.0)
    )

    result["receiver_time_delta_seconds"] = (
        result["receiver_time_delta_seconds"]
        .fillna(0.0)
        .clip(lower=0.0)
    )

    # ---------------------------------------------------------
    # Log-scaled temporal features
    # ---------------------------------------------------------

    result["sender_log_time_delta"] = np.log1p(
        result["sender_time_delta_seconds"]
    )

    result["receiver_log_time_delta"] = np.log1p(
        result["receiver_time_delta_seconds"]
    )

    # ---------------------------------------------------------
    # Combined temporal signal
    # ---------------------------------------------------------

    result["time_delta_seconds"] = (
        result[
            [
                "sender_time_delta_seconds",
                "receiver_time_delta_seconds",
            ]
        ]
        .min(axis=1)
    )

    result["log_time_delta"] = np.log1p(
        result["time_delta_seconds"]
    )

    # ---------------------------------------------------------
    # Restore original transaction order
    # ---------------------------------------------------------

    result = (
        result
        .sort_values("_original_order")
        .reset_index(drop=True)
    )

    # ---------------------------------------------------------
    # Cleanup
    # ---------------------------------------------------------

    result = result.drop(
        columns=[
            "_original_order",
            "_previous_sender_timestamp",
            "_previous_receiver_timestamp",
        ]
    )

    return result