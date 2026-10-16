import numpy as np
import pandas as pd


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

    # Keep original order so the generated features can be
    # returned in exactly the same order as the input dataframe.
    result["_original_order"] = np.arange(len(result))

    # Temporal calculations must happen chronologically.
    result = result.sort_values(
        "timestamp"
    ).copy()

    # ---------------------------------------------------------
    # Previous transaction time for sender
    # ---------------------------------------------------------

    result["_previous_sender_timestamp"] = (
        result
        .groupby("sender_account")["timestamp"]
        .shift(1)
    )

    # ---------------------------------------------------------
    # Previous transaction time for receiver
    # ---------------------------------------------------------

    result["_previous_receiver_timestamp"] = (
        result
        .groupby("receiver_account")["timestamp"]
        .shift(1)
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