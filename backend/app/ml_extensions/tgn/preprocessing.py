"""
AegisGraph - Temporal GNN Preprocessing

Shared preprocessing for TGN training and inference.

IMPORTANT:
- Keeps the exact 22 transaction features used by train_temporal_tgn.py.
- Preserves feature ordering.
- Fits StandardScaler ONLY on the chronological training data.
- Saves the scaler and feature schema so inference can reproduce
  training-time preprocessing exactly.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------
# Expected feature names
# ---------------------------------------------------------------------

# These are the exact 22 transaction-level features currently used
# by the TemporalMemoryTGN training pipeline.
EXPECTED_FEATURE_COLUMNS: List[str] = [
    "amount",
    "log_amount",
    "sender_balance_before",
    "sender_balance_after",
    "receiver_balance_before",
    "receiver_balance_after",
    "amount_to_sender_balance",
    "amount_to_receiver_balance",
    "hour",
    "day_of_week",
    "day_of_month",
    "month",
    "is_weekend",
]


# ---------------------------------------------------------------------
# Transaction feature creation
# ---------------------------------------------------------------------

def create_transaction_features(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create the transaction-level features used by TemporalMemoryTGN.

    This intentionally mirrors the feature construction currently used
    inside train_temporal_tgn.py.

    Returns
    -------
    pd.DataFrame
        Unscaled transaction features.

    Notes
    -----
    The categorical one-hot features are generated from:

    - transaction_type
    - channel
    - currency

    Therefore the final number of columns should be 22 for the
    current dataset/schema.
    """

    if df.empty:
        raise ValueError("Input transaction DataFrame is empty.")

    required_columns = [
        "amount",
        "sender_balance_before",
        "sender_balance_after",
        "receiver_balance_before",
        "receiver_balance_after",
        "timestamp",
        "transaction_type",
        "channel",
        "currency",
    ]

    missing_columns = [
        column for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "Missing required transaction columns: "
            + ", ".join(missing_columns)
        )

    # Work on a copy so the caller's DataFrame is never modified.
    data = df.copy()

    # -------------------------------------------------------------
    # Ensure timestamp is datetime
    # -------------------------------------------------------------

    data["timestamp"] = pd.to_datetime(
        data["timestamp"],
        errors="coerce",
    )

    if data["timestamp"].isna().any():
        invalid_count = int(data["timestamp"].isna().sum())

        raise ValueError(
            f"Found {invalid_count} invalid timestamp values."
        )

    # -------------------------------------------------------------
    # Numeric transaction features
    # -------------------------------------------------------------

    features = pd.DataFrame(index=data.index)

    features["amount"] = pd.to_numeric(
        data["amount"],
        errors="coerce",
    )

    # Exact log transform used by the TGN pipeline.
    #
    # clip(lower=0) prevents invalid log values if a malformed
    # negative amount appears.
    features["log_amount"] = np.log1p(
        features["amount"].clip(lower=0)
    )

    features["sender_balance_before"] = pd.to_numeric(
        data["sender_balance_before"],
        errors="coerce",
    )

    features["sender_balance_after"] = pd.to_numeric(
        data["sender_balance_after"],
        errors="coerce",
    )

    features["receiver_balance_before"] = pd.to_numeric(
        data["receiver_balance_before"],
        errors="coerce",
    )

    features["receiver_balance_after"] = pd.to_numeric(
        data["receiver_balance_after"],
        errors="coerce",
    )

    # -------------------------------------------------------------
    # Balance ratio features
    # -------------------------------------------------------------

    sender_balance = features["sender_balance_before"].abs()
    receiver_balance = features["receiver_balance_before"].abs()

    features["amount_to_sender_balance"] = (
        features["amount"] / sender_balance.replace(0, np.nan)
    )

    features["amount_to_receiver_balance"] = (
        features["amount"] / receiver_balance.replace(0, np.nan)
    )

    # -------------------------------------------------------------
    # Temporal features
    # -------------------------------------------------------------

    features["hour"] = data["timestamp"].dt.hour

    features["day_of_week"] = data["timestamp"].dt.dayofweek

    features["day_of_month"] = data["timestamp"].dt.day

    features["month"] = data["timestamp"].dt.month

    features["is_weekend"] = (
        data["timestamp"].dt.dayofweek >= 5
    ).astype(int)

    # -------------------------------------------------------------
    # Categorical one-hot features
    # -------------------------------------------------------------

    categorical = pd.get_dummies(
        data[
            [
                "transaction_type",
                "channel",
                "currency",
            ]
        ],
        columns=[
            "transaction_type",
            "channel",
            "currency",
        ],
        dtype=float,
    )

    # -------------------------------------------------------------
    # Combine numerical + categorical features
    # -------------------------------------------------------------

    features = pd.concat(
        [
            features,
            categorical,
        ],
        axis=1,
    )

    # -------------------------------------------------------------
    # Clean numerical problems
    # -------------------------------------------------------------

    features = features.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    features = features.fillna(0.0)

    # -------------------------------------------------------------
    # Ensure all feature values are numeric
    # -------------------------------------------------------------

    for column in features.columns:
        features[column] = pd.to_numeric(
            features[column],
            errors="coerce",
        )

    features = features.fillna(0.0)

    return features


# ---------------------------------------------------------------------
# Feature schema utilities
# ---------------------------------------------------------------------

def validate_feature_schema(
    features: pd.DataFrame,
    expected_columns: Optional[List[str]] = None,
) -> None:
    """
    Validate the generated feature schema.

    Raises
    ------
    ValueError
        If required columns are missing, duplicated, or unexpected
        columns are present when an expected schema is supplied.
    """

    if features.columns.duplicated().any():
        duplicated = (
            features.columns[
                features.columns.duplicated()
            ]
            .tolist()
        )

        raise ValueError(
            "Duplicate feature columns detected: "
            + ", ".join(duplicated)
        )

    if expected_columns is None:
        return

    actual = list(features.columns)
    expected = list(expected_columns)

    missing = [
        column for column in expected
        if column not in actual
    ]

    unexpected = [
        column for column in actual
        if column not in expected
    ]

    if missing or unexpected or actual != expected:
        message = ["TGN feature schema mismatch."]

        if missing:
            message.append(
                "Missing columns: "
                + ", ".join(missing)
            )

        if unexpected:
            message.append(
                "Unexpected columns: "
                + ", ".join(unexpected)
            )

        if actual != expected:
            message.append(
                "Expected column order: "
                + ", ".join(expected)
            )

            message.append(
                "Actual column order: "
                + ", ".join(actual)
            )

        raise ValueError("\n".join(message))


# ---------------------------------------------------------------------
# Training preprocessing
# ---------------------------------------------------------------------

def fit_preprocessor(
    df: pd.DataFrame,
    split_index: int,
) -> Tuple[StandardScaler, List[str], pd.DataFrame]:
    """
    Create features and fit StandardScaler using ONLY training rows.

    Parameters
    ----------
    df:
        Chronologically ordered transaction DataFrame.

    split_index:
        Number of rows belonging to the chronological training set.

    Returns
    -------
    scaler:
        Fitted StandardScaler.

    feature_columns:
        Exact ordered feature schema.

    transformed_features:
        Full dataset transformed using the scaler fitted on the
        training portion.

    IMPORTANT
    ---------
    This follows the current TGN behavior:

        scaler.fit(training_features)
        scaler.transform(all_features)

    Therefore test rows never influence scaler statistics.
    """

    if split_index <= 0:
        raise ValueError(
            f"split_index must be > 0. Received {split_index}."
        )

    if split_index > len(df):
        raise ValueError(
            f"split_index ({split_index}) cannot exceed "
            f"number of rows ({len(df)})."
        )

    # Create unscaled transaction features.
    raw_features = create_transaction_features(df)

    # Current project expects exactly 22 features.
    feature_columns = list(raw_features.columns)

    if len(feature_columns) != 22:
        raise ValueError(
            "Unexpected TGN feature count.\n"
            f"Expected: 22\n"
            f"Actual: {len(feature_columns)}\n"
            f"Columns: {feature_columns}"
        )

    # -------------------------------------------------------------
    # Fit scaler ONLY on chronological training rows.
    # -------------------------------------------------------------

    scaler = StandardScaler()

    training_features = raw_features.iloc[
        :split_index
    ].copy()

    scaler.fit(training_features)

    # -------------------------------------------------------------
    # Transform the entire chronological dataset.
    # -------------------------------------------------------------

    transformed = scaler.transform(raw_features)

    transformed_features = pd.DataFrame(
        transformed,
        columns=feature_columns,
        index=raw_features.index,
    )

    return (
        scaler,
        feature_columns,
        transformed_features,
    )


# ---------------------------------------------------------------------
# Inference preprocessing
# ---------------------------------------------------------------------

def transform_transaction_features(
    df: pd.DataFrame,
    scaler: StandardScaler,
    feature_columns: List[str],
) -> pd.DataFrame:
    """
    Transform transactions using a previously fitted TGN scaler.

    This function is intended for inference.

    Important:
    categorical columns may be missing from a new batch. Therefore
    the generated feature matrix is explicitly reindexed to the
    training feature schema.

    Missing training-time categorical columns become zero.

    Unexpected columns are ignored after schema alignment.
    """

    if scaler is None:
        raise ValueError(
            "A fitted StandardScaler is required."
        )

    if not feature_columns:
        raise ValueError(
            "feature_columns cannot be empty."
        )

    raw_features = create_transaction_features(df)

    # -------------------------------------------------------------
    # Align inference features to training feature schema.
    # -------------------------------------------------------------

    aligned_features = raw_features.reindex(
        columns=feature_columns,
        fill_value=0.0,
    )

    # -------------------------------------------------------------
    # Verify that every expected feature is present.
    # -------------------------------------------------------------

    missing_after_alignment = [
        column
        for column in feature_columns
        if column not in aligned_features.columns
    ]

    if missing_after_alignment:
        raise ValueError(
            "Unable to construct required TGN features: "
            + ", ".join(missing_after_alignment)
        )

    # -------------------------------------------------------------
    # Apply the scaler fitted during training.
    # -------------------------------------------------------------

    transformed = scaler.transform(
        aligned_features
    )

    transformed_features = pd.DataFrame(
        transformed,
        columns=feature_columns,
        index=aligned_features.index,
    )

    return transformed_features


# ---------------------------------------------------------------------
# Save preprocessor
# ---------------------------------------------------------------------

def save_preprocessor(
    scaler: StandardScaler,
    feature_columns: List[str],
    output_path: str | Path,
) -> Path:
    """
    Save the fitted scaler and feature schema.

    The saved object contains:

        {
            "scaler": StandardScaler,
            "feature_columns": [...]
        }

    Returns
    -------
    Path
        Path to the saved preprocessor.
    """

    if scaler is None:
        raise ValueError(
            "Cannot save a None scaler."
        )

    if not feature_columns:
        raise ValueError(
            "Cannot save an empty feature schema."
        )

    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    payload: Dict[str, Any] = {
        "scaler": scaler,
        "feature_columns": list(feature_columns),
        "feature_count": len(feature_columns),
        "version": 1,
    }

    joblib.dump(
        payload,
        output_path,
    )

    return output_path


# ---------------------------------------------------------------------
# Load preprocessor
# ---------------------------------------------------------------------

def load_preprocessor(
    input_path: str | Path,
) -> Tuple[StandardScaler, List[str]]:
    """
    Load a previously saved TGN preprocessor.

    Returns
    -------
    scaler:
        Fitted StandardScaler.

    feature_columns:
        Exact training-time feature schema.
    """

    input_path = Path(input_path)

    if not input_path.exists():
        raise FileNotFoundError(
            f"TGN preprocessor not found: {input_path}"
        )

    payload = joblib.load(input_path)

    if not isinstance(payload, dict):
        raise ValueError(
            "Invalid TGN preprocessor file. "
            "Expected a dictionary."
        )

    if "scaler" not in payload:
        raise ValueError(
            "TGN preprocessor file does not contain a scaler."
        )

    if "feature_columns" not in payload:
        raise ValueError(
            "TGN preprocessor file does not contain "
            "feature_columns."
        )

    scaler = payload["scaler"]
    feature_columns = payload["feature_columns"]

    if not isinstance(scaler, StandardScaler):
        raise ValueError(
            "Saved TGN scaler is not a StandardScaler."
        )

    if not isinstance(feature_columns, list):
        raise ValueError(
            "Saved feature_columns must be a list."
        )

    return scaler, feature_columns


# ---------------------------------------------------------------------
# Convenience helper
# ---------------------------------------------------------------------

def fit_and_save_preprocessor(
    df: pd.DataFrame,
    split_index: int,
    output_path: str | Path,
) -> Tuple[
    StandardScaler,
    List[str],
    pd.DataFrame,
]:
    """
    Fit the TGN preprocessor, save it, and return the fitted objects.
    """

    (
        scaler,
        feature_columns,
        transformed_features,
    ) = fit_preprocessor(
        df=df,
        split_index=split_index,
    )

    save_preprocessor(
        scaler=scaler,
        feature_columns=feature_columns,
        output_path=output_path,
    )

    return (
        scaler,
        feature_columns,
        transformed_features,
    )


# ---------------------------------------------------------------------
# Standalone validation
# ---------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 70)
    print("AEGISGRAPH TGN PREPROCESSING CHECK")
    print("=" * 70)

    print()
    print("Expected base feature columns:")
    for column in EXPECTED_FEATURE_COLUMNS:
        print(f"  - {column}")

    print()
    print(
        "This module is ready to be imported by "
        "train_temporal_tgn.py and inference.py."
    )

    print()
    print("Expected TGN input dimension: 22")