
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import torch

from sklearn.preprocessing import StandardScaler
from torch_geometric.data import HeteroData


REQUIRED_COLUMNS = [
    "transaction_id",
    "timestamp",
    "sender_account",
    "receiver_account",
    "amount",
    "currency",
    "transaction_type",
    "channel",
    "sender_country",
    "receiver_country",
    "is_fraud",
]

CATEGORICAL_COLUMNS = [
    "currency",
    "transaction_type",
    "channel",
    "sender_country",
    "receiver_country",
]

NUMERIC_COLUMNS = [
    "amount",
    "log_amount",
    "hour",
    "day_of_week",
    "day_of_month",
    "month",
    "is_weekend",
]


def _validate_transactions(
    transactions: pd.DataFrame,
    *,
    allow_empty: bool = False,
) -> pd.DataFrame:
    missing = [
        column for column in REQUIRED_COLUMNS
        if column not in transactions.columns
    ]

    if missing:
        raise ValueError(
            "Missing transaction columns: "
            + ", ".join(missing)
        )

    if transactions.empty and not allow_empty:
        raise ValueError("Transactions DataFrame is empty.")

    df = transactions.copy()

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        errors="coerce",
    )

    if df["timestamp"].isna().any():
        raise ValueError("Invalid transaction timestamps found.")

    if df["transaction_id"].duplicated().any():
        raise ValueError("Duplicate transaction IDs found.")

    if df[
        ["sender_account", "receiver_account"]
    ].isna().any().any():
        raise ValueError("Missing account IDs found.")

    df["amount"] = pd.to_numeric(
        df["amount"],
        errors="coerce",
    )

    if df["amount"].isna().any():
        raise ValueError("Invalid transaction amounts found.")

    if (df["amount"] < 0).any():
        raise ValueError("Negative transaction amounts found.")

    labels = pd.to_numeric(
        df["is_fraud"],
        errors="coerce",
    )

    if labels.isna().any() or not labels.isin([0, 1]).all():
        raise ValueError("is_fraud must contain only 0 and 1.")

    df["is_fraud"] = labels.astype(np.int64)

    for column in CATEGORICAL_COLUMNS:
        df[column] = (
            df[column]
            .fillna("UNKNOWN")
            .astype(str)
        )

    return df.sort_values(
        "timestamp"
    ).reset_index(drop=True)


def _transaction_features(
    transactions: pd.DataFrame,
    feature_columns: list[str] | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    df = transactions

    timestamps = df["timestamp"]

    numeric = pd.DataFrame(index=df.index)
    numeric["amount"] = df["amount"].astype(float)
    numeric["log_amount"] = np.log1p(
        df["amount"].clip(lower=0)
    )
    numeric["hour"] = timestamps.dt.hour
    numeric["day_of_week"] = timestamps.dt.dayofweek
    numeric["day_of_month"] = timestamps.dt.day
    numeric["month"] = timestamps.dt.month
    numeric["is_weekend"] = (
        timestamps.dt.dayofweek >= 5
    ).astype(float)

    categorical = pd.get_dummies(
        df[CATEGORICAL_COLUMNS],
        prefix=CATEGORICAL_COLUMNS,
        dtype=float,
    )

    features = pd.concat(
        [numeric, categorical],
        axis=1,
    )

    if feature_columns is None:
        feature_columns = list(features.columns)

    # Preserve the training-time column order.
    # Unseen categories in later data become all-zero columns.
    features = features.reindex(
        columns=feature_columns,
        fill_value=0.0,
    )

    features = features.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    if features.isna().any().any():
        raise ValueError(
            "Non-finite transaction feature values found."
        )

    return features, feature_columns


def _historical_account_features(
    history: pd.DataFrame,
    account_ids: list[str],
) -> pd.DataFrame:
    """Compute account aggregates from historical rows only."""

    features = pd.DataFrame(
        0.0,
        index=account_ids,
        columns=[
            "in_degree",
            "out_degree",
            "total_incoming",
            "total_outgoing",
            "unique_neighbors",
        ],
    )

    if history.empty:
        return features

    incoming = history.groupby(
        "receiver_account"
    )["amount"].agg(["size", "sum"])

    outgoing = history.groupby(
        "sender_account"
    )["amount"].agg(["size", "sum"])

    features["in_degree"] = (
        incoming["size"].reindex(account_ids).fillna(0)
    )

    features["out_degree"] = (
        outgoing["size"].reindex(account_ids).fillna(0)
    )

    features["total_incoming"] = (
        incoming["sum"].reindex(account_ids).fillna(0)
    )

    features["total_outgoing"] = (
        outgoing["sum"].reindex(account_ids).fillna(0)
    )

    neighbors: dict[str, set[str]] = {
        account: set() for account in account_ids
    }

    for sender, receiver in zip(
        history["sender_account"],
        history["receiver_account"],
    ):
        neighbors[sender].add(receiver)
        neighbors[receiver].add(sender)

    features["unique_neighbors"] = [
        len(neighbors[account])
        for account in account_ids
    ]

    return features


def build_heterogeneous_graph(
    transactions: pd.DataFrame,
    fit_scalers: bool = True,
    account_scaler: StandardScaler | None = None,
    transaction_scaler: StandardScaler | None = None,
    feature_columns: list[str] | None = None,
    history_transactions: pd.DataFrame | None = None,
) -> tuple[
    HeteroData,
    StandardScaler,
    StandardScaler,
    np.ndarray,
    list[str],
]:
    """
    Build an account-transaction heterogeneous graph.

    `transactions` contains the transaction nodes to classify.

    `history_transactions` supplies historical account aggregates.
    If omitted, transactions are used as history. For a strict
    chronological evaluation, pass only rows preceding the target
    transactions.

    Labels are returned separately and are never model features.
    """

    targets = _validate_transactions(transactions)

    if history_transactions is None:
        history = targets.copy()
    else:
        history = _validate_transactions(
            history_transactions,
            allow_empty=True,
        )

        if (
            not history.empty
            and not targets.empty
            and history["timestamp"].max()
            >= targets["timestamp"].min()
        ):
            raise ValueError(
                "History must precede all target transactions. "
                "For equal timestamps, define an explicit event "
                "ordering before building the graph."
            )

    # Stable account mapping across history and target rows.
    all_accounts = pd.unique(
        pd.concat(
            [
                history["sender_account"],
                history["receiver_account"],
                targets["sender_account"],
                targets["receiver_account"],
            ],
            ignore_index=True,
        )
    ).tolist()

    account_to_index = {
        account: index
        for index, account in enumerate(all_accounts)
    }

    # Account features use historical rows only.
    account_frame = _historical_account_features(
        history,
        all_accounts,
    )

    account_values = account_frame.to_numpy(
        dtype=np.float32
    )

    # Transaction features use a training-fitted schema.
    transaction_frame, feature_columns = _transaction_features(
        targets,
        feature_columns,
    )

    transaction_values = transaction_frame.to_numpy(
        dtype=np.float32
    )

    if not np.isfinite(account_values).all():
        raise ValueError("Non-finite account features found.")

    if not np.isfinite(transaction_values).all():
        raise ValueError("Non-finite transaction features found.")

    if fit_scalers:
        account_scaler = StandardScaler()
        transaction_scaler = StandardScaler()

        account_values = account_scaler.fit_transform(
            account_values
        ).astype(np.float32)

        transaction_values = transaction_scaler.fit_transform(
            transaction_values
        ).astype(np.float32)

    else:
        if account_scaler is None or transaction_scaler is None:
            raise ValueError(
                "Provide both fitted scalers when fit_scalers=False."
            )

        account_values = account_scaler.transform(
            account_values
        ).astype(np.float32)

        transaction_values = transaction_scaler.transform(
            transaction_values
        ).astype(np.float32)

    sender_indices = targets["sender_account"].map(
        account_to_index
    ).to_numpy(dtype=np.int64)

    receiver_indices = targets["receiver_account"].map(
        account_to_index
    ).to_numpy(dtype=np.int64)

    transaction_indices = np.arange(
        len(targets),
        dtype=np.int64,
    )

    account_sender = torch.tensor(
        sender_indices,
        dtype=torch.long,
    )

    account_receiver = torch.tensor(
        receiver_indices,
        dtype=torch.long,
    )

    transaction_index = torch.tensor(
        transaction_indices,
        dtype=torch.long,
    )

    graph = HeteroData()

    graph["account"].x = torch.tensor(
        account_values,
        dtype=torch.float32,
    )

    graph["transaction"].x = torch.tensor(
        transaction_values,
        dtype=torch.float32,
    )

    graph[
        "account", "sends", "transaction"
    ].edge_index = torch.stack(
        [account_sender, transaction_index],
        dim=0,
    )

    graph[
        "transaction", "sent_by", "account"
    ].edge_index = torch.stack(
        [transaction_index, account_sender],
        dim=0,
    )

    graph[
        "transaction", "received_by", "account"
    ].edge_index = torch.stack(
        [transaction_index, account_receiver],
        dim=0,
    )

    graph[
        "account", "receives", "transaction"
    ].edge_index = torch.stack(
        [account_receiver, transaction_index],
        dim=0,
    )

    labels = targets["is_fraud"].to_numpy(
        dtype=np.float32
    )

    return (
        graph,
        account_scaler,
        transaction_scaler,
        labels,
        feature_columns,
    )
