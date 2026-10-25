"""Temporal graph preparation built on AegisGraph's existing graph builder.

This first THGNN version models account/transaction node types and adds
explicit elapsed-time features to each transaction node. Account aggregates
are computed from history available to the graph builder; callers doing
chronological evaluation must pass history strictly earlier than targets.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import HeteroData

from app.ml_extensions.heterogeneous.preprocessing import build_heterogeneous_graph


def _isolate_target_account_nodes(graph: HeteroData) -> None:
    """Give each target transaction private account-context nodes."""
    sender_edge = graph["account", "sends", "transaction"].edge_index
    receiver_edge = graph["account", "receives", "transaction"].edge_index
    transaction_count = int(graph["transaction"].x.shape[0])

    if (
        sender_edge.shape[1] != transaction_count
        or receiver_edge.shape[1] != transaction_count
    ):
        raise ValueError(
            "Expected exactly one sender and receiver account edge "
            "for each target transaction."
        )

    transaction_ids = torch.arange(
        transaction_count,
        dtype=torch.long,
        device=sender_edge.device,
    )
    sender_accounts = sender_edge[0]
    receiver_accounts = receiver_edge[0]
    account_features = graph["account"].x
    graph["account"].x = torch.stack(
        [
            account_features[sender_accounts],
            account_features[receiver_accounts],
        ],
        dim=1,
    ).reshape(-1, account_features.shape[1])

    sender_nodes = transaction_ids * 2
    receiver_nodes = sender_nodes + 1
    graph["account", "sends", "transaction"].edge_index = torch.stack(
        [sender_nodes, transaction_ids]
    )
    graph["transaction", "sent_by", "account"].edge_index = torch.stack(
        [transaction_ids, sender_nodes]
    )
    graph["transaction", "received_by", "account"].edge_index = torch.stack(
        [transaction_ids, receiver_nodes]
    )
    graph["account", "receives", "transaction"].edge_index = torch.stack(
        [receiver_nodes, transaction_ids]
    )


def _temporal_features(
    targets: pd.DataFrame,
    history_transactions: pd.DataFrame | None,
) -> np.ndarray:
    """Return [relative_days, log_hours_since_previous_event] per target.

    Relative time is measured from the earliest available history/target event.
    The gap uses the most recent strictly earlier event from history or the
    target batch. Equal-timestamp events do not contribute to one another.
    No fraud labels are model features.
    """
    if targets.empty:
        raise ValueError("Target transactions DataFrame is empty.")

    target_times = pd.to_datetime(targets["timestamp"], errors="raise")
    if history_transactions is not None and not history_transactions.empty:
        history_times = pd.to_datetime(
            history_transactions["timestamp"], errors="raise"
        )
        reference_time = min(history_times.min(), target_times.min())
        historical_previous_time = history_times.max()
    else:
        reference_time = target_times.min()
        historical_previous_time = None

    rows: list[list[float]] = []
    for current_time in target_times:
        previous_target_times = target_times[
            target_times < current_time
        ]
        previous_target_time = (
            previous_target_times.max()
            if not previous_target_times.empty
            else None
        )
        prior_times = [
            value
            for value in (
                historical_previous_time,
                previous_target_time,
            )
            if value is not None
        ]
        previous_time = max(prior_times) if prior_times else None

        relative_days = max(
            0.0, (current_time - reference_time).total_seconds() / 86400.0
        )
        if previous_time is None:
            gap_hours = 0.0
        else:
            gap_hours = max(
                0.0, (current_time - previous_time).total_seconds() / 3600.0
            )
        rows.append([relative_days / 365.0, float(np.log1p(gap_hours))])

    values = np.asarray(rows, dtype=np.float32)
    if not np.isfinite(values).all():
        raise ValueError("Non-finite temporal features generated.")
    return values


def build_thgnn_graph(
    transactions: pd.DataFrame,
    *,
    fit_scalers: bool = True,
    account_scaler=None,
    transaction_scaler=None,
    feature_columns: list[str] | None = None,
    history_transactions: pd.DataFrame | None = None,
):
    """Build a HeteroData graph with transaction-level temporal features.

    Returns the same tuple as build_heterogeneous_graph, with an additional
    ``graph['transaction'].time_features`` tensor attached.
    """
    targets = transactions.copy()
    targets["timestamp"] = pd.to_datetime(
        targets["timestamp"],
        errors="raise",
    )

    if history_transactions is None or history_transactions.empty:
        history = targets.iloc[0:0].copy()
    else:
        history = history_transactions.copy()
        history["timestamp"] = pd.to_datetime(
            history["timestamp"],
            errors="raise",
        )
        history = history[
            history["timestamp"] < targets["timestamp"].min()
        ]

    if "transaction_id" in targets.columns:
        history = history[
            ~history["transaction_id"].isin(targets["transaction_id"])
        ]

    graph, account_scaler, transaction_scaler, labels, feature_columns = (
        build_heterogeneous_graph(
            transactions=targets,
            fit_scalers=fit_scalers,
            account_scaler=account_scaler,
            transaction_scaler=transaction_scaler,
            feature_columns=feature_columns,
            history_transactions=history,
        )
    )

    graph["transaction"].time_features = torch.as_tensor(
        _temporal_features(targets, history),
        dtype=torch.float32,
    )
    _isolate_target_account_nodes(graph)
    return (
        graph,
        account_scaler,
        transaction_scaler,
        labels,
        feature_columns,
    )
