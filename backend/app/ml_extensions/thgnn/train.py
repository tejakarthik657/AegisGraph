"""Train and chronologically evaluate AegisGraph's THGNN prototype.

Run from the backend directory:
    python -m app.ml_extensions.thgnn.train
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch_geometric.data import HeteroData
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

BACKEND_DIR = Path(__file__).resolve().parents[3]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.ml_extensions.thgnn.model import TemporalHeterogeneousFraudGNN
from app.ml_extensions.thgnn.preprocessing import build_thgnn_graph
from app.utils.config import CLEAN_TRANSACTIONS_FILE, RESULTS_DATA_DIR

SEED = 42
TRAIN_RATIO = 0.80
EPOCHS = 20
LEARNING_RATE = 0.001
HIDDEN_DIM = 64
DROPOUT = 0.2
THRESHOLD = 0.5
MODEL_FILE = RESULTS_DATA_DIR / "thgnn_model.pt"
PREPROCESSOR_FILE = RESULTS_DATA_DIR / "thgnn_preprocessor.joblib"
PREDICTIONS_FILE = RESULTS_DATA_DIR / "thgnn_predictions.csv"
METRICS_FILE = RESULTS_DATA_DIR / "thgnn_metrics.csv"


def _set_seed() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)


def _load_split() -> tuple[pd.DataFrame, pd.DataFrame]:
    if not CLEAN_TRANSACTIONS_FILE.exists():
        raise FileNotFoundError(
            f"Clean transactions not found at {CLEAN_TRANSACTIONS_FILE}. "
            "Run the data pipeline first."
        )
    df = pd.read_csv(CLEAN_TRANSACTIONS_FILE)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="raise")
    df = df.sort_values("timestamp", kind="mergesort").reset_index(drop=True)
    target_split = int(len(df) * TRAIN_RATIO)
    boundaries = [
        index
        for index in range(1, len(df))
        if df.loc[index - 1, "timestamp"] < df.loc[index, "timestamp"]
    ]
    if not boundaries:
        raise ValueError(
            "Cannot split transactions without dividing a timestamp batch."
        )
    split = min(
        boundaries,
        key=lambda index: (abs(index - target_split), -index),
    )
    if split < 2 or split >= len(df):
        raise ValueError("Not enough transactions for chronological train/test split.")
    train_df, test_df = df.iloc[:split].copy(), df.iloc[split:].copy()
    return train_df, test_df


def main() -> None:
    _set_seed()
    RESULTS_DATA_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"THGNN device: {device}")
    train_df, test_df = _load_split()
    print(f"Train transactions: {len(train_df)} | test transactions: {len(test_df)}")

    _, account_scaler, transaction_scaler, _, feature_columns = (
        build_thgnn_graph(train_df, fit_scalers=True)
    )

    # Build each training batch from a strict timestamp prefix. Transactions
    # sharing a timestamp are scored together and cannot enter one another's
    # account history.
    training_batches: list[tuple[HeteroData, torch.Tensor]] = []
    for timestamp, batch in train_df.groupby("timestamp", sort=True):
        history = train_df[train_df["timestamp"] < timestamp]
        graph, _, _, batch_labels, _ = build_thgnn_graph(
            batch,
            fit_scalers=False,
            account_scaler=account_scaler,
            transaction_scaler=transaction_scaler,
            feature_columns=feature_columns,
            history_transactions=history,
        )
        training_batches.append(
            (
                graph.to(device),
                torch.as_tensor(
                    batch_labels,
                    dtype=torch.float32,
                    device=device,
                ),
            )
        )

    first_graph = training_batches[0][0]
    model = TemporalHeterogeneousFraudGNN(
        account_input_dim=first_graph["account"].x.shape[1],
        transaction_input_dim=first_graph["transaction"].x.shape[1],
        hidden_dim=HIDDEN_DIM,
        dropout=DROPOUT,
    ).to(device)

    labels = torch.tensor(
        train_df["is_fraud"].to_numpy(),
        dtype=torch.float32,
        device=device,
    )
    positives = max(float(labels.sum().item()), 1.0)
    negatives = max(float(len(labels) - labels.sum().item()), 1.0)
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(
            negatives / positives,
            dtype=torch.float32,
            device=device,
        ),
        reduction="sum",
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        optimizer.zero_grad()
        for graph_batch, labels_batch in training_batches:
            logits = model(graph_batch)
            loss = criterion(logits, labels_batch)
            (loss / len(train_df)).backward()
            total_loss += float(loss.item())
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        if epoch == 1 or epoch % 5 == 0 or epoch == EPOCHS:
            mean_loss = total_loss / len(train_df)
            print(f"Epoch {epoch:02d}/{EPOCHS} | loss={mean_loss:.4f}")

    # Score equal-timestamp test events as one batch so they share the same
    # strictly-prior history snapshot.
    model.eval()
    records: list[dict[str, object]] = []
    probabilities: list[float] = []
    labels_true: list[int] = []
    test_history = train_df.copy()
    with torch.no_grad():
        for _, batch in test_df.groupby("timestamp", sort=True):
            current = batch.copy()
            test_graph, _, _, y, _ = build_thgnn_graph(
                current,
                fit_scalers=False,
                account_scaler=account_scaler,
                transaction_scaler=transaction_scaler,
                feature_columns=feature_columns,
                history_transactions=test_history,
            )
            batch_probabilities = torch.sigmoid(
                model(test_graph.to(device))
            ).cpu().tolist()
            probabilities.extend(float(value) for value in batch_probabilities)
            labels_true.extend(int(value) for value in y)
            for (_, row), probability, label in zip(
                batch.iterrows(),
                batch_probabilities,
                y,
            ):
                records.append({
                    "transaction_id": str(row["transaction_id"]),
                    "timestamp": row["timestamp"],
                    "is_fraud": int(label),
                    "fraud_probability": float(probability),
                    "prediction": int(probability >= THRESHOLD),
                })
            test_history = pd.concat(
                [test_history, batch],
                ignore_index=True,
            )

    y_true = np.asarray(labels_true, dtype=np.int64)
    y_prob = np.asarray(probabilities, dtype=np.float64)
    y_pred = (y_prob >= THRESHOLD).astype(np.int64)
    has_both_classes = np.unique(y_true).size == 2
    metrics = {
        "model": "THGNN_temporal_heterogeneous_prototype",
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": (
            float(roc_auc_score(y_true, y_prob))
            if has_both_classes
            else float("nan")
        ),
        "pr_auc": (
            float(average_precision_score(y_true, y_prob))
            if has_both_classes
            else float("nan")
        ),
        "threshold": THRESHOLD,
        "train_transactions": int(len(train_df)),
        "test_transactions": int(len(test_df)),
        "test_frauds": int(y_true.sum()),
    }

    torch.save({
        "model_state_dict": model.cpu().state_dict(),
        "account_input_dim": int(first_graph["account"].x.shape[1]),
        "transaction_input_dim": int(first_graph["transaction"].x.shape[1]),
        "hidden_dim": HIDDEN_DIM,
        "dropout": DROPOUT,
        "threshold": THRESHOLD,
        "feature_columns": feature_columns,
    }, MODEL_FILE)
    joblib.dump({
        "account_scaler": account_scaler,
        "transaction_scaler": transaction_scaler,
        "feature_columns": feature_columns,
    }, PREPROCESSOR_FILE)
    pd.DataFrame(records).to_csv(PREDICTIONS_FILE, index=False)
    pd.DataFrame([metrics]).to_csv(METRICS_FILE, index=False)

    print("\nTHGNN chronological test metrics")
    for key, value in metrics.items():
        print(f"{key:24s}: {value}")
    print(f"\nSaved model: {MODEL_FILE}")
    print(f"Saved predictions: {PREDICTIONS_FILE}")
    print(f"Saved metrics: {METRICS_FILE}")


if __name__ == "__main__":
    main()
