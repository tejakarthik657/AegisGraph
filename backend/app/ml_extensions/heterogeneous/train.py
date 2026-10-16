
from __future__ import annotations

import random
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from app.utils.config import CLEAN_TRANSACTIONS_FILE
from app.ml_extensions.heterogeneous.model import (
    HeterogeneousFraudGNN,
)
from app.ml_extensions.heterogeneous.preprocessing import (
    build_heterogeneous_graph,
)


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

SEED = 42
TRAIN_RATIO = 0.80
EPOCHS = 20
LEARNING_RATE = 0.001
HIDDEN_DIM = 64
DROPOUT = 0.20
MAX_GRAD_NORM = 1.0
THRESHOLD = 0.50

BACKEND_DIR = Path(__file__).resolve().parents[3]
PROJECT_ROOT = BACKEND_DIR.parent
RESULTS_DIR = PROJECT_ROOT / "data" / "results"

MODEL_FILE = RESULTS_DIR / "heterogeneous_gnn_model.pt"
PREPROCESSOR_FILE = (
    RESULTS_DIR / "heterogeneous_gnn_preprocessor.joblib"
)
PREDICTIONS_FILE = (
    RESULTS_DIR / "heterogeneous_gnn_predictions.csv"
)
METRICS_FILE = (
    RESULTS_DIR / "heterogeneous_gnn_metrics.csv"
)


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    print("Loading transaction data...")

    df = pd.read_csv(CLEAN_TRANSACTIONS_FILE)

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        errors="raise",
    )

    df = (
        df.sort_values("timestamp")
        .reset_index(drop=True)
    )

    split_index = int(len(df) * TRAIN_RATIO)

    if split_index <= 0 or split_index >= len(df):
        raise ValueError("Invalid chronological train/test split.")

    train_df = df.iloc[:split_index].copy()
    test_df = df.iloc[split_index:].copy()

    print(f"Transactions: {len(df)}")
    print(f"Training rows: {len(train_df)}")
    print(f"Testing rows: {len(test_df)}")
    print(f"Training frauds: {int(train_df['is_fraud'].sum())}")
    print(f"Testing frauds: {int(test_df['is_fraud'].sum())}")

    if train_df["is_fraud"].nunique() < 2:
        raise ValueError(
            "Training data must contain both fraud classes."
        )

    if test_df["is_fraud"].nunique() < 2:
        raise ValueError(
            "Test data must contain both fraud classes."
        )

    return train_df, test_df


def evaluate_model(
    model: HeterogeneousFraudGNN,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    account_scaler,
    transaction_scaler,
    feature_columns: list[str],
    device: torch.device,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """
    Evaluate test transactions in chronological order.

    Each test graph contains:
      - account aggregates calculated from training history;
      - edges for the current test transaction only.

    This first evaluation intentionally does not update account
    history with test transactions. It therefore measures the
    model against the fixed training-history snapshot.
    """

    model.eval()

    probabilities: list[float] = []
    labels: list[int] = []
    transaction_ids: list[str] = []
    timestamps: list[pd.Timestamp] = []

    with torch.no_grad():
        for _, row in test_df.iterrows():
            current = row.to_frame().T

            graph, _, _, current_labels, current_columns = (
                build_heterogeneous_graph(
                    transactions=current,
                    history_transactions=train_df,
                    fit_scalers=False,
                    account_scaler=account_scaler,
                    transaction_scaler=transaction_scaler,
                    feature_columns=feature_columns,
                )
            )

            if current_columns != feature_columns:
                raise ValueError(
                    "Test feature schema differs from training schema."
                )

            graph = graph.to(device)

            logits = model(graph)
            probability = torch.sigmoid(logits).item()

            probabilities.append(float(probability))
            labels.append(int(current_labels[0]))
            transaction_ids.append(str(row["transaction_id"]))
            timestamps.append(pd.Timestamp(row["timestamp"]))

    y_true = np.asarray(labels, dtype=np.int64)
    y_probability = np.asarray(probabilities, dtype=np.float64)
    y_pred = (y_probability >= THRESHOLD).astype(np.int64)

    metrics = {
        "precision": float(
            precision_score(y_true, y_pred, zero_division=0)
        ),
        "recall": float(
            recall_score(y_true, y_pred, zero_division=0)
        ),
        "f1": float(
            f1_score(y_true, y_pred, zero_division=0)
        ),
        "roc_auc": float(
            roc_auc_score(y_true, y_probability)
        ),
        "pr_auc": float(
            average_precision_score(y_true, y_probability)
        ),
        "threshold": THRESHOLD,
        "test_transactions": int(len(y_true)),
        "test_frauds": int(y_true.sum()),
    }

    predictions = pd.DataFrame(
        {
            "transaction_id": transaction_ids,
            "timestamp": timestamps,
            "is_fraud": y_true,
            "fraud_probability": y_probability,
            "prediction": y_pred,
        }
    )

    print("\nHETEROGENEOUS GNN EVALUATION")
    for name, value in metrics.items():
        print(f"{name:20s}: {value}")

    print("\nClassification report:")
    print(
        classification_report(
            y_true,
            y_pred,
            labels=[0, 1],
            zero_division=0,
        )
    )

    print("Confusion matrix:")
    print(confusion_matrix(y_true, y_pred, labels=[0, 1]))

    return predictions, metrics


def main() -> None:
    set_seed()

    device = torch.device("cpu")
    print(f"Using device: {device}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    train_df, test_df = load_data()

    print("\nBuilding training graph...")

    (
        train_graph,
        account_scaler,
        transaction_scaler,
        train_labels,
        feature_columns,
    ) = build_heterogeneous_graph(
        transactions=train_df,
        fit_scalers=True,
    )

    train_graph = train_graph.to(device)

    model = HeterogeneousFraudGNN(
        account_input_dim=train_graph["account"].x.shape[1],
        transaction_input_dim=(
            train_graph["transaction"].x.shape[1]
        ),
        hidden_dim=HIDDEN_DIM,
        dropout=DROPOUT,
    ).to(device)

    labels_tensor = torch.tensor(
        train_labels,
        dtype=torch.float32,
        device=device,
    )

    positive_count = int(labels_tensor.sum().item())
    negative_count = len(labels_tensor) - positive_count

    if positive_count == 0:
        raise ValueError("No positive fraud examples in training data.")

    positive_weight = torch.tensor(
        [negative_count / positive_count],
        dtype=torch.float32,
        device=device,
    )

    criterion = nn.BCEWithLogitsLoss(
        pos_weight=positive_weight
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=1e-4,
    )

    print("\nHETEROGENEOUS GNN TRAINING")
    print(f"Account feature count: {train_graph['account'].x.shape[1]}")
    print(
        "Transaction feature count: "
        f"{train_graph['transaction'].x.shape[1]}"
    )
    print(f"Positive class weight: {positive_weight.item():.4f}")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)

        logits = model(train_graph)

        if logits.shape[0] != len(labels_tensor):
            raise RuntimeError(
                "Model output count does not match training labels."
            )

        loss = criterion(logits, labels_tensor)

        if not torch.isfinite(loss):
            raise RuntimeError(
                f"Non-finite training loss at epoch {epoch}."
            )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            MAX_GRAD_NORM,
        )

        optimizer.step()

        if epoch == 1 or epoch % 2 == 0:
            print(
                f"Epoch {epoch:02d}/{EPOCHS} "
                f"| Loss: {loss.item():.4f}"
            )

    print("\nSaving model and preprocessing artifacts...")

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "account_input_dim": int(
                train_graph["account"].x.shape[1]
            ),
            "transaction_input_dim": int(
                train_graph["transaction"].x.shape[1]
            ),
            "hidden_dim": HIDDEN_DIM,
            "dropout": DROPOUT,
            "threshold": THRESHOLD,
            "seed": SEED,
        },
        MODEL_FILE,
    )

    joblib.dump(
        {
            "account_scaler": account_scaler,
            "transaction_scaler": transaction_scaler,
            "feature_columns": feature_columns,
            "account_feature_columns": [
                "in_degree",
                "out_degree",
                "total_incoming",
                "total_outgoing",
                "unique_neighbors",
            ],
        },
        PREPROCESSOR_FILE,
    )

    predictions, metrics = evaluate_model(
        model=model,
        train_df=train_df,
        test_df=test_df,
        account_scaler=account_scaler,
        transaction_scaler=transaction_scaler,
        feature_columns=feature_columns,
        device=device,
    )

    predictions.to_csv(
        PREDICTIONS_FILE,
        index=False,
    )

    pd.DataFrame(
        [metrics]
    ).to_csv(
        METRICS_FILE,
        index=False,
    )

    print("\nArtifacts saved:")
    print(MODEL_FILE)
    print(PREPROCESSOR_FILE)
    print(PREDICTIONS_FILE)
    print(METRICS_FILE)

    print("\nHETEROGENEOUS GNN TRAINING COMPLETE")


if __name__ == "__main__":
    main()
