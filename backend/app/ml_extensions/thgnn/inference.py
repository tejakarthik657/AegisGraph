"""Load a trained THGNN and score transaction rows with historical context."""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
import torch

from app.ml_extensions.thgnn.model import TemporalHeterogeneousFraudGNN
from app.ml_extensions.thgnn.preprocessing import build_thgnn_graph
from app.utils.config import RESULTS_DATA_DIR

MODEL_FILE = RESULTS_DATA_DIR / "thgnn_model.pt"
PREPROCESSOR_FILE = RESULTS_DATA_DIR / "thgnn_preprocessor.joblib"


def score_transactions(
    transactions: pd.DataFrame,
    history_transactions: pd.DataFrame,
    *,
    model_path: str | Path = MODEL_FILE,
    preprocessor_path: str | Path = PREPROCESSOR_FILE,
    device: str | None = None,
) -> pd.DataFrame:
    """Return input rows with ``thgnn_score`` and ``thgnn_prediction``.

    ``history_transactions`` must contain only events earlier than the target
    rows. This API scores events independently against that fixed history
    snapshot; it does not update memory with each scored event.
    """
    model_path, preprocessor_path = Path(model_path), Path(preprocessor_path)
    if not model_path.exists():
        raise FileNotFoundError(f"THGNN checkpoint not found: {model_path}. Train it first.")
    if not preprocessor_path.exists():
        raise FileNotFoundError(
            f"THGNN preprocessor not found: {preprocessor_path}. Train it first."
        )
    if transactions.empty:
        result = transactions.copy()
        result["thgnn_score"] = pd.Series(dtype=float)
        result["thgnn_prediction"] = pd.Series(dtype=int)
        return result
    device_obj = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    checkpoint = torch.load(model_path, map_location=device_obj, weights_only=False)
    preprocessor = joblib.load(preprocessor_path)
    model = TemporalHeterogeneousFraudGNN(
        account_input_dim=int(checkpoint["account_input_dim"]),
        transaction_input_dim=int(checkpoint["transaction_input_dim"]),
        hidden_dim=int(checkpoint["hidden_dim"]),
        dropout=float(checkpoint["dropout"]),
    ).to(device_obj)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    targets = transactions.copy()
    targets["timestamp"] = pd.to_datetime(
        targets["timestamp"],
        errors="raise",
    )
    if "is_fraud" not in targets.columns:
        targets["is_fraud"] = 0

    history = history_transactions.copy()
    if history.empty:
        history = targets.iloc[0:0].copy()
    else:
        history["timestamp"] = pd.to_datetime(
            history["timestamp"],
            errors="raise",
        )
        if "is_fraud" not in history.columns:
            history["is_fraud"] = 0
        history = history.sort_values(
            "timestamp",
            kind="mergesort",
        ).reset_index(drop=True)
    targets = targets.sort_values(
        "timestamp",
        kind="mergesort",
    ).reset_index(drop=True)

    scores: list[float] = []
    with torch.no_grad():
        for _, batch in targets.groupby("timestamp", sort=True):
            graph, _, _, _, _ = build_thgnn_graph(
                batch,
                fit_scalers=False,
                account_scaler=preprocessor["account_scaler"],
                transaction_scaler=preprocessor["transaction_scaler"],
                feature_columns=preprocessor["feature_columns"],
                history_transactions=history,
            )
            batch_scores = torch.sigmoid(
                model(graph.to(device_obj))
            ).cpu().tolist()
            scores.extend(float(score) for score in batch_scores)
            history = pd.concat(
                [history, batch],
                ignore_index=True,
            )

    targets["thgnn_score"] = scores
    targets["thgnn_prediction"] = (
        targets["thgnn_score"] >= float(checkpoint.get("threshold", 0.5))
    ).astype(int)
    return targets
