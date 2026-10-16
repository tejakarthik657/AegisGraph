"""
AegisGraph — Hybrid Model Training.

Training pipeline for the hybrid TGN + HGNN model.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def train_hybrid(
    tgn_predictions_path: Path | None = None,
    hgnn_predictions_path: Path | None = None,
    transactions_path: Path | None = None,
    output_path: Path | None = None,
) -> dict:
    """
    Train the hybrid model by combining TGN and HGNN outputs.

    For the initial version, this uses a simple weighted
    ensemble rather than end-to-end training.

    Returns
    -------
    Training results dict with metrics.
    """

    from app.utils.config import (
        CLEAN_TRANSACTIONS_FILE,
        RESULTS_DATA_DIR,
    )

    if transactions_path is None:
        transactions_path = CLEAN_TRANSACTIONS_FILE

    if output_path is None:
        output_path = RESULTS_DATA_DIR / "hybrid_predictions.csv"

    if tgn_predictions_path is None:
        tgn_predictions_path = (
            RESULTS_DATA_DIR / "temporal_tgn_predictions.csv"
        )

    if hgnn_predictions_path is None:
        hgnn_predictions_path = (
            RESULTS_DATA_DIR / "heterogeneous_gnn_predictions.csv"
        )

    transactions = pd.read_csv(transactions_path)

    # Default weights
    tgn_weight = 0.5
    hgnn_weight = 0.5

    # Load available predictions
    hybrid_probs = np.zeros(len(transactions))

    if tgn_predictions_path.exists():
        tgn_df = pd.read_csv(tgn_predictions_path)
        merged = transactions.merge(
            tgn_df[["transaction_id", "fraud_probability"]].rename(
                columns={"fraud_probability": "tgn_prob"}
            ),
            on="transaction_id",
            how="left",
        )
        tgn_probs = merged["tgn_prob"].fillna(0.0).values
        hybrid_probs += tgn_weight * tgn_probs

    if hgnn_predictions_path.exists():
        hgnn_df = pd.read_csv(hgnn_predictions_path)
        if "transaction_id" in hgnn_df.columns:
            merged = transactions.merge(
                hgnn_df[["transaction_id", "fraud_probability"]].rename(
                    columns={"fraud_probability": "hgnn_prob"}
                ),
                on="transaction_id",
                how="left",
            )
            hgnn_probs = merged["hgnn_prob"].fillna(0.0).values
            hybrid_probs += hgnn_weight * hgnn_probs

    results = pd.DataFrame({
        "transaction_id": transactions["transaction_id"],
        "fraud_probability": hybrid_probs,
        "predicted_fraud": (hybrid_probs >= 0.5).astype(int),
    })

    output_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_path, index=False)

    print(f"Hybrid predictions saved to: {output_path}")
    print(f"Total predictions: {len(results)}")

    return {
        "total_predictions": len(results),
        "output_path": str(output_path),
        "tgn_weight": tgn_weight,
        "hgnn_weight": hgnn_weight,
    }


if __name__ == "__main__":
    train_hybrid()
