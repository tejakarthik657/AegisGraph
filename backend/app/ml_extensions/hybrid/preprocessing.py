"""
AegisGraph — Hybrid Model Preprocessing.

Aligns and combines features from TGN and Heterogeneous
GNN pipelines for the hybrid model.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import numpy as np


def align_embeddings(
    tgn_predictions_path: Path,
    hgnn_predictions_path: Path,
    transactions_path: Path,
) -> dict:
    """
    Align predictions from TGN and HGNN models with
    the transaction dataset.

    Returns
    -------
    Dict with aligned arrays and metadata.
    """

    transactions = pd.read_csv(transactions_path)

    tgn_exists = tgn_predictions_path.exists()
    hgnn_exists = hgnn_predictions_path.exists()

    result = {
        "transaction_ids": transactions["transaction_id"].values,
        "labels": transactions["is_fraud"].values if "is_fraud" in transactions.columns else None,
        "tgn_available": tgn_exists,
        "hgnn_available": hgnn_exists,
    }

    if tgn_exists:
        tgn_df = pd.read_csv(tgn_predictions_path)
        merged = transactions.merge(
            tgn_df[["transaction_id", "fraud_probability"]],
            on="transaction_id",
            how="left",
        )
        result["tgn_probabilities"] = merged[
            "fraud_probability"
        ].fillna(0.0).values
    else:
        result["tgn_probabilities"] = np.zeros(len(transactions))

    if hgnn_exists:
        hgnn_df = pd.read_csv(hgnn_predictions_path)
        if "transaction_id" in hgnn_df.columns:
            merged = transactions.merge(
                hgnn_df[["transaction_id", "fraud_probability"]],
                on="transaction_id",
                how="left",
            )
            result["hgnn_probabilities"] = merged[
                "fraud_probability"
            ].fillna(0.0).values
        else:
            result["hgnn_probabilities"] = np.zeros(len(transactions))
    else:
        result["hgnn_probabilities"] = np.zeros(len(transactions))

    return result
