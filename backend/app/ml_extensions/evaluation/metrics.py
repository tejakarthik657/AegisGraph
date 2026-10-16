"""
Reusable evaluation metrics for AegisGraph fraud detection models.

This module contains evaluation utilities only.
It does not train models or create train/test splits.
"""

from typing import Dict

import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def calculate_binary_metrics(
    y_true,
    y_pred,
    y_probability=None,
) -> Dict[str, float]:
    """
    Calculate standard binary classification metrics.

    Parameters
    ----------
    y_true:
        Ground-truth fraud labels.

    y_pred:
        Binary model predictions.

    y_probability:
        Fraud probabilities. Required for ROC-AUC and PR-AUC.

    Returns
    -------
    dict
        Dictionary containing classification metrics.
    """

    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(
            precision_score(y_true, y_pred, zero_division=0)
        ),
        "recall": float(
            recall_score(y_true, y_pred, zero_division=0)
        ),
        "f1": float(
            f1_score(y_true, y_pred, zero_division=0)
        ),
    }

    if y_probability is not None:
        y_probability = np.asarray(y_probability, dtype=float)

        metrics["roc_auc"] = float(
            roc_auc_score(y_true, y_probability)
        )

        metrics["pr_auc"] = float(
            average_precision_score(y_true, y_probability)
        )

    return metrics


def calculate_confusion_matrix(
    y_true,
    y_pred,
) -> Dict[str, int]:
    """
    Calculate confusion-matrix components.

    Returns
    -------
    dict
        TN, FP, FN, and TP counts.
    """

    matrix = confusion_matrix(
        np.asarray(y_true).astype(int),
        np.asarray(y_pred).astype(int),
        labels=[0, 1],
    )

    tn, fp, fn, tp = matrix.ravel()

    return {
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
    }


def calculate_typology_detection(
    df: pd.DataFrame,
    label_column: str = "is_fraud",
    typology_column: str = "typology",
    prediction_column: str = "prediction",
) -> pd.DataFrame:
    """
    Calculate fraud detection counts by typology.

    This measures how many fraud cases of each typology
    were detected by the model.

    It is intentionally a detection/recall-style analysis,
    not a one-vs-rest precision/F1 calculation.
    """

    fraud_df = df[df[label_column] == 1].copy()

    if fraud_df.empty:
        return pd.DataFrame(
            columns=[
                "total",
                "detected",
                "missed",
                "recall",
            ]
        )

    results = []

    for typology, group in fraud_df.groupby(typology_column):
        total = len(group)
        detected = int((group[prediction_column] == 1).sum())
        missed = total - detected

        results.append(
            {
                "typology": typology,
                "total": total,
                "detected": detected,
                "missed": missed,
                "recall": detected / total if total else 0.0,
            }
        )

    result = pd.DataFrame(results)

    if not result.empty:
        result = result.sort_values("typology").reset_index(drop=True)

    return result