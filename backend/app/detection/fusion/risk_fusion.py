"""
AegisGraph — Risk Fusion Engine.

Combines signals from ML models, rules, graph scoring,
and pattern detection into a final risk score.
"""

from __future__ import annotations

import pandas as pd


# Weight configuration for each signal source
DEFAULT_WEIGHTS = {
    "ml_score": 0.35,
    "rule_score": 0.25,
    "pattern_score": 0.20,
    "graph_score": 0.20,
}


def fuse_risk_scores(
    transactions: pd.DataFrame,
    weights: dict[str, float] | None = None,
) -> pd.DataFrame:
    """
    Combine multiple risk signals into a final risk score.

    Parameters
    ----------
    transactions:
        DataFrame that may contain:
        - ml_score (from ML models)
        - rule_score (from AML rules)
        - pattern_score (from pattern detectors)
        - graph_score (from graph neighborhood scoring)

    weights:
        Optional weight overrides.

    Returns
    -------
    DataFrame with added `fused_risk_score` and
    `fused_risk_category` columns.
    """

    df = transactions.copy()

    if weights is None:
        weights = DEFAULT_WEIGHTS.copy()

    # Ensure all score columns exist with defaults
    score_columns = list(weights.keys())

    for col in score_columns:
        if col not in df.columns:
            df[col] = 0.0
        else:
            df[col] = df[col].fillna(0.0)

    # Weighted combination
    df["fused_risk_score"] = sum(
        weights[col] * df[col]
        for col in score_columns
    )

    # Clamp to [0, 1]
    df["fused_risk_score"] = df["fused_risk_score"].clip(
        0.0, 1.0
    )

    # Risk categories
    df["fused_risk_category"] = df[
        "fused_risk_score"
    ].apply(categorize_risk)

    return df


def categorize_risk(score: float) -> str:
    """Assign a risk category based on score."""

    if score >= 0.75:
        return "CRITICAL"
    if score >= 0.50:
        return "HIGH"
    if score >= 0.25:
        return "MEDIUM"
    return "LOW"
