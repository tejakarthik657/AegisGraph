"""AegisGraph — Risk Assessment Tool."""

from __future__ import annotations


def assess_risk(transaction_id: str, risk_data: dict | None = None) -> dict:
    """Assess risk for a specific transaction."""
    if risk_data is None:
        risk_data = {}
    return {
        "transaction_id": transaction_id,
        "risk_score": risk_data.get("risk_score", 0.0),
        "risk_category": risk_data.get("risk_category", "UNKNOWN"),
        "components": {
            "ml_score": risk_data.get("rf_probability", 0.0),
            "tgn_score": risk_data.get("tgn_probability", 0.0),
            "rule_score": risk_data.get("rule_score", 0.0),
        },
    }
