"""AegisGraph — Finish Tool."""

from __future__ import annotations


def finalize_investigation(state_dict: dict) -> dict:
    """Finalize the investigation and produce a recommendation."""
    risk_score = 0.0
    if state_dict.get("risk_assessment"):
        risk_score = float(state_dict["risk_assessment"].get("risk_score", 0.0))
    if risk_score >= 0.75:
        recommendation = "ESCALATE"
        confidence = min(risk_score, 0.95)
    elif risk_score >= 0.50:
        recommendation = "REVIEW"
        confidence = 0.70
    elif risk_score >= 0.25:
        recommendation = "MONITOR"
        confidence = 0.60
    else:
        recommendation = "CLOSE"
        confidence = 0.80
    return {
        "recommendation": recommendation,
        "confidence": confidence,
        "risk_score": risk_score,
        "summary": state_dict.get("evidence_summary", ""),
    }
