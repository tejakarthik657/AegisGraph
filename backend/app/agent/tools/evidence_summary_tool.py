"""AegisGraph — Evidence Summary Tool."""

from __future__ import annotations
from app.evidence_reporting.evidence_chain import EvidenceChain


def build_evidence_summary(state_dict: dict) -> str:
    """Build a textual evidence summary from agent state."""
    parts = []
    if state_dict.get("account_info"):
        parts.append(f"Account analysis: {len(state_dict['account_info'])} account(s) examined.")
    if state_dict.get("graph_context"):
        parts.append(f"Graph context: neighborhood analysis completed.")
    if state_dict.get("velocity_analysis"):
        parts.append(f"Velocity analysis: completed.")
    if state_dict.get("pattern_results"):
        parts.append(f"Pattern detection: completed.")
    if state_dict.get("risk_assessment"):
        score = state_dict["risk_assessment"].get("risk_score", 0.0)
        parts.append(f"Risk assessment: score = {score:.4f}")
    return " | ".join(parts) if parts else "No evidence collected."
