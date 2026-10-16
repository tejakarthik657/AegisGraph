"""AegisGraph — RAG Query Generator."""

from __future__ import annotations


def generate_query(case_data: dict) -> str:
    """Generate a regulatory query from case context."""
    parts = []
    amount = case_data.get("amount", 0)
    if amount and float(amount) > 10000:
        parts.append("CTR filing requirements large transactions")
    risk_category = case_data.get("risk_category", "")
    if risk_category in ("HIGH", "CRITICAL"):
        parts.append("SAR filing suspicious activity")
    typology = case_data.get("typology", "")
    if typology:
        parts.append(f"{typology} AML regulations")
    return " ".join(parts) if parts else "AML compliance requirements"
