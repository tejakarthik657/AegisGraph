"""
AegisGraph — GenAI Narrative Generator.

Generates human-readable investigation narratives
from structured evidence data. Uses template-based
generation (no external LLM dependency required).
"""

from __future__ import annotations

from datetime import datetime
from typing import Any


def generate_narrative(
    case_data: dict,
    evidence_chain: dict | None = None,
) -> str:
    """
    Generate a comprehensive investigation narrative
    from case data and evidence.

    Parameters
    ----------
    case_data:
        Case record with all available fields.

    evidence_chain:
        Optional evidence chain dict.

    Returns
    -------
    A multi-paragraph narrative text.
    """

    case_id = case_data.get("case_id", "UNKNOWN")
    alert_id = case_data.get("alert_id", "N/A")
    transaction_id = case_data.get("transaction_id", "N/A")
    sender = case_data.get("sender_account", "N/A")
    receiver = case_data.get("receiver_account", "N/A")
    amount = float(case_data.get("amount", 0.0))
    risk_score = float(case_data.get("risk_score", 0.0))
    risk_category = case_data.get("risk_category", "UNKNOWN")
    typology = case_data.get("typology", "none")

    # Build narrative sections
    sections = []

    # Section 1: Executive Summary
    sections.append(
        f"Investigation case {case_id} was initiated following "
        f"alert {alert_id} concerning transaction {transaction_id}. "
        f"The transaction of ${amount:,.2f} was sent from account "
        f"{sender} to account {receiver}. "
        f"The combined risk assessment yielded a score of "
        f"{risk_score:.4f}, classified as {risk_category}."
    )

    # Section 2: Risk Analysis
    if risk_score >= 0.75:
        sections.append(
            "The risk score places this transaction in the "
            "critical risk category, indicating a high probability "
            "of suspicious activity. Immediate review and potential "
            "regulatory reporting are recommended."
        )
    elif risk_score >= 0.50:
        sections.append(
            "The elevated risk score suggests this transaction "
            "warrants thorough investigation. Multiple risk factors "
            "contributed to the overall assessment."
        )
    else:
        sections.append(
            "While the risk score falls below the critical threshold, "
            "the transaction was flagged for review based on "
            "specific detection signals."
        )

    # Section 3: Typology
    if typology and typology != "none":
        sections.append(
            f"The detected typology is '{typology}', which is "
            f"consistent with known money laundering patterns. "
            f"Investigators should examine related transactions "
            f"for similar behavioral indicators."
        )

    # Section 4: Evidence Summary
    if evidence_chain:
        evidence_items = evidence_chain.get("evidence", [])
        total_items = len(evidence_items)

        if total_items > 0:
            type_counts = {}
            for item in evidence_items:
                t = item.get("evidence_type", "UNKNOWN")
                type_counts[t] = type_counts.get(t, 0) + 1

            evidence_parts = [
                f"{count} {t.lower().replace('_', ' ')} item(s)"
                for t, count in type_counts.items()
            ]

            sections.append(
                f"The investigation compiled {total_items} pieces "
                f"of evidence: {', '.join(evidence_parts)}. "
                f"Each evidence item has been documented in the "
                f"attached evidence chain."
            )

    # Section 5: Conclusion
    sections.append(
        f"This narrative was generated on "
        f"{datetime.utcnow().strftime('%Y-%m-%d at %H:%M UTC')} "
        f"based on the available data. Investigators should "
        f"review the full evidence chain and make a determination "
        f"based on the totality of findings."
    )

    return "\n\n".join(sections)
