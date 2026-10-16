"""
AegisGraph — Report Generator.

Generates structured investigation reports from
evidence chains and case data.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any


TEMPLATES_DIR = Path(__file__).parent / "templates"


def generate_report(
    case_data: dict,
    evidence_chain: dict | None = None,
    narrative: str | None = None,
) -> dict:
    """
    Generate a structured investigation report.

    Parameters
    ----------
    case_data:
        Case record dict with case_id, case_status, etc.

    evidence_chain:
        Serialized evidence chain from EvidenceChain.to_dict().

    narrative:
        Optional AI-generated narrative summary.

    Returns
    -------
    A structured report dict.
    """

    case_id = case_data.get("case_id", "UNKNOWN")

    report = {
        "report_id": f"RPT-{case_id}-{datetime.utcnow().strftime('%Y%m%dT%H%M%S')}",
        "case_id": case_id,
        "generated_at": datetime.utcnow().isoformat(),
        "report_type": "INVESTIGATION_REPORT",

        "case_summary": {
            "case_id": case_id,
            "status": case_data.get("case_status", "UNKNOWN"),
            "priority": case_data.get("case_priority", "UNKNOWN"),
            "alert_id": case_data.get("alert_id", "N/A"),
            "transaction_id": case_data.get(
                "transaction_id", "N/A"
            ),
            "amount": case_data.get("amount", 0.0),
            "risk_score": case_data.get("risk_score", 0.0),
            "risk_category": case_data.get(
                "risk_category", "UNKNOWN"
            ),
        },

        "evidence": evidence_chain or {"evidence": []},

        "narrative": narrative or _generate_default_narrative(
            case_data
        ),

        "recommendations": _generate_recommendations(
            case_data
        ),
    }

    return report


def _generate_default_narrative(
    case_data: dict,
) -> str:
    """Generate a default narrative from case data."""

    case_id = case_data.get("case_id", "UNKNOWN")
    risk_category = case_data.get("risk_category", "UNKNOWN")
    risk_score = case_data.get("risk_score", 0.0)
    amount = case_data.get("amount", 0.0)

    return (
        f"Investigation case {case_id} involves a transaction "
        f"of {amount:.2f} with a risk score of {risk_score:.4f} "
        f"classified as {risk_category}. "
        f"This case was generated for review based on "
        f"the combined assessment of machine learning models, "
        f"rule-based detection, and network analysis."
    )


def _generate_recommendations(
    case_data: dict,
) -> list[str]:
    """Generate action recommendations based on case data."""

    recommendations = []
    risk_score = float(case_data.get("risk_score", 0.0))

    if risk_score >= 0.75:
        recommendations.append(
            "IMMEDIATE REVIEW: This case has a critical risk score "
            "and should be prioritized for investigation."
        )
        recommendations.append(
            "Consider filing a Suspicious Activity Report (SAR)."
        )
    elif risk_score >= 0.50:
        recommendations.append(
            "HIGH PRIORITY: Schedule review within 24 hours."
        )
        recommendations.append(
            "Examine related transactions and account history."
        )
    elif risk_score >= 0.25:
        recommendations.append(
            "STANDARD REVIEW: Add to review queue."
        )
    else:
        recommendations.append(
            "LOW PRIORITY: Monitor but no immediate action required."
        )

    return recommendations


def format_report_text(report: dict) -> str:
    """Format a report dict into readable text."""

    lines = [
        "=" * 70,
        f"INVESTIGATION REPORT: {report['report_id']}",
        "=" * 70,
        "",
        f"Case ID: {report['case_summary']['case_id']}",
        f"Status: {report['case_summary']['status']}",
        f"Priority: {report['case_summary']['priority']}",
        f"Risk Score: {report['case_summary']['risk_score']}",
        f"Risk Category: {report['case_summary']['risk_category']}",
        f"Amount: {report['case_summary']['amount']}",
        "",
        "--- NARRATIVE ---",
        report["narrative"],
        "",
        "--- RECOMMENDATIONS ---",
    ]

    for i, rec in enumerate(report["recommendations"], 1):
        lines.append(f"  {i}. {rec}")

    lines.append("")
    lines.append(f"Generated: {report['generated_at']}")
    lines.append("=" * 70)

    return "\n".join(lines)
