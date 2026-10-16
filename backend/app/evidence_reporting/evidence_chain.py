"""
AegisGraph — Evidence Chain Builder.

Constructs a chronological chain of evidence for
a case or alert, linking transactions, rules, patterns,
and risk scores into a coherent investigation narrative.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any


class EvidenceItem:
    """A single piece of evidence in the chain."""

    def __init__(
        self,
        evidence_type: str,
        source: str,
        description: str,
        data: dict[str, Any] | None = None,
        timestamp: str | None = None,
        severity: str = "INFO",
    ):
        self.evidence_type = evidence_type
        self.source = source
        self.description = description
        self.data = data or {}
        self.timestamp = timestamp or datetime.utcnow().isoformat()
        self.severity = severity

    def to_dict(self) -> dict:
        return {
            "evidence_type": self.evidence_type,
            "source": self.source,
            "description": self.description,
            "data": self.data,
            "timestamp": self.timestamp,
            "severity": self.severity,
        }


class EvidenceChain:
    """
    Ordered collection of evidence items for an investigation.
    """

    def __init__(self, case_id: str):
        self.case_id = case_id
        self.items: list[EvidenceItem] = []
        self.created_at = datetime.utcnow().isoformat()

    def add(self, item: EvidenceItem) -> None:
        """Add an evidence item to the chain."""
        self.items.append(item)

    def add_transaction_evidence(
        self,
        transaction: dict,
    ) -> None:
        """Add a transaction as evidence."""
        self.add(
            EvidenceItem(
                evidence_type="TRANSACTION",
                source="transaction_data",
                description=(
                    f"Transaction {transaction.get('transaction_id', 'N/A')} "
                    f"of {transaction.get('amount', 0)} "
                    f"from {transaction.get('sender_account', 'N/A')} "
                    f"to {transaction.get('receiver_account', 'N/A')}"
                ),
                data=transaction,
                timestamp=str(
                    transaction.get("timestamp", "")
                ),
                severity="INFO",
            )
        )

    def add_rule_evidence(
        self,
        rule_name: str,
        triggered: bool,
        details: dict | None = None,
    ) -> None:
        """Add a rule trigger as evidence."""
        if triggered:
            self.add(
                EvidenceItem(
                    evidence_type="RULE_TRIGGER",
                    source=f"rule:{rule_name}",
                    description=f"Rule '{rule_name}' was triggered.",
                    data=details or {},
                    severity="WARNING",
                )
            )

    def add_pattern_evidence(
        self,
        pattern_name: str,
        details: dict | None = None,
    ) -> None:
        """Add a pattern detection as evidence."""
        self.add(
            EvidenceItem(
                evidence_type="PATTERN_MATCH",
                source=f"pattern:{pattern_name}",
                description=f"Pattern '{pattern_name}' detected.",
                data=details or {},
                severity="HIGH",
            )
        )

    def add_risk_score_evidence(
        self,
        score: float,
        category: str,
        components: dict | None = None,
    ) -> None:
        """Add a risk score assessment as evidence."""
        self.add(
            EvidenceItem(
                evidence_type="RISK_ASSESSMENT",
                source="risk_fusion",
                description=(
                    f"Risk score: {score:.4f} "
                    f"(category: {category})"
                ),
                data={
                    "risk_score": score,
                    "risk_category": category,
                    **(components or {}),
                },
                severity=category,
            )
        )

    def to_dict(self) -> dict:
        """Serialize the full evidence chain."""
        return {
            "case_id": self.case_id,
            "created_at": self.created_at,
            "total_evidence_items": len(self.items),
            "evidence": [
                item.to_dict() for item in self.items
            ],
        }

    @property
    def summary(self) -> str:
        """Generate a brief text summary of the evidence chain."""
        type_counts = {}
        for item in self.items:
            t = item.evidence_type
            type_counts[t] = type_counts.get(t, 0) + 1

        parts = [
            f"{count} {etype.lower().replace('_', ' ')}(s)"
            for etype, count in type_counts.items()
        ]

        return (
            f"Case {self.case_id}: "
            + ", ".join(parts)
            if parts
            else f"Case {self.case_id}: no evidence collected."
        )
