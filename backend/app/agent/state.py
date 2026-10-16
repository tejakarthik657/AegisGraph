"""
AegisGraph — Agent State.

Defines the state schema for the investigation agent,
tracking what information has been gathered and what
decisions have been made.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class AgentState:
    """
    Tracks the investigation agent's state as it
    processes an alert or case.
    """

    # Identity
    case_id: str = ""
    alert_id: str = ""
    transaction_id: str = ""

    # Current step
    current_step: str = "INIT"
    steps_completed: list[str] = field(
        default_factory=list
    )

    # Gathered evidence
    account_info: dict[str, Any] = field(
        default_factory=dict
    )

    transaction_details: dict[str, Any] = field(
        default_factory=dict
    )

    graph_context: dict[str, Any] = field(
        default_factory=dict
    )

    velocity_analysis: dict[str, Any] = field(
        default_factory=dict
    )

    pattern_results: dict[str, Any] = field(
        default_factory=dict
    )

    risk_assessment: dict[str, Any] = field(
        default_factory=dict
    )

    regulatory_references: list[dict] = field(
        default_factory=list
    )

    evidence_summary: str = ""

    # Decision
    recommendation: str = ""
    confidence: float = 0.0

    # Metadata
    started_at: str = field(
        default_factory=lambda: datetime.utcnow().isoformat()
    )

    errors: list[str] = field(default_factory=list)

    def mark_step_complete(self, step: str) -> None:
        """Record a completed step."""
        if step not in self.steps_completed:
            self.steps_completed.append(step)
        self.current_step = step

    def add_error(self, error: str) -> None:
        """Record an error encountered during processing."""
        self.errors.append(error)

    @property
    def is_complete(self) -> bool:
        """Whether the agent has finished its investigation."""
        return self.current_step == "FINISHED"

    def to_dict(self) -> dict:
        """Serialize the state."""
        return {
            "case_id": self.case_id,
            "alert_id": self.alert_id,
            "transaction_id": self.transaction_id,
            "current_step": self.current_step,
            "steps_completed": self.steps_completed,
            "account_info": self.account_info,
            "transaction_details": self.transaction_details,
            "graph_context": self.graph_context,
            "velocity_analysis": self.velocity_analysis,
            "pattern_results": self.pattern_results,
            "risk_assessment": self.risk_assessment,
            "regulatory_references": self.regulatory_references,
            "evidence_summary": self.evidence_summary,
            "recommendation": self.recommendation,
            "confidence": self.confidence,
            "started_at": self.started_at,
            "errors": self.errors,
        }
