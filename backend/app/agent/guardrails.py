"""
AegisGraph — Agent Guardrails.

Safety constraints and validation rules to prevent
the investigation agent from making inappropriate
decisions or accessing unauthorized data.
"""

from __future__ import annotations

from typing import Any


# Maximum investigation depth
MAX_HOPS = 3
MAX_TRANSACTIONS_PER_QUERY = 1000
MAX_INVESTIGATION_TIME_SECONDS = 300

# Confidence thresholds
MIN_CONFIDENCE_FOR_ESCALATION = 0.60
MIN_CONFIDENCE_FOR_CLOSURE = 0.70

# Prohibited actions
PROHIBITED_ACTIONS = {
    "DELETE_TRANSACTION",
    "MODIFY_FRAUD_LABEL",
    "REVERSE_TRANSACTION",
}


class GuardrailViolation(ValueError):
    """Raised when an agent action violates a guardrail."""


def validate_action(action: str) -> bool:
    """
    Validate that an agent action is permitted.

    Raises GuardrailViolation if the action is prohibited.
    """

    normalized = action.strip().upper()

    if normalized in PROHIBITED_ACTIONS:
        raise GuardrailViolation(
            f"Action '{action}' is prohibited by guardrails."
        )

    return True


def validate_query_scope(
    num_transactions: int,
    num_hops: int,
) -> bool:
    """
    Validate that a graph query doesn't exceed scope limits.
    """

    if num_transactions > MAX_TRANSACTIONS_PER_QUERY:
        raise GuardrailViolation(
            f"Query scope too large: {num_transactions} transactions "
            f"(max: {MAX_TRANSACTIONS_PER_QUERY})"
        )

    if num_hops > MAX_HOPS:
        raise GuardrailViolation(
            f"Query depth too large: {num_hops} hops "
            f"(max: {MAX_HOPS})"
        )

    return True


def validate_recommendation(
    recommendation: str,
    confidence: float,
) -> bool:
    """
    Validate that a recommendation meets confidence requirements.
    """

    normalized = recommendation.strip().upper()

    if normalized == "ESCALATE" and confidence < MIN_CONFIDENCE_FOR_ESCALATION:
        raise GuardrailViolation(
            f"Escalation requires confidence >= "
            f"{MIN_CONFIDENCE_FOR_ESCALATION}, "
            f"got {confidence:.2f}"
        )

    if normalized == "CLOSE" and confidence < MIN_CONFIDENCE_FOR_CLOSURE:
        raise GuardrailViolation(
            f"Closure requires confidence >= "
            f"{MIN_CONFIDENCE_FOR_CLOSURE}, "
            f"got {confidence:.2f}"
        )

    return True
