
from enum import Enum


class CaseStatus(str, Enum):
    OPEN = "OPEN"
    UNDER_REVIEW = "UNDER_REVIEW"
    ESCALATED = "ESCALATED"
    CLOSED = "CLOSED"


ALLOWED_TRANSITIONS: dict[CaseStatus, set[CaseStatus]] = {
    CaseStatus.OPEN: {
        CaseStatus.UNDER_REVIEW,
        CaseStatus.ESCALATED,
        CaseStatus.CLOSED,
    },
    CaseStatus.UNDER_REVIEW: {
        CaseStatus.OPEN,
        CaseStatus.ESCALATED,
        CaseStatus.CLOSED,
    },
    CaseStatus.ESCALATED: {
        CaseStatus.UNDER_REVIEW,
        CaseStatus.CLOSED,
    },
    CaseStatus.CLOSED: set(),
}


class InvalidCaseTransitionError(ValueError):
    """Raised when a case status or transition is invalid."""


def normalize_status(status: str | CaseStatus) -> CaseStatus:
    """Convert a status string into a validated CaseStatus."""
    if isinstance(status, CaseStatus):
        return status

    if not isinstance(status, str) or not status.strip():
        raise InvalidCaseTransitionError(
            "Case status must be a non-empty string."
        )

    try:
        return CaseStatus(status.strip().upper())
    except ValueError as exc:
        valid_statuses = ", ".join(item.value for item in CaseStatus)
        raise InvalidCaseTransitionError(
            f"Unknown case status '{status}'. "
            f"Valid statuses: {valid_statuses}"
        ) from exc


def validate_transition(
    current_status: str | CaseStatus,
    target_status: str | CaseStatus,
) -> bool:
    """Return True if the requested status transition is permitted."""
    current = normalize_status(current_status)
    target = normalize_status(target_status)

    if current == target:
        raise InvalidCaseTransitionError(
            f"Case is already in status {current.value}."
        )

    if target not in ALLOWED_TRANSITIONS[current]:
        raise InvalidCaseTransitionError(
            f"Transition from {current.value} "
            f"to {target.value} is not allowed."
        )

    return True


def get_allowed_transitions(
    current_status: str | CaseStatus,
) -> list[str]:
    """Return the allowed next statuses in a predictable order."""
    current = normalize_status(current_status)

    return sorted(
        status.value for status in ALLOWED_TRANSITIONS[current]
    )
