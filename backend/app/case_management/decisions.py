
from app.case_management.review_actions import (
    ReviewAction,
    normalize_review_action,
)


# Define the workflow outcome of each investigator action.
# These mappings represent workflow intent, not confirmed fraud labels.
ACTION_STATUS_MAP = {
    ReviewAction.MARK_AS_FRAUD: "ESCALATED",
    ReviewAction.MARK_AS_LEGITIMATE: "CLOSED",
    ReviewAction.REQUEST_MORE_INFORMATION: "UNDER_REVIEW",
    ReviewAction.ESCALATE: "ESCALATED",
    ReviewAction.CLOSE: "CLOSED",
}


class InvalidCaseDecisionError(ValueError):
    """Raised when a case decision cannot be resolved."""


def resolve_case_decision(action: str) -> dict:
    """
    Resolve an investigator action into a workflow decision.

    This function does not update case records, persist reviews,
    or modify fraud labels.
    """
    try:
        normalized_action = normalize_review_action(action)
    except ValueError as exc:
        raise InvalidCaseDecisionError(str(exc)) from exc

    target_status = ACTION_STATUS_MAP.get(normalized_action)

    if target_status is None:
        raise InvalidCaseDecisionError(
            f"No workflow mapping exists for action: {normalized_action}"
        )

    return {
        "review_action": normalized_action,
        "target_status": target_status,
    }
