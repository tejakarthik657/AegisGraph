
from datetime import datetime, timezone
from typing import Any


class InvalidReviewActionError(ValueError):
    """Raised when a review action is invalid."""


class InvalidReviewRecordError(ValueError):
    """Raised when required review information is missing or invalid."""


class ReviewAction:
    MARK_AS_FRAUD = "MARK_AS_FRAUD"
    MARK_AS_LEGITIMATE = "MARK_AS_LEGITIMATE"
    REQUEST_MORE_INFORMATION = "REQUEST_MORE_INFORMATION"
    ESCALATE = "ESCALATE"
    CLOSE = "CLOSE"


ALLOWED_REVIEW_ACTIONS = {
    ReviewAction.MARK_AS_FRAUD,
    ReviewAction.MARK_AS_LEGITIMATE,
    ReviewAction.REQUEST_MORE_INFORMATION,
    ReviewAction.ESCALATE,
    ReviewAction.CLOSE,
}


def normalize_review_action(action: str) -> str:
    """Normalize and validate a supported investigator action."""
    if not isinstance(action, str) or not action.strip():
        raise InvalidReviewActionError(
            "Review action must be a non-empty string."
        )

    normalized = action.strip().upper()

    if normalized not in ALLOWED_REVIEW_ACTIONS:
        allowed = ", ".join(sorted(ALLOWED_REVIEW_ACTIONS))
        raise InvalidReviewActionError(
            f"Unsupported review action '{action}'. "
            f"Allowed actions: {allowed}"
        )

    return normalized


def build_review_record(
    case_id: str,
    action: str,
    reviewer: str,
    comments: str = "",
) -> dict[str, Any]:
    """
    Build a validated review record.

    This function does not persist the record or update a case.
    The reviewer value must come from a trusted caller; authentication
    will be integrated at the API layer.
    """
    if not isinstance(case_id, str) or not case_id.strip():
        raise InvalidReviewRecordError(
            "case_id must be a non-empty string."
        )

    if not isinstance(reviewer, str) or not reviewer.strip():
        raise InvalidReviewRecordError(
            "reviewer must be a non-empty string."
        )

    if comments is None:
        comments = ""

    if not isinstance(comments, str):
        raise InvalidReviewRecordError(
            "comments must be a string."
        )

    normalized_action = normalize_review_action(action)

    return {
        "case_id": case_id.strip(),
        "review_action": normalized_action,
        "reviewer": reviewer.strip(),
        "comments": comments.strip(),
        "reviewed_at": datetime.now(timezone.utc).isoformat(),
    }
