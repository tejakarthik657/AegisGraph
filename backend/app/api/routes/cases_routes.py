
from datetime import datetime, timezone
from pathlib import Path
from threading import RLock
import os
import tempfile

import pandas as pd
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.case_management.case_service import (
    CaseNotFoundError,
    CaseServiceError,
    get_case as get_case_record,
)
from app.case_management.case_state_machine import (
    InvalidCaseTransitionError,
    validate_transition,
)
from app.case_management.review_actions import (
    InvalidReviewActionError,
    build_review_record,
)
from app.case_management.decisions import resolve_case_decision


router = APIRouter(
    prefix="/api/case-management",
    tags=["Case Management"],
)

PROJECT_ROOT = Path(__file__).resolve().parents[4]
RESULTS_DIR = PROJECT_ROOT / "data" / "results"

CASES_FILE = RESULTS_DIR / "investigation_cases.csv"
REVIEWS_FILE = RESULTS_DIR / "case_reviews.csv"
HISTORY_FILE = RESULTS_DIR / "case_review_history.csv"

_FILE_LOCK = RLock()

HISTORY_COLUMNS = [
    "review_id",
    "case_id",
    "review_action",
    "target_status",
    "reviewer",
    "comments",
    "reviewed_at",
]


class ReviewRequest(BaseModel):
    action: str = Field(min_length=1, max_length=64)
    reviewer: str = Field(min_length=1, max_length=128)
    comments: str = Field(default="", max_length=2000)


class StatusUpdateRequest(BaseModel):
    target_status: str = Field(min_length=1, max_length=32)


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise CaseServiceError(
            f"Required file was not found: {path.name}"
        )

    try:
        return pd.read_csv(path, dtype={"case_id": str})
    except Exception as exc:
        raise CaseServiceError(
            f"Could not read {path.name}: {exc}"
        ) from exc


def _atomic_write(path: Path, dataframe: pd.DataFrame) -> None:
    """Write one CSV using a temporary file and atomic replacement."""
    temp_path = None

    try:
        path.parent.mkdir(parents=True, exist_ok=True)

        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            prefix=f"{path.stem}_",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as temp_file:
            temp_path = Path(temp_file.name)
            dataframe.to_csv(temp_file, index=False)

        os.replace(temp_path, path)

    except Exception as exc:
        if temp_path is not None and temp_path.exists():
            try:
                temp_path.unlink()
            except OSError:
                pass

        raise CaseServiceError(
            f"Could not save {path.name}: {exc}"
        ) from exc


def _snapshot(path: Path) -> bytes | None:
    """Capture existing file bytes for rollback."""
    return path.read_bytes() if path.exists() else None


def _restore(path: Path, original: bytes | None) -> None:
    """Restore a previous file state after a failed multi-file write."""
    if original is None:
        if path.exists():
            path.unlink()
        return

    temp_path = None

    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f"{path.stem}_rollback_",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as temp_file:
            temp_path = Path(temp_file.name)
            temp_file.write(original)

        os.replace(temp_path, path)

    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink()


def _validate_unique_case_ids(dataframe: pd.DataFrame) -> None:
    if "case_id" not in dataframe.columns:
        raise CaseServiceError(
            "CSV is missing the required case_id column."
        )

    if dataframe["case_id"].isna().any():
        raise CaseServiceError("CSV contains an empty case_id.")

    dataframe["case_id"] = dataframe["case_id"].astype(str).str.strip()

    if dataframe["case_id"].eq("").any():
        raise CaseServiceError("CSV contains a blank case_id.")

    if dataframe["case_id"].duplicated().any():
        raise CaseServiceError("CSV contains duplicate case IDs.")


def _record_review(
    case_id: str,
    action: str,
    reviewer: str,
    comments: str,
    target_status: str,
) -> dict:
    """Update the latest review and append its history entry."""
    with _FILE_LOCK:
        cases = _read_csv(CASES_FILE)
        reviews = _read_csv(REVIEWS_FILE)

        _validate_unique_case_ids(cases)
        _validate_unique_case_ids(reviews)

        if "case_status" not in cases.columns:
            raise CaseServiceError(
                "Case CSV is missing case_status."
            )

        case_matches = cases["case_id"] == case_id
        review_matches = reviews["case_id"] == case_id

        if not case_matches.any():
            raise CaseNotFoundError(f"Case not found: {case_id}")

        if not review_matches.any():
            raise CaseServiceError(
                f"No initialized review record exists for {case_id}."
            )

        current_status = str(
            cases.loc[case_matches, "case_status"].iloc[0]
        )

        validate_transition(current_status, target_status)

        required_review_columns = [
            "review_status",
            "investigator_decision",
            "investigator_notes",
            "investigator_summary",
            "case_status",
        ]

        for column in required_review_columns:
            if column not in reviews.columns:
                raise CaseServiceError(
                    f"Review CSV is missing {column}."
                )

        # Convert columns explicitly so empty CSV values do not cause
        # pandas dtype errors when text is assigned.
        for column in required_review_columns:
            reviews[column] = reviews[column].astype(object)

        review = build_review_record(
            case_id=case_id,
            action=action,
            reviewer=reviewer,
            comments=comments,
        )

        if HISTORY_FILE.exists():
            history = _read_csv(HISTORY_FILE)
        else:
            history = pd.DataFrame(columns=HISTORY_COLUMNS)

        for column in HISTORY_COLUMNS:
            if column not in history.columns:
                history[column] = None

        review["target_status"] = target_status
        review["review_id"] = (
            "REV-"
            + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        )

        history_record = {
            column: review.get(column)
            for column in HISTORY_COLUMNS
        }

        decision_value = action.strip().upper()

        reviews.loc[review_matches, "case_status"] = target_status
        reviews.loc[review_matches, "review_status"] = "REVIEWED"
        reviews.loc[review_matches, "investigator_decision"] = decision_value

        if comments.strip():
            reviews.loc[review_matches, "investigator_notes"] = comments.strip()

        reviews.loc[review_matches, "investigator_summary"] = (
            f"Action: {decision_value}; "
            f"resulting status: {target_status}; "
            f"reviewer: {reviewer.strip()}"
        )

        cases["case_status"] = cases["case_status"].astype(object)
        cases.loc[case_matches, "case_status"] = target_status

        updated_history = pd.concat(
            [
                history,
                pd.DataFrame([history_record]),
            ],
            ignore_index=True,
        )

        # Capture originals before writing so failed writes can be rolled back.
        originals = {
            CASES_FILE: _snapshot(CASES_FILE),
            REVIEWS_FILE: _snapshot(REVIEWS_FILE),
            HISTORY_FILE: _snapshot(HISTORY_FILE),
        }

        try:
            _atomic_write(REVIEWS_FILE, reviews)
            _atomic_write(HISTORY_FILE, updated_history)
            _atomic_write(CASES_FILE, cases)

        except Exception as write_error:
            rollback_errors = []

            for path, original in originals.items():
                try:
                    _restore(path, original)
                except Exception as rollback_error:
                    rollback_errors.append(
                        f"{path.name}: {rollback_error}"
                    )

            if rollback_errors:
                raise CaseServiceError(
                    "A review write failed and rollback was incomplete: "
                    + "; ".join(rollback_errors)
                ) from write_error

            raise

        return {
            **review,
            "review_status": "REVIEWED",
        }


@router.get("/{case_id}")
def read_managed_case(case_id: str):
    try:
        return get_case_record(case_id)
    except CaseNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except CaseServiceError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.patch("/{case_id}/status")
def change_case_status(
    case_id: str,
    request: StatusUpdateRequest,
):
    """Update only the case lifecycle status."""
    from app.case_management.case_service import update_case_status

    try:
        updated_case = update_case_status(
            case_id,
            request.target_status,
        )

        return {
            "message": "Case status updated.",
            "case": updated_case,
        }

    except CaseNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except (InvalidCaseTransitionError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except CaseServiceError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/{case_id}/reviews")
def submit_case_review(
    case_id: str,
    request: ReviewRequest,
):
    """Record the latest review and append a review-history entry."""
    try:
        review = build_review_record(
            case_id=case_id,
            action=request.action,
            reviewer=request.reviewer,
            comments=request.comments,
        )

        decision = resolve_case_decision(request.action)

        recorded_review = _record_review(
            case_id=review["case_id"],
            action=decision["review_action"],
            reviewer=review["reviewer"],
            comments=review["comments"],
            target_status=decision["target_status"],
        )

        return {
            "message": "Review submitted.",
            "review": recorded_review,
            "case": get_case_record(case_id),
        }

    except CaseNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except (
        InvalidReviewActionError,
        InvalidCaseTransitionError,
        ValueError,
    ) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except CaseServiceError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
