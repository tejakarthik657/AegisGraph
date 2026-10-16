
from pathlib import Path
from threading import RLock
import os
import tempfile

import pandas as pd

from app.case_management.case_state_machine import (
    InvalidCaseTransitionError,
    normalize_status,
    validate_transition,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
RESULTS_DIR = PROJECT_ROOT / "data" / "results"
CASES_FILE = RESULTS_DIR / "investigation_cases.csv"

_CASES_LOCK = RLock()


class CaseNotFoundError(LookupError):
    """Raised when a requested case does not exist."""


class CaseServiceError(RuntimeError):
    """Raised when case records cannot be safely processed."""


def _read_cases() -> pd.DataFrame:
    """Load and validate the case records CSV."""
    if not CASES_FILE.is_file():
        raise CaseServiceError(
            f"Case records file was not found: {CASES_FILE}"
        )

    try:
        cases = pd.read_csv(
            CASES_FILE,
            dtype={"case_id": str, "case_status": str},
        )
    except Exception as exc:
        raise CaseServiceError(
            f"Could not read case records: {exc}"
        ) from exc

    required = {"case_id", "case_status"}
    missing = required - set(cases.columns)

    if missing:
        raise CaseServiceError(
            "Case records are missing required columns: "
            + ", ".join(sorted(missing))
        )

    if cases["case_id"].isna().any():
        raise CaseServiceError(
            "Case records contain an empty case_id."
        )

    cases["case_id"] = cases["case_id"].str.strip()

    if cases["case_id"].eq("").any():
        raise CaseServiceError(
            "Case records contain a blank case_id."
        )

    if cases["case_id"].duplicated().any():
        raise CaseServiceError(
            "Case records contain duplicate case IDs."
        )

    return cases


def _record_to_dict(record: pd.Series) -> dict:
    """Convert a pandas record to JSON-friendly Python values."""
    result = {}

    for key, value in record.items():
        if pd.isna(value):
            result[key] = None
        elif hasattr(value, "item"):
            result[key] = value.item()
        else:
            result[key] = value

    return result


def get_case(case_id: str) -> dict:
    """Return a single case record by its case ID."""
    normalized_id = str(case_id).strip()

    if not normalized_id:
        raise ValueError("case_id is required.")

    with _CASES_LOCK:
        cases = _read_cases()
        matching = cases.loc[cases["case_id"] == normalized_id]

        if matching.empty:
            raise CaseNotFoundError(
                f"Case not found: {normalized_id}"
            )

        return _record_to_dict(matching.iloc[0])


def _atomic_write_cases(cases: pd.DataFrame) -> None:
    """Write CSV data through a temporary file in the same directory."""
    temp_path = None

    try:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)

        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            suffix=".tmp",
            prefix="investigation_cases_",
            dir=RESULTS_DIR,
            delete=False,
        ) as temp_file:
            temp_path = Path(temp_file.name)
            cases.to_csv(temp_file, index=False)

        os.replace(temp_path, CASES_FILE)

    except Exception as exc:
        if temp_path is not None and temp_path.exists():
            try:
                temp_path.unlink()
            except OSError:
                pass

        raise CaseServiceError(
            f"Could not save case records: {exc}"
        ) from exc


def update_case_status(
    case_id: str,
    target_status: str,
) -> dict:
    """
    Validate and persist a case status transition.

    Other case records and columns are preserved.
    """
    normalized_id = str(case_id).strip()

    if not normalized_id:
        raise ValueError("case_id is required.")

    target = normalize_status(target_status)

    with _CASES_LOCK:
        cases = _read_cases()
        matching_indices = cases.index[
            cases["case_id"] == normalized_id
        ].tolist()

        if not matching_indices:
            raise CaseNotFoundError(
                f"Case not found: {normalized_id}"
            )

        index = matching_indices[0]
        current = normalize_status(cases.at[index, "case_status"])

        validate_transition(current, target)

        cases.at[index, "case_status"] = target.value
        _atomic_write_cases(cases)

        return _record_to_dict(cases.loc[index])
