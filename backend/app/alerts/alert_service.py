"""
AegisGraph — Alert Service.

Business logic for alert lifecycle management.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from app.utils.config import RESULTS_DATA_DIR


ALERTS_FILE = RESULTS_DATA_DIR / "alerts.csv"


class AlertNotFoundError(LookupError):
    """Raised when a requested alert does not exist."""


class AlertServiceError(RuntimeError):
    """Raised when an alert operation fails."""


def load_alerts() -> pd.DataFrame:
    """Load all alerts from CSV."""

    if not ALERTS_FILE.exists():
        raise AlertServiceError(
            f"Alerts file not found: {ALERTS_FILE}"
        )

    return pd.read_csv(ALERTS_FILE)


def get_alert(alert_id: str) -> dict:
    """Retrieve a single alert by its ID."""

    df = load_alerts()

    matches = df[
        df["alert_id"].astype(str) == str(alert_id)
    ]

    if matches.empty:
        raise AlertNotFoundError(
            f"Alert not found: {alert_id}"
        )

    record = matches.iloc[0]

    return {
        key: (
            None if pd.isna(value)
            else value.item() if hasattr(value, "item")
            else value
        )
        for key, value in record.items()
    }


def get_alerts_by_status(status: str) -> list[dict]:
    """Retrieve all alerts with the given status."""

    df = load_alerts()

    if "alert_status" not in df.columns:
        return []

    filtered = df[
        df["alert_status"].astype(str).str.upper()
        == status.strip().upper()
    ]

    return filtered.to_dict(orient="records")


def get_alert_statistics() -> dict:
    """Return aggregate alert statistics."""

    df = load_alerts()

    stats = {
        "total_alerts": len(df),
        "by_status": {},
        "by_risk_category": {},
        "average_risk_score": 0.0,
    }

    if "alert_status" in df.columns:
        stats["by_status"] = (
            df["alert_status"]
            .value_counts()
            .to_dict()
        )

    if "risk_category" in df.columns:
        stats["by_risk_category"] = (
            df["risk_category"]
            .value_counts()
            .to_dict()
        )

    if "risk_score" in df.columns:
        stats["average_risk_score"] = float(
            df["risk_score"].mean()
        )

    return stats
