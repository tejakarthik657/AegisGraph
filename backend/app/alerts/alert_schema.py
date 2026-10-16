"""
AegisGraph — Alert Schema.

Pydantic models for alert data validation.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class AlertSeverity(str, Enum):
    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


class AlertStatus(str, Enum):
    OPEN = "OPEN"
    UNDER_REVIEW = "UNDER_REVIEW"
    ESCALATED = "ESCALATED"
    RESOLVED = "RESOLVED"
    DISMISSED = "DISMISSED"


class AlertRecord(BaseModel):
    """Validated alert record."""

    alert_id: str = Field(
        description="Unique alert identifier"
    )

    transaction_id: str = Field(
        description="Associated transaction ID"
    )

    timestamp: str = Field(
        description="Transaction timestamp"
    )

    sender_account: str = Field(
        description="Sender account ID"
    )

    receiver_account: str = Field(
        description="Receiver account ID"
    )

    amount: float = Field(
        ge=0.0,
        description="Transaction amount"
    )

    risk_score: float = Field(
        ge=0.0, le=1.0,
        description="Combined risk score"
    )

    risk_category: str = Field(
        description="Risk category label"
    )

    alert_status: AlertStatus = Field(
        default=AlertStatus.OPEN,
        description="Current alert status"
    )

    triggered_rules: Optional[str] = Field(
        default="",
        description="Pipe-separated list of triggered rules"
    )

    typology: Optional[str] = Field(
        default=None,
        description="Detected fraud typology"
    )

    is_fraud: Optional[int] = Field(
        default=None,
        description="Ground truth fraud label"
    )


class AlertUpdate(BaseModel):
    """Schema for updating an alert."""

    alert_status: Optional[AlertStatus] = None
    notes: Optional[str] = None
    assignee: Optional[str] = None
