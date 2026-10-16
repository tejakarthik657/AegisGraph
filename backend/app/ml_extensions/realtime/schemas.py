"""
AegisGraph — Realtime Schemas.

Pydantic models for real-time transaction scoring.
"""

from __future__ import annotations

from pydantic import BaseModel, Field
from typing import Optional


class RealtimeTransaction(BaseModel):
    """Incoming transaction for real-time scoring."""

    transaction_id: str
    sender_account: str
    receiver_account: str
    amount: float = Field(ge=0.0)
    currency: str = "USD"
    timestamp: str
    transaction_type: Optional[str] = None
    channel: Optional[str] = None
    sender_country: Optional[str] = None
    receiver_country: Optional[str] = None


class RealtimeScore(BaseModel):
    """Real-time scoring result."""

    transaction_id: str
    risk_score: float = Field(ge=0.0, le=1.0)
    risk_category: str
    fraud_probability: float = Field(ge=0.0, le=1.0)
    triggered_rules: list[str] = []
    should_alert: bool = False
    processing_time_ms: float = 0.0
