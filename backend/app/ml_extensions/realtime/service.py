"""
AegisGraph — Realtime Scoring Service.

Provides low-latency fraud scoring for incoming transactions.
"""

from __future__ import annotations

import time
from typing import Any

from app.ml_extensions.realtime.schemas import (
    RealtimeTransaction,
    RealtimeScore,
)


class RealtimeService:
    """
    Real-time transaction scoring service.

    Uses cached model artifacts for fast inference.
    """

    def __init__(self):
        self._model_loaded = False
        self._rule_thresholds = {
            "large_transaction": 10000.0,
            "high_risk_countries": {"IR", "KP", "SY", "MM"},
        }

    def score_transaction(
        self,
        transaction: RealtimeTransaction,
    ) -> RealtimeScore:
        """Score a single transaction in real-time."""

        start = time.time()

        triggered_rules = []
        risk_signals = []

        # Rule: Large transaction
        if transaction.amount >= self._rule_thresholds["large_transaction"]:
            triggered_rules.append("large_transaction")
            risk_signals.append(0.3)

        # Rule: High-risk country
        hr_countries = self._rule_thresholds["high_risk_countries"]
        if (
            transaction.sender_country
            and transaction.sender_country.upper() in hr_countries
        ) or (
            transaction.receiver_country
            and transaction.receiver_country.upper() in hr_countries
        ):
            triggered_rules.append("high_risk_country")
            risk_signals.append(0.4)

        # Compute risk score
        fraud_probability = (
            sum(risk_signals) / max(len(risk_signals), 1)
            if risk_signals
            else 0.05
        )

        risk_score = min(fraud_probability, 1.0)

        if risk_score >= 0.75:
            risk_category = "CRITICAL"
        elif risk_score >= 0.50:
            risk_category = "HIGH"
        elif risk_score >= 0.25:
            risk_category = "MEDIUM"
        else:
            risk_category = "LOW"

        processing_time = (time.time() - start) * 1000

        return RealtimeScore(
            transaction_id=transaction.transaction_id,
            risk_score=risk_score,
            risk_category=risk_category,
            fraud_probability=fraud_probability,
            triggered_rules=triggered_rules,
            should_alert=risk_score >= 0.50,
            processing_time_ms=round(processing_time, 2),
        )
