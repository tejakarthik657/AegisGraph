"""
AegisGraph — Risk Thresholds Configuration.

Configurable thresholds used across the detection layer
to determine when alerts should be generated.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class RiskThresholds:
    """Risk threshold configuration."""

    # Overall risk score thresholds
    critical_threshold: float = 0.75
    high_threshold: float = 0.50
    medium_threshold: float = 0.25
    low_threshold: float = 0.10

    # Alert generation thresholds
    alert_threshold: float = 0.50
    auto_escalation_threshold: float = 0.85

    # Rule-specific thresholds
    large_transaction_amount: float = 10000.0
    structuring_margin: float = 0.20
    velocity_count: int = 10
    velocity_window_hours: int = 24
    rapid_hold_minutes: int = 30

    # Pattern-specific thresholds
    smurfing_min_receivers: int = 3
    funnel_min_senders: int = 5
    layering_min_hops: int = 3
    cycle_max_length: int = 5

    def should_alert(self, risk_score: float) -> bool:
        """Whether a risk score warrants an alert."""
        return risk_score >= self.alert_threshold

    def should_auto_escalate(
        self, risk_score: float
    ) -> bool:
        """Whether a risk score warrants auto-escalation."""
        return risk_score >= self.auto_escalation_threshold

    def risk_category(self, risk_score: float) -> str:
        """Map a risk score to a category string."""

        if risk_score >= self.critical_threshold:
            return "CRITICAL"
        if risk_score >= self.high_threshold:
            return "HIGH"
        if risk_score >= self.medium_threshold:
            return "MEDIUM"
        if risk_score >= self.low_threshold:
            return "LOW"
        return "VERY_LOW"


# Default singleton instance
DEFAULT_THRESHOLDS = RiskThresholds()
