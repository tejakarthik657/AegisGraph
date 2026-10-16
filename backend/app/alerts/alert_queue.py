"""
AegisGraph — Alert Queue.

Priority queue for processing alerts based on risk score
and severity. Supports FIFO ordering within same priority.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass, field
from typing import Any


@dataclass(order=True)
class PrioritizedAlert:
    """Alert wrapper with priority ordering."""

    priority: float = field(compare=True)
    sequence: int = field(compare=True)
    alert: dict = field(compare=False)


class AlertQueue:
    """
    Priority queue for processing alerts.

    Alerts with higher risk scores are processed first.
    Within the same priority, FIFO order is maintained.
    """

    def __init__(self):
        self._heap: list[PrioritizedAlert] = []
        self._sequence = 0

    def push(self, alert: dict) -> None:
        """Add an alert to the queue."""

        risk_score = float(
            alert.get("risk_score", 0.0)
        )

        # Negate for max-heap behavior
        priority = -risk_score

        entry = PrioritizedAlert(
            priority=priority,
            sequence=self._sequence,
            alert=alert,
        )

        heapq.heappush(self._heap, entry)
        self._sequence += 1

    def pop(self) -> dict | None:
        """
        Remove and return the highest-priority alert.
        Returns None if queue is empty.
        """

        if not self._heap:
            return None

        entry = heapq.heappop(self._heap)
        return entry.alert

    def peek(self) -> dict | None:
        """
        Return the highest-priority alert without removing it.
        """

        if not self._heap:
            return None

        return self._heap[0].alert

    @property
    def size(self) -> int:
        """Number of alerts in the queue."""
        return len(self._heap)

    @property
    def is_empty(self) -> bool:
        """Whether the queue is empty."""
        return len(self._heap) == 0

    def drain(self) -> list[dict]:
        """Remove and return all alerts in priority order."""

        alerts = []
        while not self.is_empty:
            alerts.append(self.pop())

        return alerts
