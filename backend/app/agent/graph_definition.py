"""
AegisGraph — Agent Graph Definition.

Defines the investigation workflow as a directed graph
of processing steps. Each node represents an investigation
tool or analysis step.
"""

from __future__ import annotations

from typing import Any


# Define the investigation workflow steps
INVESTIGATION_STEPS = [
    "INIT",
    "LOAD_ACCOUNT_INFO",
    "LOAD_TRANSACTION_DETAILS",
    "ANALYZE_GRAPH",
    "ANALYZE_VELOCITY",
    "DETECT_PATTERNS",
    "ASSESS_RISK",
    "CHECK_REGULATIONS",
    "BUILD_EVIDENCE",
    "GENERATE_RECOMMENDATION",
    "FINISHED",
]


# Adjacency: step -> allowed next steps
WORKFLOW_GRAPH = {
    "INIT": ["LOAD_ACCOUNT_INFO"],
    "LOAD_ACCOUNT_INFO": ["LOAD_TRANSACTION_DETAILS"],
    "LOAD_TRANSACTION_DETAILS": [
        "ANALYZE_GRAPH",
        "ANALYZE_VELOCITY",
    ],
    "ANALYZE_GRAPH": [
        "DETECT_PATTERNS",
        "ANALYZE_VELOCITY",
    ],
    "ANALYZE_VELOCITY": [
        "DETECT_PATTERNS",
        "ASSESS_RISK",
    ],
    "DETECT_PATTERNS": ["ASSESS_RISK"],
    "ASSESS_RISK": [
        "CHECK_REGULATIONS",
        "BUILD_EVIDENCE",
    ],
    "CHECK_REGULATIONS": ["BUILD_EVIDENCE"],
    "BUILD_EVIDENCE": ["GENERATE_RECOMMENDATION"],
    "GENERATE_RECOMMENDATION": ["FINISHED"],
    "FINISHED": [],
}


def get_next_steps(current_step: str) -> list[str]:
    """Return allowed next steps from the current step."""
    return WORKFLOW_GRAPH.get(current_step, [])


def is_valid_transition(
    current_step: str,
    next_step: str,
) -> bool:
    """Check if a step transition is valid."""
    allowed = get_next_steps(current_step)
    return next_step in allowed


def get_default_path() -> list[str]:
    """Return the default linear investigation path."""
    return list(INVESTIGATION_STEPS)
