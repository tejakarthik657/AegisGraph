
import unittest

from app.case_management.case_state_machine import (
    CaseStatus,
    InvalidCaseTransitionError,
    get_allowed_transitions,
    validate_transition,
)
from app.case_management.review_actions import (
    InvalidReviewActionError,
    build_review_record,
)
from app.case_management.decisions import (
    InvalidCaseDecisionError,
    resolve_case_decision,
)


class TestCaseStateMachine(unittest.TestCase):
    def test_valid_transition(self):
        self.assertTrue(
            validate_transition("OPEN", "UNDER_REVIEW")
        )

    def test_closed_case_cannot_be_reopened(self):
        with self.assertRaises(InvalidCaseTransitionError):
            validate_transition("CLOSED", "OPEN")

    def test_same_status_is_rejected(self):
        with self.assertRaises(InvalidCaseTransitionError):
            validate_transition("OPEN", "OPEN")

    def test_unknown_status_is_rejected(self):
        with self.assertRaises(InvalidCaseTransitionError):
            validate_transition("OPEN", "INVALID")

    def test_closed_has_no_allowed_transitions(self):
        self.assertEqual(
            get_allowed_transitions(CaseStatus.CLOSED),
            [],
        )


class TestReviewActions(unittest.TestCase):
    def test_valid_review_record(self):
        record = build_review_record(
            case_id="CASE000001",
            action="REQUEST_MORE_INFORMATION",
            reviewer="analyst_test",
            comments="Need additional transaction evidence.",
        )

        self.assertEqual(record["case_id"], "CASE000001")
        self.assertEqual(
            record["review_action"],
            "REQUEST_MORE_INFORMATION",
        )
        self.assertEqual(record["reviewer"], "analyst_test")
        self.assertTrue(record["reviewed_at"])

    def test_missing_reviewer_is_rejected(self):
        with self.assertRaises(ValueError):
            build_review_record(
                case_id="CASE000001",
                action="CLOSE",
                reviewer="",
            )

    def test_invalid_action_is_rejected(self):
        with self.assertRaises(InvalidReviewActionError):
            build_review_record(
                case_id="CASE000001",
                action="UNKNOWN_ACTION",
                reviewer="analyst_test",
            )


class TestCaseDecisions(unittest.TestCase):
    def test_request_information_decision(self):
        decision = resolve_case_decision(
            "REQUEST_MORE_INFORMATION"
        )

        self.assertEqual(
            decision["target_status"],
            "UNDER_REVIEW",
        )

    def test_escalation_decision(self):
        decision = resolve_case_decision("ESCALATE")

        self.assertEqual(
            decision["target_status"],
            "ESCALATED",
        )

    def test_invalid_decision_is_rejected(self):
        with self.assertRaises(InvalidCaseDecisionError):
            resolve_case_decision("NOT_A_DECISION")


if __name__ == "__main__":
    unittest.main()
