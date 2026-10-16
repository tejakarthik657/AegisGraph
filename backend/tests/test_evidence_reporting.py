"""
AegisGraph — Evidence Reporting Tests.

Tests evidence chain, report generator, and narrative.
"""

import unittest


class TestEvidenceChain(unittest.TestCase):
    """Test evidence chain builder."""

    def test_create_chain(self):
        from app.evidence_reporting.evidence_chain import EvidenceChain

        chain = EvidenceChain(case_id="CASE001")
        self.assertEqual(chain.case_id, "CASE001")
        self.assertEqual(len(chain.items), 0)

    def test_add_transaction_evidence(self):
        from app.evidence_reporting.evidence_chain import EvidenceChain

        chain = EvidenceChain(case_id="CASE001")
        chain.add_transaction_evidence({
            "transaction_id": "TX001",
            "amount": 5000.0,
            "sender_account": "ACC001",
            "receiver_account": "ACC002",
            "timestamp": "2024-01-01",
        })

        self.assertEqual(len(chain.items), 1)
        self.assertEqual(
            chain.items[0].evidence_type, "TRANSACTION"
        )

    def test_add_rule_evidence(self):
        from app.evidence_reporting.evidence_chain import EvidenceChain

        chain = EvidenceChain(case_id="CASE001")
        chain.add_rule_evidence("large_transaction", True)

        self.assertEqual(len(chain.items), 1)
        self.assertEqual(
            chain.items[0].evidence_type, "RULE_TRIGGER"
        )

    def test_rule_not_triggered_not_added(self):
        from app.evidence_reporting.evidence_chain import EvidenceChain

        chain = EvidenceChain(case_id="CASE001")
        chain.add_rule_evidence("large_transaction", False)

        self.assertEqual(len(chain.items), 0)

    def test_add_pattern_evidence(self):
        from app.evidence_reporting.evidence_chain import EvidenceChain

        chain = EvidenceChain(case_id="CASE001")
        chain.add_pattern_evidence("smurfing")

        self.assertEqual(len(chain.items), 1)
        self.assertEqual(
            chain.items[0].evidence_type, "PATTERN_MATCH"
        )

    def test_add_risk_score_evidence(self):
        from app.evidence_reporting.evidence_chain import EvidenceChain

        chain = EvidenceChain(case_id="CASE001")
        chain.add_risk_score_evidence(0.85, "CRITICAL")

        self.assertEqual(len(chain.items), 1)

    def test_to_dict(self):
        from app.evidence_reporting.evidence_chain import EvidenceChain

        chain = EvidenceChain(case_id="CASE001")
        chain.add_transaction_evidence({
            "transaction_id": "TX001",
            "amount": 1000.0,
            "sender_account": "ACC001",
            "receiver_account": "ACC002",
        })

        d = chain.to_dict()
        self.assertEqual(d["case_id"], "CASE001")
        self.assertEqual(d["total_evidence_items"], 1)
        self.assertEqual(len(d["evidence"]), 1)

    def test_summary(self):
        from app.evidence_reporting.evidence_chain import EvidenceChain

        chain = EvidenceChain(case_id="CASE001")
        chain.add_transaction_evidence({
            "transaction_id": "TX001",
            "amount": 500.0,
            "sender_account": "A",
            "receiver_account": "B",
        })
        chain.add_rule_evidence("structuring", True)

        summary = chain.summary
        self.assertIn("CASE001", summary)
        self.assertIn("transaction", summary)


class TestReportGenerator(unittest.TestCase):
    """Test report generation."""

    def test_generate_report(self):
        from app.evidence_reporting.report_generator import generate_report

        report = generate_report({
            "case_id": "CASE001",
            "case_status": "OPEN",
            "case_priority": "HIGH",
            "risk_score": 0.85,
            "risk_category": "CRITICAL",
            "amount": 50000.0,
        })

        self.assertIn("report_id", report)
        self.assertEqual(report["case_id"], "CASE001")
        self.assertIn("narrative", report)
        self.assertIn("recommendations", report)
        self.assertIsInstance(report["recommendations"], list)

    def test_format_report_text(self):
        from app.evidence_reporting.report_generator import (
            generate_report,
            format_report_text,
        )

        report = generate_report({
            "case_id": "CASE002",
            "case_status": "OPEN",
            "risk_score": 0.60,
            "risk_category": "HIGH",
            "amount": 25000.0,
        })

        text = format_report_text(report)
        self.assertIn("CASE002", text)
        self.assertIn("INVESTIGATION REPORT", text)


class TestGenAINarrative(unittest.TestCase):
    """Test AI narrative generation."""

    def test_generate_narrative(self):
        from app.evidence_reporting.genai_narrative import generate_narrative

        narrative = generate_narrative({
            "case_id": "CASE001",
            "alert_id": "ALT001",
            "transaction_id": "TX001",
            "sender_account": "ACC001",
            "receiver_account": "ACC002",
            "amount": 50000.0,
            "risk_score": 0.85,
            "risk_category": "CRITICAL",
            "typology": "smurfing",
        })

        self.assertIn("CASE001", narrative)
        self.assertIn("CRITICAL", narrative)
        self.assertIn("smurfing", narrative)

    def test_narrative_with_evidence(self):
        from app.evidence_reporting.genai_narrative import generate_narrative

        evidence = {
            "evidence": [
                {"evidence_type": "TRANSACTION", "source": "data"},
                {"evidence_type": "RULE_TRIGGER", "source": "rules"},
            ]
        }

        narrative = generate_narrative(
            {"case_id": "CASE002", "risk_score": 0.50},
            evidence_chain=evidence,
        )

        self.assertIn("2 pieces", narrative)


class TestAlertQueue(unittest.TestCase):
    """Test alert priority queue."""

    def test_push_and_pop(self):
        from app.alerts.alert_queue import AlertQueue

        queue = AlertQueue()
        queue.push({"alert_id": "A1", "risk_score": 0.5})
        queue.push({"alert_id": "A2", "risk_score": 0.9})

        # Highest risk first
        top = queue.pop()
        self.assertEqual(top["alert_id"], "A2")

    def test_empty_queue(self):
        from app.alerts.alert_queue import AlertQueue

        queue = AlertQueue()
        self.assertTrue(queue.is_empty)
        self.assertIsNone(queue.pop())
        self.assertIsNone(queue.peek())

    def test_drain(self):
        from app.alerts.alert_queue import AlertQueue

        queue = AlertQueue()
        queue.push({"alert_id": "A1", "risk_score": 0.3})
        queue.push({"alert_id": "A2", "risk_score": 0.7})
        queue.push({"alert_id": "A3", "risk_score": 0.5})

        alerts = queue.drain()
        self.assertEqual(len(alerts), 3)
        self.assertTrue(queue.is_empty)

        # Should be in descending risk order
        scores = [a["risk_score"] for a in alerts]
        self.assertEqual(scores, sorted(scores, reverse=True))


if __name__ == "__main__":
    unittest.main()
