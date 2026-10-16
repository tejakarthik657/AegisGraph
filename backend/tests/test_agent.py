"""
AegisGraph — Agent Module Tests.

Tests agent state, graph definition, guardrails, and tools.
"""

import unittest


class TestAgentState(unittest.TestCase):
    """Test agent state management."""

    def test_initial_state(self):
        from app.agent.state import AgentState

        state = AgentState(case_id="CASE001")
        self.assertEqual(state.case_id, "CASE001")
        self.assertEqual(state.current_step, "INIT")
        self.assertFalse(state.is_complete)
        self.assertEqual(len(state.steps_completed), 0)

    def test_mark_step_complete(self):
        from app.agent.state import AgentState

        state = AgentState()
        state.mark_step_complete("LOAD_ACCOUNT_INFO")

        self.assertIn("LOAD_ACCOUNT_INFO", state.steps_completed)
        self.assertEqual(state.current_step, "LOAD_ACCOUNT_INFO")

    def test_add_error(self):
        from app.agent.state import AgentState

        state = AgentState()
        state.add_error("Test error")
        self.assertEqual(len(state.errors), 1)

    def test_is_complete(self):
        from app.agent.state import AgentState

        state = AgentState()
        self.assertFalse(state.is_complete)
        state.current_step = "FINISHED"
        self.assertTrue(state.is_complete)

    def test_to_dict(self):
        from app.agent.state import AgentState

        state = AgentState(
            case_id="CASE001",
            alert_id="ALT001",
        )
        d = state.to_dict()
        self.assertEqual(d["case_id"], "CASE001")
        self.assertEqual(d["alert_id"], "ALT001")
        self.assertIn("started_at", d)


class TestGraphDefinition(unittest.TestCase):
    """Test agent workflow graph."""

    def test_get_next_steps(self):
        from app.agent.graph_definition import get_next_steps

        steps = get_next_steps("INIT")
        self.assertIn("LOAD_ACCOUNT_INFO", steps)

    def test_finished_has_no_next(self):
        from app.agent.graph_definition import get_next_steps

        steps = get_next_steps("FINISHED")
        self.assertEqual(len(steps), 0)

    def test_valid_transition(self):
        from app.agent.graph_definition import is_valid_transition

        self.assertTrue(
            is_valid_transition("INIT", "LOAD_ACCOUNT_INFO")
        )
        self.assertFalse(
            is_valid_transition("INIT", "FINISHED")
        )

    def test_default_path(self):
        from app.agent.graph_definition import get_default_path

        path = get_default_path()
        self.assertEqual(path[0], "INIT")
        self.assertEqual(path[-1], "FINISHED")


class TestGuardrails(unittest.TestCase):
    """Test agent safety guardrails."""

    def test_valid_action(self):
        from app.agent.guardrails import validate_action

        self.assertTrue(validate_action("ESCALATE"))

    def test_prohibited_action(self):
        from app.agent.guardrails import (
            validate_action,
            GuardrailViolation,
        )

        with self.assertRaises(GuardrailViolation):
            validate_action("DELETE_TRANSACTION")

    def test_query_scope_valid(self):
        from app.agent.guardrails import validate_query_scope

        self.assertTrue(
            validate_query_scope(num_transactions=100, num_hops=2)
        )

    def test_query_scope_too_large(self):
        from app.agent.guardrails import (
            validate_query_scope,
            GuardrailViolation,
        )

        with self.assertRaises(GuardrailViolation):
            validate_query_scope(num_transactions=10000, num_hops=2)

    def test_recommendation_confidence(self):
        from app.agent.guardrails import (
            validate_recommendation,
            GuardrailViolation,
        )

        self.assertTrue(
            validate_recommendation("ESCALATE", 0.80)
        )

        with self.assertRaises(GuardrailViolation):
            validate_recommendation("ESCALATE", 0.30)


class TestAgentTools(unittest.TestCase):
    """Test agent tool functions."""

    def test_finish_tool_high_risk(self):
        from app.agent.tools.finish_tool import finalize_investigation

        state = {
            "risk_assessment": {"risk_score": 0.85},
            "evidence_summary": "Test summary",
        }

        result = finalize_investigation(state)
        self.assertEqual(result["recommendation"], "ESCALATE")
        self.assertGreater(result["confidence"], 0.0)

    def test_finish_tool_low_risk(self):
        from app.agent.tools.finish_tool import finalize_investigation

        state = {
            "risk_assessment": {"risk_score": 0.10},
        }

        result = finalize_investigation(state)
        self.assertEqual(result["recommendation"], "CLOSE")

    def test_evidence_summary_tool(self):
        from app.agent.tools.evidence_summary_tool import build_evidence_summary

        state = {
            "account_info": {"ACC001": {}},
            "risk_assessment": {"risk_score": 0.75},
        }

        summary = build_evidence_summary(state)
        self.assertIn("Account analysis", summary)
        self.assertIn("Risk assessment", summary)

    def test_risk_tool(self):
        from app.agent.tools.risk_tool import assess_risk

        result = assess_risk(
            "TX001",
            {"risk_score": 0.85, "risk_category": "HIGH"},
        )
        self.assertEqual(result["transaction_id"], "TX001")
        self.assertEqual(result["risk_score"], 0.85)

    def test_pattern_tool(self):
        from app.agent.tools.pattern_tool import detect_patterns

        result = detect_patterns(["TX001", "TX002"])
        self.assertEqual(result["transaction_count"], 2)
        self.assertIn("smurfing", result["results"])

    def test_regulatory_rag_tool(self):
        from app.agent.tools.regulatory_rag_tool import query_regulations

        result = query_regulations("SAR filing requirements")
        self.assertIn("results", result)
        self.assertGreater(len(result["results"]), 0)


class TestRAGModules(unittest.TestCase):
    """Test RAG retriever and query generator."""

    def test_retriever(self):
        from app.agent.rag.retriever import retrieve

        results = retrieve("SAR suspicious transaction")
        self.assertIsInstance(results, list)
        self.assertGreater(len(results), 0)

    def test_query_generator(self):
        from app.agent.rag.query_generator import generate_query

        query = generate_query({
            "amount": 50000,
            "risk_category": "HIGH",
        })
        self.assertIn("SAR", query)


if __name__ == "__main__":
    unittest.main()
