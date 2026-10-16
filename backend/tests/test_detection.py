"""
AegisGraph — Detection Module Tests.

Tests rules, patterns, graph scoring, and fusion.
"""

import unittest

import pandas as pd
import numpy as np
import networkx as nx


class TestVelocityBreak(unittest.TestCase):
    """Test velocity break detection."""

    def _make_transactions(self):
        timestamps = pd.date_range(
            "2024-01-01", periods=15, freq="1h"
        )
        return pd.DataFrame({
            "transaction_id": [f"TX{i:03d}" for i in range(15)],
            "sender_account": ["ACC001"] * 15,
            "receiver_account": [f"ACC{i:03d}" for i in range(100, 115)],
            "amount": [500.0] * 15,
            "timestamp": timestamps,
        })

    def test_detect_velocity_break(self):
        from app.detection.rules.velocity_break import detect_velocity_break

        df = self._make_transactions()
        result = detect_velocity_break(
            df, window_hours=24, velocity_threshold=5
        )

        self.assertIn("velocity_break", result.columns)
        self.assertIn("sender_velocity", result.columns)
        self.assertTrue(result["velocity_break"].sum() > 0)


class TestStructuring(unittest.TestCase):
    """Test structuring detection."""

    def test_detect_structuring(self):
        from app.detection.rules.structuring import detect_structuring

        timestamps = pd.date_range(
            "2024-01-01", periods=5, freq="2h"
        )
        df = pd.DataFrame({
            "transaction_id": [f"TX{i}" for i in range(5)],
            "sender_account": ["ACC001"] * 5,
            "receiver_account": ["ACC002"] * 5,
            "amount": [9500.0, 9200.0, 9800.0, 9100.0, 9300.0],
            "timestamp": timestamps,
        })

        result = detect_structuring(
            df, threshold=10000.0, margin_pct=0.15, min_count=3
        )
        self.assertIn("structuring", result.columns)


class TestHighRiskCountry(unittest.TestCase):
    """Test high-risk country detection."""

    def test_detect_high_risk_sender(self):
        from app.detection.rules.high_risk_country import detect_high_risk_country

        df = pd.DataFrame({
            "sender_country": ["US", "IR", "UK"],
            "receiver_country": ["UK", "US", "US"],
        })

        result = detect_high_risk_country(df)
        self.assertIn("high_risk_country", result.columns)
        self.assertEqual(result["high_risk_country"].iloc[1], 1)

    def test_normal_countries_not_flagged(self):
        from app.detection.rules.high_risk_country import detect_high_risk_country

        df = pd.DataFrame({
            "sender_country": ["US", "UK", "DE"],
            "receiver_country": ["UK", "FR", "US"],
        })

        result = detect_high_risk_country(df)
        self.assertEqual(result["high_risk_country"].sum(), 0)


class TestSmurfingPattern(unittest.TestCase):
    """Test smurfing pattern detection."""

    def test_detect_smurfing(self):
        from app.detection.patterns.smurfing import detect_smurfing

        timestamps = pd.date_range(
            "2024-01-01", periods=5, freq="1h"
        )
        df = pd.DataFrame({
            "transaction_id": [f"TX{i}" for i in range(5)],
            "sender_account": ["ACC001"] * 5,
            "receiver_account": [
                "ACC100", "ACC101", "ACC102", "ACC103", "ACC104"
            ],
            "amount": [3000.0] * 5,
            "timestamp": timestamps,
        })

        result = detect_smurfing(
            df, threshold=10000.0, min_receivers=3
        )
        self.assertIn("smurfing", result.columns)


class TestCycleDetection(unittest.TestCase):
    """Test cycle detection."""

    def test_detect_cycles(self):
        from app.detection.patterns.cycle import detect_cycles

        df = pd.DataFrame({
            "transaction_id": ["TX1", "TX2", "TX3"],
            "sender_account": ["A", "B", "C"],
            "receiver_account": ["B", "C", "A"],
            "amount": [100, 200, 300],
        })

        result = detect_cycles(df, max_cycle_length=5)
        self.assertIn("cycle_member", result.columns)
        self.assertTrue(result["cycle_member"].sum() > 0)


class TestNeighborhoodRisk(unittest.TestCase):
    """Test neighborhood risk scoring."""

    def test_compute_neighborhood_risk(self):
        from app.detection.graph_scoring.neighborhood_risk import (
            compute_neighborhood_risk,
        )

        graph = nx.DiGraph()
        graph.add_edge("A", "B", transaction_id="TX1")
        graph.add_edge("B", "C", transaction_id="TX2")
        graph.add_edge("C", "D", transaction_id="TX3")

        risk_df = pd.DataFrame({
            "transaction_id": ["TX1", "TX2", "TX3"],
            "risk_score": [0.9, 0.3, 0.1],
        })

        result = compute_neighborhood_risk(graph, risk_df, hops=2)
        self.assertIn("account_id", result.columns)
        self.assertIn("neighborhood_risk", result.columns)
        self.assertEqual(len(result), 4)


class TestRiskFusion(unittest.TestCase):
    """Test detection risk fusion."""

    def test_fuse_risk_scores(self):
        from app.detection.fusion.risk_fusion import fuse_risk_scores

        df = pd.DataFrame({
            "ml_score": [0.8, 0.2, 0.5],
            "rule_score": [0.6, 0.0, 0.3],
            "pattern_score": [0.4, 0.0, 0.1],
            "graph_score": [0.7, 0.1, 0.2],
        })

        result = fuse_risk_scores(df)
        self.assertIn("fused_risk_score", result.columns)
        self.assertIn("fused_risk_category", result.columns)
        self.assertTrue(
            all(0.0 <= s <= 1.0 for s in result["fused_risk_score"])
        )


class TestThresholds(unittest.TestCase):
    """Test risk threshold configuration."""

    def test_default_thresholds(self):
        from app.detection.fusion.thresholds import DEFAULT_THRESHOLDS

        self.assertTrue(DEFAULT_THRESHOLDS.should_alert(0.75))
        self.assertFalse(DEFAULT_THRESHOLDS.should_alert(0.25))
        self.assertTrue(DEFAULT_THRESHOLDS.should_auto_escalate(0.90))
        self.assertEqual(
            DEFAULT_THRESHOLDS.risk_category(0.80), "CRITICAL"
        )
        self.assertEqual(
            DEFAULT_THRESHOLDS.risk_category(0.10), "LOW"
        )
        self.assertEqual(
            DEFAULT_THRESHOLDS.risk_category(0.05), "VERY_LOW"
        )


if __name__ == "__main__":
    unittest.main()
