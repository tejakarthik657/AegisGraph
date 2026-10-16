"""
AegisGraph — API Integration Tests.

Tests the FastAPI endpoints for health, transactions,
alerts, cases, reviews, dossiers, graph, and risk.
"""

import unittest
from unittest.mock import patch, MagicMock
from pathlib import Path

import pandas as pd


class TestHealthEndpoints(unittest.TestCase):
    """Test health check endpoints."""

    def test_root_endpoint_structure(self):
        """Verify the root response structure."""
        from app.main import root

        response = root()
        self.assertEqual(response["service"], "AegisGraph API")
        self.assertEqual(response["status"], "running")
        self.assertIn("version", response)

    def test_health_endpoint_structure(self):
        """Verify the health response structure."""
        from app.main import health

        response = health()
        self.assertEqual(response["status"], "healthy")
        self.assertEqual(response["service"], "aegisgraph")


class TestHelperFunctions(unittest.TestCase):
    """Test utility functions in main.py."""

    def test_safe_float_with_valid_value(self):
        from app.main import safe_float

        self.assertEqual(safe_float(3.14), 3.14)
        self.assertEqual(safe_float("42.5"), 42.5)
        self.assertEqual(safe_float(0), 0.0)

    def test_safe_float_with_none(self):
        from app.main import safe_float

        self.assertEqual(safe_float(None), 0.0)
        self.assertEqual(safe_float(None, 99.0), 99.0)

    def test_safe_float_with_invalid(self):
        from app.main import safe_float

        self.assertEqual(safe_float("abc"), 0.0)

    def test_safe_int_with_valid_value(self):
        from app.main import safe_int

        self.assertEqual(safe_int(5), 5)
        self.assertEqual(safe_int(5.9), 5)
        self.assertEqual(safe_int("10"), 10)

    def test_safe_int_with_none(self):
        from app.main import safe_int

        self.assertEqual(safe_int(None), 0)

    def test_safe_string_with_valid_value(self):
        from app.main import safe_string

        self.assertEqual(safe_string("hello"), "hello")
        self.assertEqual(safe_string(42), "42")

    def test_safe_string_with_none(self):
        from app.main import safe_string

        self.assertIsNone(safe_string(None))
        self.assertEqual(safe_string(None, "default"), "default")

    def test_dataframe_to_records_handles_nan(self):
        from app.main import dataframe_to_records

        df = pd.DataFrame({
            "col_a": [1, None, 3],
            "col_b": ["x", "y", None],
        })

        records = dataframe_to_records(df)

        self.assertEqual(len(records), 3)
        self.assertIsNone(records[1]["col_a"])
        self.assertIsNone(records[2]["col_b"])

    def test_dataframe_to_records_handles_numpy_types(self):
        import numpy as np
        from app.main import dataframe_to_records

        df = pd.DataFrame({
            "int_col": pd.array([1, 2, 3], dtype="int64"),
            "float_col": pd.array([1.1, 2.2, 3.3], dtype="float64"),
        })

        records = dataframe_to_records(df)

        for record in records:
            for value in record.values():
                self.assertNotIsInstance(
                    value, (np.integer, np.floating)
                )


class TestEdgeFunctions(unittest.TestCase):
    """Test graph edge helper functions."""

    def test_edge_transaction_id(self):
        from app.main import edge_transaction_id

        data = {"transaction_id": "TX001"}
        self.assertEqual(edge_transaction_id(data), "TX001")

    def test_edge_transaction_id_missing(self):
        from app.main import edge_transaction_id

        self.assertIsNone(edge_transaction_id({}))

    def test_graph_edge_record(self):
        from app.main import graph_edge_record

        data = {
            "transaction_id": "TX001",
            "amount": 500.0,
            "timestamp": "2024-01-01",
            "currency": "USD",
            "transaction_type": "wire",
            "channel": "online",
            "sender_country": "US",
            "receiver_country": "UK",
            "status": "completed",
            "is_fraud": 0,
            "typology": "none",
        }

        record = graph_edge_record("ACC001", "ACC002", data)

        self.assertEqual(record["transaction_id"], "TX001")
        self.assertEqual(record["sender_account"], "ACC001")
        self.assertEqual(record["receiver_account"], "ACC002")
        self.assertEqual(record["amount"], 500.0)
        self.assertEqual(record["is_fraud"], 0)


if __name__ == "__main__":
    unittest.main()
