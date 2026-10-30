"""
AegisGraph — API Integration Tests.

Tests the FastAPI endpoints for health, transactions,
alerts, cases, reviews, dossiers, graph, and risk.
"""

import unittest
from unittest.mock import patch, MagicMock
from pathlib import Path
from tempfile import TemporaryDirectory

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


class TestModelMetricsEndpoint(unittest.TestCase):
    def test_endpoint_returns_saved_metrics_and_metadata(self):
        from fastapi.testclient import TestClient

        from app.main import app

        with TemporaryDirectory() as temporary_directory:
            results_dir = Path(temporary_directory)
            source_data = {
                "model_metrics.csv": {
                    "model": ["RandomForest_Baseline"],
                    "precision": [0.91],
                    "recall": [0.82],
                    "f1": [0.86],
                    "roc_auc": [0.95],
                    "pr_auc": [0.88],
                },
                "tgn_metrics.csv": {
                    "model": ["TGN"],
                    "train_transactions": [80],
                    "test_transactions": [20],
                    "test_accounts": [12],
                    "threshold": [0.5],
                    "precision": [0.7],
                    "recall": [0.6],
                    "f1": [0.64],
                    "roc_auc": [float("nan")],
                    "pr_auc": [0.68],
                },
                "thgnn_metrics.csv": {
                    "model": ["THGNN"],
                    "threshold": [0.4],
                    "train_transactions": [80],
                    "test_transactions": [20],
                    "test_frauds": [3],
                    "precision": [0.8],
                    "recall": [0.5],
                    "f1": [0.62],
                    "roc_auc": [0.9],
                    "pr_auc": [0.7],
                },
                "aml_rule_evaluation.csv": {
                    "model": ["AML Rules"],
                    "test_transactions": [20],
                    "test_fraud": [3],
                    "accuracy": [0.85],
                    "precision": [0.5],
                    "recall": [0.6],
                    "f1": [0.55],
                },
                "risk_fusion_metrics.csv": {
                    "threshold": [0.5],
                    "test_transactions": [20],
                    "precision": [0.9],
                    "recall": [0.8],
                    "f1": [0.84],
                    "roc_auc": [0.97],
                    "pr_auc": [0.9],
                },
                "aml_individual_rule_metrics.csv": {
                    "rule": ["large_transaction"],
                    "test_transactions": [20],
                    "test_fraud": [3],
                    "accuracy": [0.95],
                    "precision": [1.0],
                    "recall": [0.5],
                    "f1": [0.67],
                },
            }

            for filename, rows in source_data.items():
                pd.DataFrame(rows).to_csv(
                    results_dir / filename,
                    index=False,
                )

            metric_sources = (
                {
                    "key": "random_forest",
                    "name": "Random Forest",
                    "file": results_dir / "model_metrics.csv",
                },
                {
                    "key": "tgn",
                    "name": "TGN",
                    "file": results_dir / "tgn_metrics.csv",
                },
                {
                    "key": "thgnn",
                    "name": "THGNN",
                    "file": results_dir / "thgnn_metrics.csv",
                },
                {
                    "key": "aml_rules",
                    "name": "AML Rule Detection",
                    "file": results_dir / "aml_rule_evaluation.csv",
                },
                {
                    "key": "risk_fusion",
                    "name": "Risk Fusion",
                    "file": results_dir / "risk_fusion_metrics.csv",
                },
            )

            with (
                patch(
                    "app.main.MODEL_METRIC_SOURCES",
                    metric_sources,
                ),
                patch(
                    "app.main.AML_RULE_METRICS_FILE",
                    results_dir / "aml_individual_rule_metrics.csv",
                ),
            ):
                with TestClient(app) as client:
                    response = client.get("/api/model-metrics")
                    openapi = client.get("/openapi.json").json()

            self.assertEqual(response.status_code, 200)
            self.assertIn("/api/model-metrics", openapi["paths"])

            models = {
                model["key"]: model
                for model in response.json()["models"]
            }
            self.assertEqual(
                models["random_forest"]["metrics"]["precision"],
                0.91,
            )
            self.assertEqual(
                models["random_forest"]["model_name"],
                "Random Forest",
            )
            self.assertEqual(
                models["random_forest"]["source_model_name"],
                "RandomForest_Baseline",
            )
            self.assertIsNone(
                models["random_forest"]["metrics"]["accuracy"]
            )
            self.assertEqual(models["tgn"]["threshold"], 0.5)
            self.assertIsNone(models["tgn"]["metrics"]["roc_auc"])
            self.assertEqual(
                models["tgn"]["evaluation"]["test_accounts"],
                12,
            )
            self.assertEqual(
                models["thgnn"]["evaluation"]["test_frauds"],
                3,
            )
            self.assertEqual(
                models["aml_rules"]["metrics"]["accuracy"],
                0.85,
            )
            self.assertEqual(
                models["aml_rules"]["rules"][0]["rule_name"],
                "large_transaction",
            )
            self.assertEqual(
                models["risk_fusion"]["metrics"]["roc_auc"],
                0.97,
            )

    def test_missing_metric_files_return_unavailable_models(self):
        from fastapi.testclient import TestClient

        from app.main import app

        with TemporaryDirectory() as temporary_directory:
            results_dir = Path(temporary_directory)
            metric_sources = tuple(
                {
                    "key": key,
                    "name": name,
                    "file": results_dir / filename,
                }
                for key, name, filename in (
                    ("random_forest", "Random Forest", "rf.csv"),
                    ("tgn", "TGN", "tgn.csv"),
                    ("thgnn", "THGNN", "thgnn.csv"),
                    ("aml_rules", "AML Rule Detection", "aml.csv"),
                    ("risk_fusion", "Risk Fusion", "fusion.csv"),
                )
            )

            with (
                patch(
                    "app.main.MODEL_METRIC_SOURCES",
                    metric_sources,
                ),
                patch(
                    "app.main.AML_RULE_METRICS_FILE",
                    results_dir / "aml_details.csv",
                ),
            ):
                with TestClient(app) as client:
                    response = client.get("/api/model-metrics")

            self.assertEqual(response.status_code, 200)
            models = response.json()["models"]
            self.assertEqual(len(models), 5)
            for model in models:
                self.assertFalse(model["available"])
                self.assertTrue(
                    all(value is None for value in model["metrics"].values())
                )
                self.assertIsNone(model["threshold"])
                self.assertEqual(model["evaluation"], {})


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
