"""
AegisGraph — Data Pipeline Tests.

Tests the data generation, validation, normalization,
and deduplication modules.
"""

import unittest
from unittest.mock import patch

import pandas as pd
import numpy as np


class TestDataValidation(unittest.TestCase):
    """Test data validation functions."""

    def test_validate_transactions_with_valid_data(self):
        from app.data.ingestion.validate import validate_transactions

        df = pd.DataFrame({
            "transaction_id": ["TX001", "TX002"],
            "sender_account": ["ACC001", "ACC002"],
            "receiver_account": ["ACC003", "ACC004"],
            "amount": [100.0, 200.0],
            "timestamp": ["2024-01-01", "2024-01-02"],
        })

        # Should not raise
        validate_transactions(df)


class TestDataNormalization(unittest.TestCase):
    """Test data normalization functions."""

    @patch("app.data.ingestion.normalize.create_directories")
    @patch("app.data.ingestion.normalize.CLEAN_TRANSACTIONS_FILE")
    def test_normalize_transactions(self, mock_file, mock_dirs):
        """Verify normalize handles all required columns."""
        from app.data.ingestion.normalize import normalize_transactions

        # Mock CSV write to avoid disk I/O
        mock_file.__str__ = lambda s: "/tmp/test.csv"

        df = pd.DataFrame({
            "transaction_id": ["TX001", "TX002"],
            "sender_account": ["ACC001", "ACC002"],
            "receiver_account": ["ACC003", "ACC004"],
            "amount": [100.0, 200.0],
            "timestamp": ["2024-01-01", "2024-01-02"],
            "currency": ["USD", "usd"],
            "transaction_type": ["wire", "ACH"],
            "typology": ["None", "smurfing"],
            "channel": ["online", "BRANCH"],
            "status": ["completed", "Completed"],
            "sender_country": ["US", "uk"],
            "receiver_country": ["UK", "us"],
            "is_fraud": [0, 1],
            "sender_balance_before": [10000.0, 20000.0],
            "sender_balance_after": [9900.0, 19800.0],
            "receiver_balance_before": [5000.0, 8000.0],
            "receiver_balance_after": [5100.0, 8200.0],
        })

        with patch.object(df, "to_csv"):
            result = normalize_transactions(df)

        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), 2)
        # Verify currency was uppercased
        self.assertTrue(
            all(c == "USD" for c in result["currency"])
        )


class TestDataDeduplication(unittest.TestCase):
    """Test data deduplication."""

    def test_deduplicate_removes_duplicates(self):
        from app.data.ingestion.dedupe import deduplicate_transactions

        df = pd.DataFrame({
            "transaction_id": ["TX001", "TX001", "TX002"],
            "sender_account": ["ACC001", "ACC001", "ACC002"],
            "receiver_account": ["ACC003", "ACC003", "ACC004"],
            "amount": [100.0, 100.0, 200.0],
            "timestamp": ["2024-01-01", "2024-01-01", "2024-01-02"],
        })

        result = deduplicate_transactions(df)
        self.assertLessEqual(len(result), 2)


class TestConfigPaths(unittest.TestCase):
    """Test configuration paths."""

    def test_project_root_exists(self):
        from app.utils.config import PROJECT_ROOT

        self.assertTrue(PROJECT_ROOT.exists())

    def test_data_dir_paths(self):
        from app.utils.config import (
            RAW_DATA_DIR,
            PROCESSED_DATA_DIR,
            RESULTS_DATA_DIR,
        )

        self.assertIn("raw", str(RAW_DATA_DIR))
        self.assertIn("processed", str(PROCESSED_DATA_DIR))
        self.assertIn("results", str(RESULTS_DATA_DIR))


if __name__ == "__main__":
    unittest.main()
