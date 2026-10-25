"""
AegisGraph — Baseline Model Tests.

Tests the baseline Random Forest model wrapper,
feature engineering, and prediction pipeline.
"""

import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import numpy as np


class TestBaselineModel(unittest.TestCase):
    """Test BaselineFraudModel class."""

    def _make_sample_data(self, n=100):
        np.random.seed(42)
        X = pd.DataFrame({
            "amount": np.random.uniform(10, 10000, n),
            "hour_of_day": np.random.randint(0, 24, n),
            "day_of_week": np.random.randint(0, 7, n),
            "sender_tx_count": np.random.randint(1, 50, n),
            "receiver_tx_count": np.random.randint(1, 50, n),
        })
        y = pd.Series(
            np.random.choice([0, 1], n, p=[0.9, 0.1])
        )
        return X, y

    def test_model_fit(self):
        from app.baseline_model.baseline_model import BaselineFraudModel

        model = BaselineFraudModel(
            n_estimators=10, max_depth=5
        )
        X, y = self._make_sample_data()
        model.fit(X, y)
        self.assertIsNotNone(model.model)

    def test_model_predict(self):
        from app.baseline_model.baseline_model import BaselineFraudModel

        model = BaselineFraudModel(
            n_estimators=10, max_depth=5
        )
        X, y = self._make_sample_data()
        model.fit(X, y)

        predictions = model.predict(X)
        self.assertEqual(len(predictions), len(X))
        self.assertTrue(
            all(p in [0, 1] for p in predictions)
        )

    def test_model_predict_proba(self):
        from app.baseline_model.baseline_model import BaselineFraudModel

        model = BaselineFraudModel(
            n_estimators=10, max_depth=5
        )
        X, y = self._make_sample_data()
        model.fit(X, y)

        proba = model.predict_proba(X)
        self.assertEqual(proba.shape[0], len(X))
        self.assertEqual(proba.shape[1], 2)

    def test_fraud_probability(self):
        from app.baseline_model.baseline_model import BaselineFraudModel

        model = BaselineFraudModel(
            n_estimators=10, max_depth=5
        )
        X, y = self._make_sample_data()
        model.fit(X, y)

        fraud_prob = model.fraud_probability(X)
        self.assertEqual(len(fraud_prob), len(X))
        self.assertTrue(
            all(0.0 <= p <= 1.0 for p in fraud_prob)
        )

    def test_feature_importance(self):
        from app.baseline_model.baseline_model import BaselineFraudModel

        model = BaselineFraudModel(
            n_estimators=10, max_depth=5
        )
        X, y = self._make_sample_data()
        model.fit(X, y)

        importance = model.feature_importance(
            list(X.columns)
        )
        self.assertEqual(len(importance), len(X.columns))
        self.assertIn("feature", importance.columns)
        self.assertIn("importance", importance.columns)


class TestBaselineFeatureSchema(unittest.TestCase):
    def _transactions(self, n=10):
        return pd.DataFrame({
            "transaction_id": [f"TX{i}" for i in range(n)],
            "timestamp": pd.date_range("2025-01-01", periods=n, freq="h"),
            "sender_account": ["A"] * n,
            "receiver_account": [f"R{i % 3}" for i in range(n)],
            "sender_country": ["US"] * n,
            "receiver_country": ["US"] * n,
            "status": ["COMPLETED"] * n,
            "amount": np.arange(n, dtype=float) + 10,
            "sender_balance_before": np.arange(n, dtype=float) + 100,
            "sender_balance_after": np.arange(n, dtype=float) + 90,
            "receiver_balance_before": np.arange(n, dtype=float) + 200,
            "receiver_balance_after": np.arange(n, dtype=float) + 210,
            "channel": ["ONLINE"] * n,
            "transaction_type": ["TRANSFER"] * n,
            "currency": ["USD"] * n,
            "is_fraud": [index % 2 for index in range(n)],
            "typology": ["none"] * n,
        })

    def test_rf_training_and_inference_share_excluded_feature_schema(self):
        from app.baseline_model.features.feature_engineering import (
            build_feature_dataset,
            build_features,
        )
        from app.baseline_model.predict import prepare_test_data
        from app.baseline_model.train_baseline import prepare_data

        transactions = self._transactions()
        engineered = build_feature_dataset(transactions)
        _, _, _, _, training_columns = prepare_data(engineered)

        forbidden = {
            "sender_balance_after",
            "receiver_balance_after",
            "is_fraud",
            "typology",
        }
        self.assertTrue(forbidden.isdisjoint(training_columns))

        inference_features = build_features(
            transactions,
            feature_columns=training_columns,
        )
        self.assertEqual(list(inference_features.columns), training_columns)
        self.assertTrue(forbidden.isdisjoint(inference_features.columns))

        with tempfile.TemporaryDirectory() as temp:
            transaction_file = Path(temp) / "transactions.csv"
            transactions.to_csv(transaction_file, index=False)
            with patch(
                "app.baseline_model.predict.CLEAN_TRANSACTIONS_FILE",
                transaction_file,
            ):
                test_features, test_transactions = prepare_test_data(
                    training_columns
                )

        self.assertEqual(list(test_features.columns), training_columns)
        self.assertEqual(len(test_features), 2)
        self.assertEqual(
            test_transactions["transaction_id"].tolist(),
            ["TX8", "TX9"],
        )

    def test_rf_schema_rejects_post_transaction_and_label_columns(self):
        from app.baseline_model.features.feature_engineering import build_features

        with self.assertRaisesRegex(ValueError, "excluded columns"):
            build_features(
                self._transactions(),
                ["amount", "sender_balance_after"],
            )

    def test_same_timestamp_transactions_do_not_change_history_features(self):
        from app.baseline_model.features.feature_engineering import (
            build_feature_dataset,
        )

        transactions = self._transactions(4)
        transactions["timestamp"] = pd.to_datetime([
            "2025-01-01 00:00:00",
            "2025-01-01 01:00:00",
            "2025-01-01 01:00:00",
            "2025-01-01 02:00:00",
        ])
        transactions["sender_account"] = ["A", "A", "B", "C"]
        transactions["receiver_account"] = ["R0", "R1", "A", "R2"]

        history_columns = [
            "sender_out_degree",
            "sender_total_outgoing",
            "sender_unique_neighbors",
            "receiver_in_degree",
            "receiver_total_incoming",
            "sender_activity_ratio",
            "receiver_activity_ratio",
            "sender_receiver_degree_ratio",
        ]
        baseline = build_feature_dataset(transactions).set_index(
            "transaction_id"
        )

        added_peer = transactions.iloc[[1]].copy()
        added_peer["transaction_id"] = "TX-peer"
        added_peer["sender_account"] = "A"
        added_peer["receiver_account"] = "R-new"
        added_peer["amount"] = 9999
        augmented = pd.concat(
            [transactions, added_peer],
            ignore_index=True,
        )
        augmented = pd.concat(
            [
                augmented[augmented["timestamp"] != pd.Timestamp("2025-01-01 01:00:00")],
                augmented[augmented["timestamp"] == pd.Timestamp("2025-01-01 01:00:00")].iloc[::-1],
            ],
            ignore_index=True,
        )
        with_peer = build_feature_dataset(augmented).set_index(
            "transaction_id"
        )

        for transaction_id in ["TX1", "TX2"]:
            pd.testing.assert_series_equal(
                baseline.loc[transaction_id, history_columns],
                with_peer.loc[transaction_id, history_columns],
                check_names=False,
            )
        self.assertEqual(baseline.loc["TX1", "sender_out_degree"], 1)
        self.assertEqual(baseline.loc["TX1", "sender_total_outgoing"], 10)
        self.assertEqual(baseline.loc["TX1", "sender_unique_neighbors"], 1)

    def test_future_transactions_do_not_change_earlier_history_features(self):
        from app.baseline_model.features.feature_engineering import (
            build_feature_dataset,
        )

        transactions = self._transactions(4)
        transactions["timestamp"] = pd.to_datetime([
            "2025-01-01 00:00:00",
            "2025-01-01 01:00:00",
            "2025-01-01 02:00:00",
            "2025-01-01 03:00:00",
        ])
        transactions["sender_account"] = ["A", "A", "A", "A"]
        transactions["receiver_account"] = ["R0", "R1", "R2", "R3"]
        history_columns = [
            "sender_out_degree",
            "sender_total_outgoing",
            "sender_unique_neighbors",
            "receiver_in_degree",
            "receiver_total_incoming",
            "sender_activity_ratio",
            "receiver_activity_ratio",
            "sender_receiver_degree_ratio",
        ]

        original = build_feature_dataset(
            transactions.iloc[:2]
        ).set_index("transaction_id")
        with_future = build_feature_dataset(
            transactions
        ).set_index("transaction_id")

        for transaction_id in ["TX0", "TX1"]:
            pd.testing.assert_series_equal(
                original.loc[transaction_id, history_columns],
                with_future.loc[transaction_id, history_columns],
                check_names=False,
            )

    def test_training_schema_excludes_categories_seen_only_after_split(self):
        from app.baseline_model.features.feature_engineering import (
            NON_MODEL_COLUMNS,
            build_feature_dataset,
        )
        from app.baseline_model.train_baseline import prepare_data

        transactions = self._transactions()
        transactions.loc[8:, "channel"] = "TEST_ONLY"
        engineered = build_feature_dataset(transactions)
        *_, feature_columns = prepare_data(engineered)

        self.assertNotIn("channel_TEST_ONLY", feature_columns)
        training_schema = [
            column
            for column in build_feature_dataset(
                transactions.iloc[:8]
            ).columns
            if column not in NON_MODEL_COLUMNS
        ]
        schema_aligned_features = build_feature_dataset(
            transactions,
            feature_columns=training_schema,
        )
        self.assertNotIn(
            "channel_TEST_ONLY",
            schema_aligned_features.columns,
        )

    def test_chronological_split_keeps_timestamp_batches_together(self):
        from app.baseline_model.features.feature_engineering import (
            chronological_split_index,
        )

        transactions = self._transactions(10)
        transactions["timestamp"] = pd.to_datetime([
            "2025-01-01 00:00:00",
            "2025-01-01 01:00:00",
            "2025-01-01 02:00:00",
            "2025-01-01 03:00:00",
            "2025-01-01 04:00:00",
            "2025-01-01 05:00:00",
            "2025-01-01 06:00:00",
            "2025-01-01 07:00:00",
            "2025-01-01 07:00:00",
            "2025-01-01 08:00:00",
        ])
        split_index = chronological_split_index(transactions)

        self.assertNotEqual(
            transactions.iloc[split_index - 1]["timestamp"],
            transactions.iloc[split_index]["timestamp"],
        )
        self.assertEqual(split_index, 9)


if __name__ == "__main__":
    unittest.main()
