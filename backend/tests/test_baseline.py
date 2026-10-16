"""
AegisGraph — Baseline Model Tests.

Tests the baseline Random Forest model wrapper,
feature engineering, and prediction pipeline.
"""

import unittest

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


if __name__ == "__main__":
    unittest.main()
