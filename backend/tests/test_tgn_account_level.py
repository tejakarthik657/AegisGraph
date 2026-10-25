import unittest

import numpy as np
import pandas as pd

from app.ml_extensions.tgn.train_tgn import (
    FORBIDDEN_FEATURE_COLUMNS,
    build_node_features,
    build_node_labels,
    chronological_train_test_split,
    create_account_mapping,
    create_transaction_features,
    fit_transaction_preprocessor,
)


class TestAccountLevelTGNLeakage(unittest.TestCase):
    def _transactions(self):
        return pd.DataFrame(
            {
                "transaction_id": [f"TX{i}" for i in range(10)],
                "timestamp": pd.to_datetime(
                    [
                        "2025-01-01 08:00",
                        "2025-01-01 09:00",
                        "2025-01-01 10:00",
                        "2025-01-01 11:00",
                        "2025-01-01 12:00",
                        "2025-01-01 13:00",
                        "2025-01-01 14:00",
                        "2025-01-01 15:00",
                        "2025-01-01 16:00",
                        "2025-01-01 17:00",
                    ]
                ),
                "sender_account": ["A", "A", "B", "C", "D", "A", "B", "C", "D", "A"],
                "receiver_account": ["B", "C", "C", "D", "A", "D", "A", "B", "C", "B"],
                "amount": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0],
                "sender_balance_before": [100.0] * 10,
                "sender_balance_after": [90.0] * 10,
                "receiver_balance_before": [200.0] * 10,
                "receiver_balance_after": [210.0] * 10,
                "transaction_type": ["TRANSFER"] * 10,
                "channel": ["ONLINE"] * 10,
                "currency": ["USD"] * 10,
                "is_fraud": [0, 1, 0, 0, 0, 1, 0, 0, 1, 0],
                "typology": ["none", "fraud"] * 5,
            }
        )

    def test_chronological_split_is_80_20_without_timestamp_ties(self):
        transactions = self._transactions()
        train, test, prediction_timestamp = chronological_train_test_split(
            transactions
        )
        self.assertEqual(len(train), 8)
        self.assertEqual(len(test), 2)
        self.assertLess(train["timestamp"].max(), test["timestamp"].min())
        self.assertEqual(prediction_timestamp, test["timestamp"].min())

    def test_equal_timestamp_group_is_not_split_and_is_not_prior_history(self):
        transactions = self._transactions()
        tied_time = pd.Timestamp("2025-01-01 16:00")
        transactions.loc[8, "timestamp"] = tied_time
        transactions.loc[9, "timestamp"] = tied_time

        train, test, prediction_timestamp = chronological_train_test_split(
            transactions
        )
        self.assertTrue((train["timestamp"] != tied_time).all())
        self.assertTrue((test["timestamp"] == tied_time).all())
        self.assertEqual(prediction_timestamp, tied_time)

    def test_labels_and_post_transaction_fields_are_not_node_inputs(self):
        transactions = self._transactions()
        train = transactions.iloc[:8].copy()
        mapping = create_account_mapping(train)
        features = create_transaction_features(train)
        self.assertTrue(FORBIDDEN_FEATURE_COLUMNS.isdisjoint(features.columns))

        changed = train.copy()
        changed["is_fraud"] = 1 - changed["is_fraud"]
        changed["typology"] = "changed"
        changed["sender_balance_after"] = -999999.0
        changed["receiver_balance_after"] = -999999.0
        changed_features = create_transaction_features(changed)
        pd.testing.assert_frame_equal(features, changed_features)

        labels = build_node_labels(train, mapping)
        changed_labels = build_node_labels(changed, mapping)
        self.assertFalse(np.array_equal(labels, changed_labels))

    def test_future_transactions_do_not_change_pre_cutoff_account_features(self):
        transactions = self._transactions()
        train = transactions.iloc[:8].copy()
        prediction_timestamp = transactions.iloc[8]["timestamp"]
        mapping = create_account_mapping(train)
        scaler, _, scaled_train, _ = fit_transaction_preprocessor(
            train,
            transactions.iloc[8:],
        )
        del scaler
        baseline = build_node_features(
            scaled_train,
            train,
            mapping,
            prediction_timestamp=prediction_timestamp,
        )

        extended = transactions.copy()
        extended.loc[8:, "amount"] = [1e12, 2e12]
        extended_features = create_transaction_features(extended)
        training_raw = create_transaction_features(train)
        training_scaler, columns, _, _ = fit_transaction_preprocessor(
            train,
            extended.iloc[8:],
        )
        extended_scaled = pd.DataFrame(
            training_scaler.transform(
                extended_features.reindex(columns=columns, fill_value=0)
            ),
            columns=columns,
        )
        changed = build_node_features(
            extended_scaled,
            extended,
            mapping,
            prediction_timestamp=prediction_timestamp,
        )
        del training_raw
        np.testing.assert_allclose(baseline, changed)

    def test_preprocessing_is_fit_only_on_training_rows_and_schema(self):
        transactions = self._transactions()
        train = transactions.iloc[:8].copy()
        test = transactions.iloc[8:].copy()
        test["amount"] = [1e12, 2e12]
        test["transaction_type"] = ["TEST_ONLY", "TEST_ONLY"]

        scaler, columns, scaled_train, scaled_test = (
            fit_transaction_preprocessor(train, test)
        )
        raw_train = create_transaction_features(train)
        amount_index = columns.index("amount")
        self.assertAlmostEqual(
            scaler.mean_[amount_index],
            raw_train["amount"].mean(),
        )
        self.assertEqual(list(scaled_train.columns), columns)
        self.assertEqual(list(scaled_test.columns), columns)
        self.assertFalse(any("TEST_ONLY" in column for column in columns))
        self.assertTrue(FORBIDDEN_FEATURE_COLUMNS.isdisjoint(columns))


if __name__ == "__main__":
    unittest.main()
