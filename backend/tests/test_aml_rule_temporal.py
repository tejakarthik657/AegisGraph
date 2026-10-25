import unittest

import pandas as pd

from app.rules.aml_rules import score_transactions_chronologically
from app.rules.evaluate_rules import run_rules_on_test_history


class TestAMLRuleTemporalScoring(unittest.TestCase):
    def _transactions(self):
        return pd.DataFrame(
            [
                {
                    "transaction_id": "T1",
                    "timestamp": "2025-01-01 10:00:00",
                    "sender_account": "A",
                    "receiver_account": "B",
                    "amount": 60_000.0,
                    "is_fraud": 0,
                    "typology": "none",
                },
                {
                    "transaction_id": "T2",
                    "timestamp": "2025-01-01 11:00:00",
                    "sender_account": "B",
                    "receiver_account": "C",
                    "amount": 10_000.0,
                    "is_fraud": 0,
                    "typology": "none",
                },
                {
                    "transaction_id": "T3",
                    "timestamp": "2025-01-01 12:00:00",
                    "sender_account": "C",
                    "receiver_account": "A",
                    "amount": 10_000.0,
                    "is_fraud": 1,
                    "typology": "circular",
                },
            ]
        )

    def test_adding_future_transactions_does_not_change_earlier_scores(self):
        transactions = self._transactions()
        original = score_transactions_chronologically(transactions)

        future = pd.DataFrame(
            [
                {
                    "transaction_id": "T4",
                    "timestamp": "2025-01-02 10:00:00",
                    "sender_account": "A",
                    "receiver_account": "D",
                    "amount": 60_000.0,
                },
                {
                    "transaction_id": "T5",
                    "timestamp": "2025-01-03 10:00:00",
                    "sender_account": "D",
                    "receiver_account": "E",
                    "amount": 60_000.0,
                },
            ]
        )
        extended = score_transactions_chronologically(
            pd.concat([transactions, future], ignore_index=True)
        )

        earlier_ids = transactions["transaction_id"]
        score_columns = [
            "large_transaction",
            "structuring",
            "rapid_movement",
            "fan_out",
            "circular_movement",
            "rule_prediction",
        ]
        pd.testing.assert_frame_equal(
            original.set_index("transaction_id")
            .loc[earlier_ids, score_columns],
            extended.set_index("transaction_id")
            .loc[earlier_ids, score_columns],
        )
        self.assertEqual(
            int(original.loc[original["transaction_id"] == "T3",
                             "circular_movement"].iloc[0]),
            1,
        )

    def test_same_timestamp_events_do_not_influence_each_other(self):
        transactions = pd.DataFrame(
            [
                {
                    "transaction_id": "PRIOR",
                    "timestamp": "2025-01-01 10:00:00",
                    "sender_account": "A",
                    "receiver_account": "B",
                    "amount": 10_000.0,
                },
                {
                    "transaction_id": "TIED_1",
                    "timestamp": "2025-01-01 11:00:00",
                    "sender_account": "A",
                    "receiver_account": "C",
                    "amount": 10_000.0,
                },
                {
                    "transaction_id": "TIED_2",
                    "timestamp": "2025-01-01 11:00:00",
                    "sender_account": "A",
                    "receiver_account": "D",
                    "amount": 10_000.0,
                },
            ]
        )

        scored = score_transactions_chronologically(transactions)
        tied = scored[scored["transaction_id"].str.startswith("TIED")]
        self.assertTrue((tied["structuring"] == 0).all())
        self.assertTrue((tied["fan_out"] == 0).all())

    def test_equal_timestamp_edges_do_not_form_a_causal_cycle(self):
        transactions = pd.DataFrame(
            [
                {
                    "transaction_id": "FIRST",
                    "timestamp": "2025-01-01 10:00:00",
                    "sender_account": "A",
                    "receiver_account": "B",
                    "amount": 60_000.0,
                },
                {
                    "transaction_id": "SECOND",
                    "timestamp": "2025-01-01 10:00:00",
                    "sender_account": "B",
                    "receiver_account": "C",
                    "amount": 10_000.0,
                },
                {
                    "transaction_id": "CLOSING",
                    "timestamp": "2025-01-01 11:00:00",
                    "sender_account": "C",
                    "receiver_account": "A",
                    "amount": 10_000.0,
                },
            ]
        )

        scored = score_transactions_chronologically(transactions)
        closing = scored.loc[
            scored["transaction_id"] == "CLOSING",
            "circular_movement",
        ].iloc[0]
        self.assertEqual(int(closing), 0)

    def test_test_scoring_can_use_only_prior_training_history(self):
        prior_history = pd.DataFrame(
            [
                {
                    "transaction_id": "TRAIN_1",
                    "timestamp": "2025-01-01 08:00:00",
                    "sender_account": "A",
                    "receiver_account": "B",
                    "amount": 10_000.0,
                },
                {
                    "transaction_id": "TRAIN_2",
                    "timestamp": "2025-01-01 09:00:00",
                    "sender_account": "A",
                    "receiver_account": "C",
                    "amount": 10_000.0,
                },
            ]
        )
        test_transactions = pd.DataFrame(
            [
                {
                    "transaction_id": "TEST_1",
                    "timestamp": "2025-01-01 10:00:00",
                    "sender_account": "A",
                    "receiver_account": "D",
                    "amount": 10_000.0,
                    "is_fraud": 0,
                    "typology": "none",
                }
            ]
        )

        matches = run_rules_on_test_history(
            test_transactions,
            prior_history=prior_history,
        )
        self.assertEqual(matches["structuring"], {"TEST_1"})

    def test_ground_truth_labels_do_not_change_rule_predictions(self):
        transactions = self._transactions()
        original = score_transactions_chronologically(transactions)

        relabeled = transactions.copy()
        relabeled["is_fraud"] = [1, 1, 0]
        relabeled["typology"] = ["fan_out", "structuring", "none"]
        changed_labels = score_transactions_chronologically(relabeled)

        prediction_columns = [
            "transaction_id",
            "large_transaction",
            "structuring",
            "rapid_movement",
            "fan_out",
            "circular_movement",
            "rule_prediction",
        ]
        pd.testing.assert_frame_equal(
            original[prediction_columns],
            changed_labels[prediction_columns],
        )

        matches = run_rules_on_test_history(transactions)
        relabeled_matches = run_rules_on_test_history(relabeled)
        self.assertEqual(matches, relabeled_matches)


if __name__ == "__main__":
    unittest.main()
