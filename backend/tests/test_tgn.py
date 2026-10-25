"""
AegisGraph — TGN Module Tests.

Tests the Temporal Graph Network components:
memory, message, time encoding, and model architecture.
"""

import unittest

import numpy as np
import pandas as pd
import torch


class TestTGNMemory(unittest.TestCase):
    """Test TGN memory module."""

    def test_memory_manager_init(self):
        from app.ml_extensions.realtime.memory_manager import MemoryManager

        manager = MemoryManager(memory_dim=32, max_nodes=100)
        self.assertEqual(manager.memory_dim, 32)
        self.assertEqual(manager.max_nodes, 100)
        self.assertEqual(manager.node_count, 0)

    def test_memory_get_unseen_node(self):
        from app.ml_extensions.realtime.memory_manager import MemoryManager

        manager = MemoryManager(memory_dim=16)
        memory = manager.get_memory("unseen_node")

        self.assertEqual(len(memory), 16)
        self.assertTrue(np.all(memory == 0.0))

    def test_memory_update_and_retrieve(self):
        from app.ml_extensions.realtime.memory_manager import MemoryManager

        manager = MemoryManager(memory_dim=8)
        new_memory = np.ones(8, dtype=np.float32)

        manager.update_memory("node_1", new_memory)
        retrieved = manager.get_memory("node_1")

        np.testing.assert_array_equal(retrieved, new_memory)
        self.assertEqual(manager.node_count, 1)

    def test_memory_eviction(self):
        from app.ml_extensions.realtime.memory_manager import MemoryManager

        manager = MemoryManager(memory_dim=4, max_nodes=3)

        for i in range(5):
            manager.update_memory(
                f"node_{i}",
                np.ones(4) * i,
            )

        # Only last 3 should remain
        self.assertEqual(manager.node_count, 3)
        self.assertFalse(manager.has_memory("node_0"))
        self.assertFalse(manager.has_memory("node_1"))
        self.assertTrue(manager.has_memory("node_4"))

    def test_memory_reset(self):
        from app.ml_extensions.realtime.memory_manager import MemoryManager

        manager = MemoryManager(memory_dim=4)
        manager.update_memory("node_1", np.ones(4))
        manager.reset()

        self.assertEqual(manager.node_count, 0)


class TestRealtimeSchemas(unittest.TestCase):
    """Test realtime Pydantic schemas."""

    def test_realtime_transaction(self):
        from app.ml_extensions.realtime.schemas import RealtimeTransaction

        txn = RealtimeTransaction(
            transaction_id="TX001",
            sender_account="ACC001",
            receiver_account="ACC002",
            amount=500.0,
            timestamp="2024-01-01T10:00:00",
        )

        self.assertEqual(txn.transaction_id, "TX001")
        self.assertEqual(txn.amount, 500.0)

    def test_realtime_score(self):
        from app.ml_extensions.realtime.schemas import RealtimeScore

        score = RealtimeScore(
            transaction_id="TX001",
            risk_score=0.85,
            risk_category="CRITICAL",
            fraud_probability=0.90,
        )

        self.assertEqual(score.risk_category, "CRITICAL")
        self.assertTrue(score.risk_score >= 0.0)


class TestRealtimeService(unittest.TestCase):
    """Test the realtime scoring service."""

    def test_score_normal_transaction(self):
        from app.ml_extensions.realtime.service import RealtimeService
        from app.ml_extensions.realtime.schemas import RealtimeTransaction

        service = RealtimeService()

        txn = RealtimeTransaction(
            transaction_id="TX001",
            sender_account="ACC001",
            receiver_account="ACC002",
            amount=100.0,
            timestamp="2024-01-01T10:00:00",
        )

        result = service.score_transaction(txn)
        self.assertEqual(result.risk_category, "LOW")
        self.assertFalse(result.should_alert)

    def test_score_large_transaction(self):
        from app.ml_extensions.realtime.service import RealtimeService
        from app.ml_extensions.realtime.schemas import RealtimeTransaction

        service = RealtimeService()

        txn = RealtimeTransaction(
            transaction_id="TX002",
            sender_account="ACC001",
            receiver_account="ACC002",
            amount=50000.0,
            timestamp="2024-01-01T10:00:00",
        )

        result = service.score_transaction(txn)
        self.assertIn("large_transaction", result.triggered_rules)

    def test_score_high_risk_country(self):
        from app.ml_extensions.realtime.service import RealtimeService
        from app.ml_extensions.realtime.schemas import RealtimeTransaction

        service = RealtimeService()

        txn = RealtimeTransaction(
            transaction_id="TX003",
            sender_account="ACC001",
            receiver_account="ACC002",
            amount=5000.0,
            timestamp="2024-01-01T10:00:00",
            sender_country="IR",
        )

        result = service.score_transaction(txn)
        self.assertIn("high_risk_country", result.triggered_rules)


class TestTGNFeatureSchema(unittest.TestCase):
    def _transactions(self):
        return pd.DataFrame({
            "transaction_id": [f"TX{i}" for i in range(6)],
            "timestamp": pd.date_range("2025-01-01", periods=6, freq="h"),
            "amount": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
            "sender_balance_before": [100.0] * 6,
            "receiver_balance_before": [200.0] * 6,
            "sender_balance_after": [90.0] * 6,
            "receiver_balance_after": [210.0] * 6,
            "transaction_type": ["TRANSFER", "TRANSFER", "PAYMENT"] * 2,
            "channel": ["ONLINE", "BRANCH", "ONLINE"] * 2,
            "currency": ["USD"] * 6,
            "is_fraud": [0, 1, 0, 1, 0, 1],
            "typology": ["none", "synthetic"] * 3,
        })

    def test_training_and_inference_schemas_exclude_post_event_values(self):
        from app.ml_extensions.tgn.preprocessing import (
            create_transaction_features,
            fit_preprocessor,
            transform_transaction_features,
        )

        transactions = self._transactions()
        scaler, training_columns, training_features = fit_preprocessor(
            transactions,
            split_index=4,
        )
        inference_features = transform_transaction_features(
            transactions.drop(
                columns=[
                    "sender_balance_after",
                    "receiver_balance_after",
                ]
            ),
            scaler,
            training_columns,
        )

        forbidden = {
            "sender_balance_after",
            "receiver_balance_after",
            "is_fraud",
            "typology",
        }
        self.assertTrue(forbidden.isdisjoint(training_columns))
        self.assertEqual(list(training_features.columns), training_columns)
        self.assertEqual(list(inference_features.columns), training_columns)

        raw_features = create_transaction_features(transactions)
        self.assertTrue(forbidden.isdisjoint(raw_features.columns))

    def test_temporal_features_exclude_equal_timestamp_events(self):
        from app.ml_extensions.tgn.temporal_features import (
            create_temporal_features,
        )

        transactions = pd.DataFrame({
            "transaction_id": ["TX0", "TX1", "TX2", "TX3"],
            "timestamp": pd.to_datetime([
                "2025-01-01 00:00:00",
                "2025-01-01 01:00:00",
                "2025-01-01 01:00:00",
                "2025-01-01 02:00:00",
            ]),
            "sender_account": ["A", "A", "A", "A"],
            "receiver_account": ["R", "R", "R", "X"],
        })
        baseline = create_temporal_features(transactions)

        extra_peer = transactions.iloc[[1]].copy()
        extra_peer["sender_account"] = "A"
        extra_peer["receiver_account"] = "R-new"
        augmented = pd.concat(
            [transactions, extra_peer],
            ignore_index=True,
        )
        reordered = pd.concat(
            [
                augmented.iloc[[0]],
                augmented.iloc[[3, 2, 1, 4]],
            ],
            ignore_index=True,
        )
        augmented_features = create_temporal_features(augmented)
        reordered_features = create_temporal_features(reordered)

        for column in (
            "sender_time_delta_seconds",
            "receiver_time_delta_seconds",
            "log_time_delta",
        ):
            self.assertAlmostEqual(
                baseline.loc[1, column],
                3600.0 if column != "log_time_delta" else np.log1p(3600.0),
            )
            original_values = augmented_features.set_index(
                "transaction_id"
            ).loc[["TX1", "TX2"], column]
            reordered_values = reordered_features.set_index(
                "transaction_id"
            ).loc[["TX1", "TX2"], column]
            np.testing.assert_array_equal(
                original_values.to_numpy(),
                reordered_values.to_numpy(),
            )

    def test_future_events_do_not_change_earlier_temporal_features(self):
        from app.ml_extensions.tgn.temporal_features import (
            create_temporal_features,
        )

        transactions = pd.DataFrame({
            "timestamp": pd.to_datetime([
                "2025-01-01 00:00:00",
                "2025-01-01 01:00:00",
                "2025-01-01 02:00:00",
            ]),
            "sender_account": ["A", "A", "A"],
            "receiver_account": ["R0", "R1", "R2"],
        })
        earlier = create_temporal_features(transactions.iloc[:2])
        with_future = create_temporal_features(transactions)

        pd.testing.assert_frame_equal(
            earlier.drop(columns=["timestamp", "sender_account", "receiver_account"]),
            with_future.iloc[:2].drop(
                columns=["timestamp", "sender_account", "receiver_account"]
            ),
        )

    def test_split_keeps_equal_timestamp_batch_on_one_side(self):
        from app.ml_extensions.tgn.temporal_features import (
            chronological_split_index,
        )

        transactions = pd.DataFrame({
            "timestamp": pd.to_datetime([
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
        })
        split_index = chronological_split_index(transactions)

        self.assertEqual(split_index, 9)
        self.assertLess(
            transactions.iloc[split_index - 1]["timestamp"],
            transactions.iloc[split_index]["timestamp"],
        )

    def test_training_batches_events_with_equal_timestamps(self):
        from app.ml_extensions.tgn.train_temporal_tgn import (
            _timestamp_batch_ranges,
        )

        timestamps = torch.tensor(
            [10.0, 10.0, 20.0, 30.0, 30.0],
            dtype=torch.float64,
        )
        self.assertEqual(
            list(_timestamp_batch_ranges(timestamps, 0, len(timestamps))),
            [(0, 2), (2, 3), (3, 5)],
        )

    def test_training_updates_timestamp_batch_in_one_pass(self):
        from app.ml_extensions.tgn.train_temporal_tgn import train_one_epoch

        model = self._make_memory_model()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        loss = train_one_epoch(
            model=model,
            optimizer=optimizer,
            criterion=torch.nn.BCEWithLogitsLoss(),
            features=torch.tensor([
                [0.2, 1.0],
                [0.7, 2.0],
                [0.4, 3.0],
            ]),
            sender_ids=torch.tensor([0, 0, 3]),
            receiver_ids=torch.tensor([1, 2, 1]),
            delta_t=torch.zeros((3, 1)),
            timestamps=torch.full((3,), 100.0, dtype=torch.float64),
            labels=torch.tensor([0.0, 1.0, 0.0]),
            start_index=0,
            end_index=3,
        )

        self.assertTrue(np.isfinite(loss))
        self.assertTrue(torch.count_nonzero(model.memory.memory).item() > 0)

    def test_validation_scores_same_timestamp_rows_before_memory_update(self):
        from app.ml_extensions.tgn.train_temporal_tgn import evaluate

        initial_state = self._make_memory_model().state_dict()
        inputs = {
            "features": torch.tensor([
                [0.2, 1.0],
                [0.7, 2.0],
                [0.4, 3.0],
            ]),
            "sender_ids": torch.tensor([0, 0, 3]),
            "receiver_ids": torch.tensor([1, 2, 1]),
            "delta_t": torch.zeros((3, 1)),
            "timestamps": torch.full((3,), 100.0, dtype=torch.float64),
            "labels": torch.tensor([0.0, 1.0, 0.0]),
        }
        full_batch_model = self._make_memory_model()
        full_batch_model.load_state_dict(initial_state)
        full_result = evaluate(
            model=full_batch_model,
            start_index=0,
            end_index=3,
            **inputs,
        )

        target_only_model = self._make_memory_model()
        target_only_model.load_state_dict(initial_state)
        target_only_result = evaluate(
            model=target_only_model,
            features=inputs["features"][:1],
            sender_ids=inputs["sender_ids"][:1],
            receiver_ids=inputs["receiver_ids"][:1],
            delta_t=inputs["delta_t"][:1],
            timestamps=inputs["timestamps"][:1],
            labels=inputs["labels"][:1],
            start_index=0,
            end_index=1,
        )

        self.assertAlmostEqual(
            full_result["probabilities"][0],
            target_only_result["probabilities"][0],
            places=7,
        )

    def _make_memory_model(self):
        from app.ml_extensions.tgn.tgn_model import TemporalMemoryTGN

        torch.manual_seed(19)
        return TemporalMemoryTGN(
            input_dim=2,
            num_nodes=4,
            hidden_dim=4,
            message_dim=4,
        )

    def test_same_timestamp_predictions_are_order_and_addition_invariant(self):
        from app.ml_extensions.tgn.inference import predict_event_batch

        senders = torch.tensor([0, 0, 3])
        receivers = torch.tensor([1, 2, 1])
        features = torch.tensor([
            [0.2, 1.0],
            [0.7, 2.0],
            [0.4, 3.0],
        ])
        deltas = torch.zeros((3, 1))
        timestamps = torch.full((3,), 100.0)

        initial_state = self._make_memory_model().state_dict()
        base_model = self._make_memory_model()
        base_model.load_state_dict(initial_state)
        two_event_model = self._make_memory_model()
        two_event_model.load_state_dict(initial_state)
        target_only_model = self._make_memory_model()
        target_only_model.load_state_dict(initial_state)
        base_probabilities = predict_event_batch(
            base_model,
            senders,
            receivers,
            features,
            deltas,
            timestamps,
        )
        two_event_probabilities = predict_event_batch(
            two_event_model,
            senders[:2],
            receivers[:2],
            features[:2],
            deltas[:2],
            timestamps[:2],
        )
        target_only_probability = predict_event_batch(
            target_only_model,
            senders[:1],
            receivers[:1],
            features[:1],
            deltas[:1],
            timestamps[:1],
        )[0]

        reordered_model = self._make_memory_model()
        reordered_model.load_state_dict(initial_state)
        order = torch.tensor([2, 0, 1])
        reordered_probabilities = predict_event_batch(
            reordered_model,
            senders[order],
            receivers[order],
            features[order],
            deltas[order],
            timestamps[order],
        )

        self.assertAlmostEqual(
            base_probabilities[0],
            target_only_probability,
            places=7,
        )
        np.testing.assert_allclose(
            base_probabilities[:2],
            two_event_probabilities,
            rtol=1e-6,
            atol=1e-7,
        )
        np.testing.assert_allclose(
            base_probabilities,
            reordered_probabilities[[1, 2, 0]],
            rtol=1e-6,
            atol=1e-7,
        )
        torch.testing.assert_close(
            base_model.memory.memory,
            reordered_model.memory.memory,
            rtol=1e-6,
            atol=1e-7,
        )

    def test_future_event_does_not_change_earlier_prediction(self):
        from app.ml_extensions.tgn.inference import predict_event_batch

        first_model = self._make_memory_model()
        second_model = self._make_memory_model()
        second_model.load_state_dict(first_model.state_dict())

        earlier = predict_event_batch(
            first_model,
            torch.tensor([0]),
            torch.tensor([1]),
            torch.tensor([[0.2, 1.0]]),
            torch.tensor([[0.0]]),
            torch.tensor([100.0]),
        )[0]
        with_future_first = predict_event_batch(
            second_model,
            torch.tensor([0]),
            torch.tensor([1]),
            torch.tensor([[0.2, 1.0]]),
            torch.tensor([[0.0]]),
            torch.tensor([100.0]),
        )[0]
        _ = predict_event_batch(
            second_model,
            torch.tensor([0]),
            torch.tensor([2]),
            torch.tensor([[9.0, 8.0]]),
            torch.tensor([[1.0]]),
            torch.tensor([200.0]),
        )

        self.assertAlmostEqual(earlier, with_future_first, places=7)

    def test_tgn_rejects_forbidden_saved_feature_schema(self):
        from app.ml_extensions.tgn.preprocessing import (
            fit_preprocessor,
            transform_transaction_features,
        )

        transactions = self._transactions()
        scaler, _, _ = fit_preprocessor(transactions, split_index=4)
        with self.assertRaisesRegex(ValueError, "excluded columns"):
            transform_transaction_features(
                transactions,
                scaler,
                ["amount", "is_fraud"],
            )

    def test_transaction_labels_do_not_change_tgn_input_features(self):
        from app.ml_extensions.tgn.preprocessing import (
            create_transaction_features,
        )

        transactions = self._transactions()
        original = create_transaction_features(transactions)
        transactions["is_fraud"] = 1 - transactions["is_fraud"]
        transactions["typology"] = "different_label"
        changed_labels = create_transaction_features(transactions)

        pd.testing.assert_frame_equal(original, changed_labels)


if __name__ == "__main__":
    unittest.main()
