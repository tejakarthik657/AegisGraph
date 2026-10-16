"""
AegisGraph — TGN Module Tests.

Tests the Temporal Graph Network components:
memory, message, time encoding, and model architecture.
"""

import unittest

import numpy as np


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


if __name__ == "__main__":
    unittest.main()
