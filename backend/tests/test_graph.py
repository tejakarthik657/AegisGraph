"""
AegisGraph — Graph Module Tests.

Tests the graph builder and graph utility functions.
"""

import unittest

import networkx as nx
import pandas as pd


class TestGraphBuilder(unittest.TestCase):
    """Test graph building from transactions."""

    def _make_transactions(self):
        return pd.DataFrame({
            "transaction_id": ["TX001", "TX002", "TX003"],
            "sender_account": ["ACC001", "ACC002", "ACC001"],
            "receiver_account": ["ACC002", "ACC003", "ACC003"],
            "amount": [100.0, 200.0, 300.0],
            "timestamp": [
                "2024-01-01 10:00:00",
                "2024-01-01 11:00:00",
                "2024-01-01 12:00:00",
            ],
            "currency": ["USD", "USD", "USD"],
            "is_fraud": [0, 0, 1],
            "typology": ["none", "none", "smurfing"],
        })

    def test_build_graph_creates_nodes(self):
        """Ensure graph has correct accounts as nodes."""
        df = self._make_transactions()

        graph = nx.MultiDiGraph()
        for _, row in df.iterrows():
            graph.add_edge(
                row["sender_account"],
                row["receiver_account"],
                transaction_id=row["transaction_id"],
                amount=row["amount"],
            )

        self.assertIn("ACC001", graph.nodes())
        self.assertIn("ACC002", graph.nodes())
        self.assertIn("ACC003", graph.nodes())

    def test_build_graph_creates_edges(self):
        """Ensure graph has correct number of edges."""
        df = self._make_transactions()

        graph = nx.MultiDiGraph()
        for _, row in df.iterrows():
            graph.add_edge(
                row["sender_account"],
                row["receiver_account"],
                transaction_id=row["transaction_id"],
                amount=row["amount"],
            )

        self.assertEqual(graph.number_of_edges(), 3)

    def test_graph_edge_attributes(self):
        """Test edge attributes are preserved."""
        graph = nx.MultiDiGraph()
        graph.add_edge(
            "ACC001", "ACC002",
            transaction_id="TX001",
            amount=500.0,
        )

        edges = list(graph.edges(data=True))
        self.assertEqual(len(edges), 1)
        self.assertEqual(
            edges[0][2]["transaction_id"], "TX001"
        )
        self.assertEqual(edges[0][2]["amount"], 500.0)


class TestFindTransactionEdge(unittest.TestCase):
    """Test the find_transaction_edge utility."""

    def test_find_existing_edge(self):
        from app.main import find_transaction_edge

        graph = nx.MultiDiGraph()
        graph.add_edge(
            "ACC001", "ACC002",
            transaction_id="TX001",
            amount=100.0,
        )

        result = find_transaction_edge(graph, "TX001")
        self.assertIsNotNone(result)
        self.assertEqual(result["source"], "ACC001")
        self.assertEqual(result["target"], "ACC002")

    def test_find_missing_edge(self):
        from app.main import find_transaction_edge

        graph = nx.MultiDiGraph()
        graph.add_edge(
            "ACC001", "ACC002",
            transaction_id="TX001",
        )

        result = find_transaction_edge(graph, "TX999")
        self.assertIsNone(result)


class TestBuildAccountNode(unittest.TestCase):
    """Test the build_account_node utility."""

    def test_existing_node(self):
        from app.main import build_account_node

        graph = nx.MultiDiGraph()
        graph.add_edge("ACC001", "ACC002")

        node = build_account_node(
            graph, "ACC001", "sender", central=True
        )

        self.assertEqual(node["id"], "account:ACC001")
        self.assertEqual(node["type"], "sender")
        self.assertTrue(node["central"])
        self.assertGreater(node["connections"], 0)

    def test_missing_node(self):
        from app.main import build_account_node

        graph = nx.MultiDiGraph()

        node = build_account_node(
            graph, "ACC999", "account"
        )

        self.assertEqual(node["connections"], 0)
        self.assertFalse(node["central"])


if __name__ == "__main__":
    unittest.main()
