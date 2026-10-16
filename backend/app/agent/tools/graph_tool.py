"""AegisGraph — Graph Analysis Tool."""

from __future__ import annotations
import networkx as nx
from app.utils.config import GRAPH_FILE


def analyze_graph_neighborhood(account_id: str, hops: int = 2) -> dict:
    """Analyze the graph neighborhood of an account."""
    if not GRAPH_FILE.exists():
        return {"error": "Graph file not found", "account_id": account_id}
    graph = nx.read_graphml(GRAPH_FILE)
    if account_id not in graph:
        return {"error": "Account not in graph", "account_id": account_id}
    neighbors = set()
    current = {account_id}
    for _ in range(hops):
        next_level = set()
        for n in current:
            next_level.update(graph.neighbors(n))
        neighbors.update(next_level)
        current = next_level - {account_id}
    neighbors.discard(account_id)
    return {
        "account_id": account_id,
        "degree": int(graph.degree(account_id)),
        "neighbor_count": len(neighbors),
        "hops": hops,
    }
