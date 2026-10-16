"""
AegisGraph — Neighborhood Risk Scoring.

Computes a risk score for each account node based on
the risk characteristics of its graph neighborhood.
"""

from __future__ import annotations

import networkx as nx
import pandas as pd


def compute_neighborhood_risk(
    graph: nx.Graph,
    risk_df: pd.DataFrame,
    hops: int = 2,
) -> pd.DataFrame:
    """
    Score each account based on the average risk of
    transactions in its neighborhood.

    Parameters
    ----------
    graph:
        Transaction graph (accounts as nodes, transactions
        as edges).

    risk_df:
        DataFrame containing transaction_id and risk_score.

    hops:
        Neighborhood depth.

    Returns
    -------
    DataFrame with columns: account_id, neighborhood_risk,
        neighbor_count, high_risk_neighbors.
    """

    risk_lookup = {}

    if "transaction_id" in risk_df.columns:
        risk_lookup = dict(
            zip(
                risk_df["transaction_id"].astype(str),
                risk_df["risk_score"],
            )
        )

    results = []

    for node in graph.nodes():
        try:
            neighbors = set()
            current_level = {node}

            for _ in range(hops):
                next_level = set()
                for n in current_level:
                    next_level.update(graph.neighbors(n))
                neighbors.update(next_level)
                current_level = next_level - {node}

            neighbors.discard(node)

            # Collect risk scores from edges in neighborhood
            neighborhood_risks = []

            for neighbor in neighbors:
                if graph.is_multigraph():
                    for _, _, data in graph.edges(
                        neighbor, data=True
                    ):
                        tx_id = str(
                            data.get("transaction_id", "")
                        )
                        if tx_id in risk_lookup:
                            neighborhood_risks.append(
                                risk_lookup[tx_id]
                            )
                else:
                    for _, _, data in graph.edges(
                        neighbor, data=True
                    ):
                        tx_id = str(
                            data.get("transaction_id", "")
                        )
                        if tx_id in risk_lookup:
                            neighborhood_risks.append(
                                risk_lookup[tx_id]
                            )

            avg_risk = 0.0
            high_risk_count = 0

            if neighborhood_risks:
                avg_risk = sum(neighborhood_risks) / len(
                    neighborhood_risks
                )
                high_risk_count = sum(
                    1 for r in neighborhood_risks if r >= 0.75
                )

            results.append({
                "account_id": str(node),
                "neighborhood_risk": round(avg_risk, 4),
                "neighbor_count": len(neighbors),
                "high_risk_neighbors": high_risk_count,
            })

        except Exception:
            results.append({
                "account_id": str(node),
                "neighborhood_risk": 0.0,
                "neighbor_count": 0,
                "high_risk_neighbors": 0,
            })

    return pd.DataFrame(results)
