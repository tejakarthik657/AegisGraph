"""
AegisGraph — Cycle Detection.

Detects circular transaction patterns where funds
flow through a chain and return to the originator.
"""

from __future__ import annotations

import networkx as nx
import pandas as pd


DEFAULT_MAX_CYCLE_LENGTH = 5


def detect_cycles(
    transactions: pd.DataFrame,
    max_cycle_length: int = DEFAULT_MAX_CYCLE_LENGTH,
) -> pd.DataFrame:
    """
    Detect circular fund flows in the transaction graph.

    Returns
    -------
    DataFrame with an added `cycle_member` boolean column.
    """

    df = transactions.copy()
    df["cycle_member"] = 0

    # Build directed graph
    graph = nx.DiGraph()

    for _, row in df.iterrows():
        graph.add_edge(
            str(row["sender_account"]),
            str(row["receiver_account"]),
            transaction_id=str(
                row.get("transaction_id", "")
            ),
        )

    # Find simple cycles
    try:
        cycles = list(
            nx.simple_cycles(graph, length_bound=max_cycle_length)
        )
    except TypeError:
        # Fallback for older networkx without length_bound
        cycles = [
            c for c in nx.simple_cycles(graph)
            if len(c) <= max_cycle_length
        ]

    cycle_accounts = set()
    for cycle in cycles:
        cycle_accounts.update(cycle)

    if cycle_accounts:
        sender_mask = df["sender_account"].astype(str).isin(
            cycle_accounts
        )
        receiver_mask = df["receiver_account"].astype(str).isin(
            cycle_accounts
        )

        df.loc[sender_mask & receiver_mask, "cycle_member"] = 1

    return df
