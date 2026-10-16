import sys
from pathlib import Path

import networkx as nx
import pandas as pd


# ---------------------------------------------------------
# Add backend directory to Python path
# ---------------------------------------------------------
BACKEND_DIR = Path(__file__).resolve().parents[3]
sys.path.append(str(BACKEND_DIR))


# ---------------------------------------------------------
# Project configuration
# ---------------------------------------------------------
from app.utils.config import (
    CLEAN_TRANSACTIONS_FILE,
    GRAPH_FILE,
    NODE_FEATURES_FILE,
    create_directories,
)


# ---------------------------------------------------------
# Build transaction graph
# ---------------------------------------------------------
def build_transaction_graph(
    transactions: pd.DataFrame,
) -> nx.MultiDiGraph:
    """
    Build a directed multi-graph from transactions.

    Nodes:
        Bank accounts

    Edges:
        Financial transactions between accounts

    Multiple transactions between the same accounts
    are preserved using MultiDiGraph.
    """

    print(
        "\n========== GRAPH CONSTRUCTION =========="
    )

    # -----------------------------------------------------
    # Create MultiDiGraph
    # -----------------------------------------------------
    graph = nx.MultiDiGraph()

    # -----------------------------------------------------
    # Add transaction edges
    # -----------------------------------------------------
    for _, transaction in transactions.iterrows():

        sender = transaction[
            "sender_account"
        ]

        receiver = transaction[
            "receiver_account"
        ]

        # -------------------------------------------------
        # Add sender node
        # -------------------------------------------------
        if not graph.has_node(sender):

            graph.add_node(
                sender,
                account_id=sender,
            )

        # -------------------------------------------------
        # Add receiver node
        # -------------------------------------------------
        if not graph.has_node(receiver):

            graph.add_node(
                receiver,
                account_id=receiver,
            )

        # -------------------------------------------------
        # Add transaction edge
        # -------------------------------------------------
        graph.add_edge(
            sender,
            receiver,

            transaction_id=(
                transaction["transaction_id"]
            ),

            amount=float(
                transaction["amount"]
            ),

            timestamp=str(
                transaction["timestamp"]
            ),

            currency=(
                transaction["currency"]
            ),

            transaction_type=(
                transaction["transaction_type"]
            ),

            channel=(
                transaction["channel"]
            ),

            sender_country=(
                transaction["sender_country"]
            ),

            receiver_country=(
                transaction["receiver_country"]
            ),

            status=(
                transaction["status"]
            ),

            is_fraud=int(
                transaction["is_fraud"]
            ),

            typology=(
                transaction["typology"]
            ),
        )

    print(
        f"Graph nodes: {graph.number_of_nodes()}"
    )

    print(
        f"Graph edges: {graph.number_of_edges()}"
    )

    return graph


# ---------------------------------------------------------
# Generate node features
# ---------------------------------------------------------
def generate_node_features(
    graph: nx.MultiDiGraph,
) -> pd.DataFrame:
    """
    Generate account-level graph features.

    These features describe the structural behavior
    of each account in the transaction network.
    """

    print(
        "\n========== NODE FEATURES =========="
    )

    records = []

    for node in graph.nodes:

        # -------------------------------------------------
        # Basic degree features
        # -------------------------------------------------
        in_degree = graph.in_degree(
            node
        )

        out_degree = graph.out_degree(
            node
        )

        # -------------------------------------------------
        # Incoming transaction amount
        # -------------------------------------------------
        total_incoming = 0.0

        for _, _, data in graph.in_edges(
            node,
            data=True,
        ):
            total_incoming += float(
                data.get(
                    "amount",
                    0,
                )
            )

        # -------------------------------------------------
        # Outgoing transaction amount
        # -------------------------------------------------
        total_outgoing = 0.0

        for _, _, data in graph.out_edges(
            node,
            data=True,
        ):
            total_outgoing += float(
                data.get(
                    "amount",
                    0,
                )
            )

        # -------------------------------------------------
        # Unique neighbors
        # -------------------------------------------------
        neighbors = set()

        for neighbor in graph.predecessors(
            node
        ):
            neighbors.add(neighbor)

        for neighbor in graph.successors(
            node
        ):
            neighbors.add(neighbor)

        unique_neighbors = len(
            neighbors
        )

        # -------------------------------------------------
        # Account feature record
        # -------------------------------------------------
        records.append(
            {
                "account_id": node,

                "in_degree": in_degree,

                "out_degree": out_degree,

                "total_incoming":
                    round(
                        total_incoming,
                        2,
                    ),

                "total_outgoing":
                    round(
                        total_outgoing,
                        2,
                    ),

                "unique_neighbors":
                    unique_neighbors,
            }
        )

    node_features = pd.DataFrame(
        records
    )

    print(
        f"Generated features for "
        f"{len(node_features)} accounts."
    )

    return node_features


# ---------------------------------------------------------
# Save graph
# ---------------------------------------------------------
def save_graph(
    graph: nx.MultiDiGraph,
):
    """
    Save graph to GraphML.
    """

    create_directories()

    nx.write_graphml(
        graph,
        GRAPH_FILE,
    )

    print(
        f"Graph saved to:"
        f"\n{GRAPH_FILE}"
    )


# ---------------------------------------------------------
# Save node features
# ---------------------------------------------------------
def save_node_features(
    node_features: pd.DataFrame,
):
    """
    Save node-level graph features.
    """

    create_directories()

    node_features.to_csv(
        NODE_FEATURES_FILE,
        index=False,
    )

    print(
        f"Node features saved to:"
        f"\n{NODE_FEATURES_FILE}"
    )


# ---------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------
def build_graph_pipeline():

    print(
        "\n========================================"
    )

    print(
        "        AEGISGRAPH GRAPH BUILDER"
    )

    print(
        "========================================"
    )

    # -----------------------------------------------------
    # Load cleaned transactions
    # -----------------------------------------------------
    print(
        "\nLoading transactions from:"
        f"\n{CLEAN_TRANSACTIONS_FILE}"
    )

    transactions = pd.read_csv(
        CLEAN_TRANSACTIONS_FILE
    )

    print(
        f"Loaded {len(transactions)} transactions."
    )

    # -----------------------------------------------------
    # Build graph
    # -----------------------------------------------------
    graph = build_transaction_graph(
        transactions
    )

    # -----------------------------------------------------
    # Generate node features
    # -----------------------------------------------------
    node_features = (
        generate_node_features(
            graph
        )
    )

    # -----------------------------------------------------
    # Save outputs
    # -----------------------------------------------------
    save_graph(graph)

    save_node_features(
        node_features
    )

    # -----------------------------------------------------
    # Final summary
    # -----------------------------------------------------
    print(
        "\n========================================"
    )

    print(
        "       GRAPH BUILDING COMPLETE"
    )

    print(
        "========================================"
    )

    print(
        f"Nodes : "
        f"{graph.number_of_nodes()}"
    )

    print(
        f"Edges : "
        f"{graph.number_of_edges()}"
    )

    print(
        f"Features: "
        f"{len(node_features)} accounts"
    )

    return graph, node_features


# ---------------------------------------------------------
# Script entry point
# ---------------------------------------------------------
if __name__ == "__main__":
    build_graph_pipeline()