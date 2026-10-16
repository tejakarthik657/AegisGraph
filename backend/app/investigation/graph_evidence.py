from pathlib import Path

import networkx as nx
import pandas as pd


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]

RESULTS_DIR = PROJECT_ROOT / "data" / "results"

GRAPH_FILE = PROJECT_ROOT / "data" / "processed" / "transaction_graph.graphml"
INPUT_FILE = RESULTS_DIR / "case_reviews.csv"
OUTPUT_FILE = RESULTS_DIR / "case_graph_evidence.csv"


# ============================================================
# LOAD GRAPH
# ============================================================

def load_graph():
    if not GRAPH_FILE.exists():
        raise FileNotFoundError(
            f"Transaction graph not found:\n{GRAPH_FILE}"
        )

    graph = nx.read_graphml(GRAPH_FILE)

    return graph


# ============================================================
# LOAD CASE REVIEWS
# ============================================================

def load_cases():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Case review file not found:\n{INPUT_FILE}"
        )

    df = pd.read_csv(INPUT_FILE)

    required_columns = [
        "case_id",
        "alert_id",
        "transaction_id",
        "sender_account",
        "receiver_account",
        "amount",
        "risk_score",
        "risk_category",
        "case_priority",
        "review_status",
        "investigator_decision",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "Missing required case-review columns:\n"
            + "\n".join(missing_columns)
        )

    return df


# ============================================================
# GET ACCOUNT NEIGHBORS
# ============================================================

def get_neighbors(graph, account):
    if account not in graph:
        return []

    neighbors = set()

    # Outgoing relationships
    neighbors.update(
        str(node)
        for node in graph.successors(account)
    )

    # Incoming relationships
    neighbors.update(
        str(node)
        for node in graph.predecessors(account)
    )

    neighbors.discard(str(account))

    return sorted(neighbors)


# ============================================================
# GET ACCOUNT DEGREE
# ============================================================

def get_account_degree(graph, account):
    if account not in graph:
        return {
            "in_degree": 0,
            "out_degree": 0,
            "total_degree": 0,
        }

    in_degree = graph.in_degree(account)
    out_degree = graph.out_degree(account)

    return {
        "in_degree": int(in_degree),
        "out_degree": int(out_degree),
        "total_degree": int(in_degree + out_degree),
    }


# ============================================================
# GET RELATED TRANSACTION COUNT
# ============================================================

def get_related_transaction_count(
    graph,
    sender_account,
    receiver_account,
):
    count = 0

    if sender_account in graph:
        count += graph.out_degree(sender_account)
        count += graph.in_degree(sender_account)

    if receiver_account in graph:
        count += graph.out_degree(receiver_account)
        count += graph.in_degree(receiver_account)

    # The transaction connecting sender and receiver is counted
    # from both sides, so remove that overlap where applicable.
    if (
        sender_account in graph
        and receiver_account in graph
        and graph.has_edge(sender_account, receiver_account)
    ):
        count -= 2

    return max(int(count), 0)


# ============================================================
# BUILD GRAPH EVIDENCE FOR ONE CASE
# ============================================================

def build_case_evidence(graph, row):
    sender = str(row["sender_account"])
    receiver = str(row["receiver_account"])

    sender_neighbors = get_neighbors(
        graph,
        sender,
    )

    receiver_neighbors = get_neighbors(
        graph,
        receiver,
    )

    sender_degree = get_account_degree(
        graph,
        sender,
    )

    receiver_degree = get_account_degree(
        graph,
        receiver,
    )

    related_transaction_count = (
        get_related_transaction_count(
            graph,
            sender,
            receiver,
        )
    )

    return {
        "case_id": row["case_id"],
        "alert_id": row["alert_id"],
        "transaction_id": row["transaction_id"],

        "sender_account": sender,
        "receiver_account": receiver,

        "sender_in_degree": sender_degree["in_degree"],
        "sender_out_degree": sender_degree["out_degree"],
        "sender_total_degree": sender_degree["total_degree"],

        "receiver_in_degree": receiver_degree["in_degree"],
        "receiver_out_degree": receiver_degree["out_degree"],
        "receiver_total_degree": receiver_degree["total_degree"],

        "sender_neighbor_count": len(
            sender_neighbors
        ),

        "receiver_neighbor_count": len(
            receiver_neighbors
        ),

        "sender_neighbors": "|".join(
            sender_neighbors
        ),

        "receiver_neighbors": "|".join(
            receiver_neighbors
        ),

        "related_transaction_count": (
            related_transaction_count
        ),
    }


# ============================================================
# BUILD GRAPH EVIDENCE TABLE
# ============================================================

def build_evidence_table(cases, graph):
    evidence_rows = []

    for _, row in cases.iterrows():
        evidence = build_case_evidence(
            graph,
            row,
        )

        evidence_rows.append(evidence)

    return pd.DataFrame(evidence_rows)


# ============================================================
# VALIDATE GRAPH EVIDENCE
# ============================================================

def validate_evidence(evidence_df, cases):
    if len(evidence_df) != len(cases):
        raise ValueError(
            "Graph evidence row count does not match "
            "case count."
        )

    if evidence_df["case_id"].duplicated().any():
        raise ValueError(
            "Duplicate case IDs detected."
        )

    if evidence_df["transaction_id"].duplicated().any():
        raise ValueError(
            "Duplicate transaction IDs detected."
        )

    required_columns = [
        "sender_in_degree",
        "sender_out_degree",
        "receiver_in_degree",
        "receiver_out_degree",
        "related_transaction_count",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in evidence_df.columns
    ]

    if missing_columns:
        raise ValueError(
            "Missing graph evidence columns:\n"
            + "\n".join(missing_columns)
        )


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 70)
    print("AEGISGRAPH GRAPH INVESTIGATION EVIDENCE")
    print("=" * 70)

    graph = load_graph()

    print(
        f"Graph nodes: {graph.number_of_nodes()}"
    )

    print(
        f"Graph edges: {graph.number_of_edges()}"
    )

    cases = load_cases()

    print(
        f"Investigation cases loaded: {len(cases)}"
    )

    evidence_df = build_evidence_table(
        cases,
        graph,
    )

    validate_evidence(
        evidence_df,
        cases,
    )

    print()
    print("Graph evidence summary:")

    print(
        f"  Cases enriched: {len(evidence_df)}"
    )

    print(
        "  Average sender neighbor count: "
        f"{evidence_df['sender_neighbor_count'].mean():.2f}"
    )

    print(
        "  Average receiver neighbor count: "
        f"{evidence_df['receiver_neighbor_count'].mean():.2f}"
    )

    print(
        "  Average related transaction count: "
        f"{evidence_df['related_transaction_count'].mean():.2f}"
    )

    print()
    print("Validation:")

    print(
        f"  Duplicate case IDs: "
        f"{evidence_df['case_id'].duplicated().sum()}"
    )

    print(
        f"  Duplicate transaction IDs: "
        f"{evidence_df['transaction_id'].duplicated().sum()}"
    )

    evidence_df.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print()
    print("Saved to:")
    print(OUTPUT_FILE)

    print("=" * 70)


if __name__ == "__main__":
    main()