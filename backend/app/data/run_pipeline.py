import sys
from pathlib import Path

import pandas as pd

BACKEND_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(BACKEND_DIR))

from app.data.generator.generate_customers import (
    generate_customers,
)

from app.data.generator.generate_accounts import (
    generate_accounts,
)

from app.data.generator.generate_transactions import (
    generate_transactions,
)

from app.data.generator.inject_typologies import (
    inject_fraud_typologies,
)

from app.data.ingestion.validate import (
    validate_transactions,
)

from app.data.ingestion.normalize import (
    normalize_transactions,
)

from app.data.ingestion.dedupe import (
    deduplicate_transactions,
)

from app.data.graph.graph_builder import (
    build_transaction_graph,
)

from app.utils.config import (
    TRANSACTIONS_FILE,
    CLEAN_TRANSACTIONS_FILE,
)


def run_module_1():

    print("\n")
    print("=" * 60)
    print("        AEGISGRAPH — MODULE 1")
    print("        DATA PIPELINE")
    print("=" * 60)

    # STEP 1
    print("\n[1/8] Generating customers...")
    generate_customers(
        num_customers=1000
    )

    # STEP 2
    print("\n[2/8] Generating accounts...")
    generate_accounts()

    # STEP 3
    print("\n[3/8] Generating transactions...")
    generate_transactions(
        num_transactions=10000
    )

    # STEP 4
    print("\n[4/8] Injecting fraud typologies...")
    inject_fraud_typologies()

    # Load final transaction data
    df = pd.read_csv(
        TRANSACTIONS_FILE
    )

    # STEP 5
    print("\n[5/8] Validating...")
    validate_transactions(df)

    # STEP 6
    print("\n[6/8] Normalizing...")
    df = normalize_transactions(df)

    # STEP 7
    print("\n[7/8] Deduplicating...")
    df = deduplicate_transactions(df)

    # STEP 8
    print("\n[8/8] Building graph...")
    graph, node_features = (
        build_transaction_graph()
    )

    print("\n")
    print("=" * 60)
    print("        MODULE 1 COMPLETED")
    print("=" * 60)

    print(
        f"\nCustomers : 1000"
    )

    print(
        f"Accounts  : "
        f"{len(pd.read_csv('data/raw/accounts.csv'))}"
    )

    print(
        f"Transactions : {len(df)}"
    )

    print(
        f"Graph Nodes : "
        f"{graph.number_of_nodes()}"
    )

    print(
        f"Graph Edges : "
        f"{graph.number_of_edges()}"
    )

    print("\nFraud distribution:")

    print(
        df[df["is_fraud"] == 1]
        ["typology"]
        .value_counts()
    )

    print("\nModule 1 output files:")

    print("✓ data/raw/customers.csv")
    print("✓ data/raw/accounts.csv")
    print("✓ data/raw/transactions.csv")
    print("✓ data/processed/transactions_clean.csv")
    print("✓ data/processed/transaction_graph.graphml")
    print("✓ data/processed/node_features.csv")


if __name__ == "__main__":
    run_module_1()