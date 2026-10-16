from pathlib import Path


# backend/app/utils/config.py
# parents[0] = utils
# parents[1] = app
# parents[2] = backend
# parents[3] = aegisgraph

PROJECT_ROOT = Path(__file__).resolve().parents[3]

DATA_DIR = PROJECT_ROOT / "data"

RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
RESULTS_DATA_DIR = DATA_DIR / "results"


CUSTOMERS_FILE = RAW_DATA_DIR / "customers.csv"
ACCOUNTS_FILE = RAW_DATA_DIR / "accounts.csv"
TRANSACTIONS_FILE = RAW_DATA_DIR / "transactions.csv"

CLEAN_TRANSACTIONS_FILE = (
    PROCESSED_DATA_DIR / "transactions_clean.csv"
)

GRAPH_FILE = (
    PROCESSED_DATA_DIR / "transaction_graph.graphml"
)

NODE_FEATURES_FILE = (
    PROCESSED_DATA_DIR / "node_features.csv"
)


def create_directories():
    """Create required data directories."""

    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DATA_DIR.mkdir(parents=True, exist_ok=True)