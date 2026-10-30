from pathlib import Path

import networkx as nx
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from app.api.routes.cases_routes import router as case_management_router
from app.rules.aml_rules import score_transactions_chronologically

# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = PROJECT_ROOT / "data" / "results"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

# ============================================================
# TRANSACTION DATA
# ============================================================

TRANSACTIONS_FILE = (
    PROCESSED_DIR / "transactions_clean.csv"
)


_transaction_dataframe = None


def load_transaction_dataframe():
    """
    Load the cleaned transaction dataset once and reuse it.

    The transaction dataset is read-only here.
    No fraud labels or model outputs are modified.
    """
    global _transaction_dataframe

    if _transaction_dataframe is None:
        if not TRANSACTIONS_FILE.exists():
            raise HTTPException(
                status_code=404,
                detail=(
                    "Transaction dataset not found: "
                    f"{TRANSACTIONS_FILE}"
                ),
            )

        try:
            df = pd.read_csv(TRANSACTIONS_FILE)

            if "timestamp" in df.columns:
                df["timestamp"] = df["timestamp"].astype(str)

            _transaction_dataframe = df

        except Exception as exc:
            raise HTTPException(
                status_code=500,
                detail=(
                    "Failed to load transaction dataset: "
                    f"{exc}"
                ),
            )

    return _transaction_dataframe

# ============================================================
# FILE PATHS
# ============================================================

ALERTS_FILE = RESULTS_DIR / "alerts.csv"
CASES_FILE = RESULTS_DIR / "investigation_cases.csv"
REVIEWS_FILE = RESULTS_DIR / "case_reviews.csv"
DOSSIERS_FILE = RESULTS_DIR / "investigation_dossiers.csv"
RISK_FILE = RESULTS_DIR / "risk_fusion_predictions.csv"
AML_RULE_RESULTS_FILE = RESULTS_DIR / "aml_rule_results.csv"
MODEL_METRIC_SOURCES = (
    {
        "key": "random_forest",
        "name": "Random Forest",
        "file": RESULTS_DIR / "model_metrics.csv",
    },
    {
        "key": "tgn",
        "name": "Temporal Graph Network (TGN)",
        "file": RESULTS_DIR / "tgn_metrics.csv",
    },
    {
        "key": "thgnn",
        "name": (
            "Temporal Heterogeneous Graph Neural Network (THGNN)"
        ),
        "file": RESULTS_DIR / "thgnn_metrics.csv",
    },
    {
        "key": "aml_rules",
        "name": "AML Rule Detection",
        "file": RESULTS_DIR / "aml_rule_evaluation.csv",
    },
    {
        "key": "risk_fusion",
        "name": "Risk Fusion",
        "file": RESULTS_DIR / "risk_fusion_metrics.csv",
    },
)
AML_RULE_METRICS_FILE = (
    RESULTS_DIR / "aml_individual_rule_metrics.csv"
)

ACTIVITY_RULES = (
    "rapid_movement",
    "fan_out",
    "structuring",
    "large_transaction",
    "circular_movement",
)

_activity_pattern_counts = None

NETWORK_GRAPH_FILE = (
    PROCESSED_DIR / "transaction_graph.graphml"
)


# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="AegisGraph API",
    description=(
        "AML and fraud detection backend API for "
        "risk, alerts, investigations, and graph evidence."
    ),
    version="1.1.0",
)


# ============================================================
# CORS CONFIGURATION
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:5173",
        "http://localhost:5173",
        "http://127.0.0.1:5500",
        "http://localhost:5500",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(case_management_router)

# ============================================================
# GRAPH CACHE
# ============================================================

_GRAPH = None


def load_transaction_graph():
    """
    Load the existing NetworkX transaction graph.

    The graph is loaded once and reused for subsequent requests.
    """
    global _GRAPH

    if _GRAPH is not None:
        return _GRAPH

    if not NETWORK_GRAPH_FILE.exists():
        raise HTTPException(
            status_code=404,
            detail=(
                "Transaction graph not found: "
                f"{NETWORK_GRAPH_FILE}"
            ),
        )

    try:
        _GRAPH = nx.read_graphml(
            NETWORK_GRAPH_FILE
        )

        return _GRAPH

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=(
                "Failed to load transaction graph: "
                f"{exc}"
            ),
        )


# ============================================================
# HELPERS
# ============================================================

def load_csv(file_path: Path):
    """
    Load a CSV file from the project results directory.
    """
    if not file_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Data file not found: {file_path.name}",
        )

    try:
        return pd.read_csv(file_path)

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=(
                f"Failed to load {file_path.name}: "
                f"{exc}"
            ),
        )


def metric_value(row, column):
    if column not in row:
        return None

    value = row[column]
    if pd.isna(value):
        return None

    if hasattr(value, "item"):
        return value.item()

    return value


def read_evaluation_metric(file_path, model_name):
    metric_names = (
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "pr_auc",
        "accuracy",
    )
    evaluation_fields = (
        "train_transactions",
        "test_transactions",
        "test_accounts",
        "test_frauds",
        "test_fraud",
    )
    result = {
        "key": next(
            source["key"]
            for source in MODEL_METRIC_SOURCES
            if source["file"] == file_path
        ),
        "model_name": model_name,
        "available": False,
        "source_file": file_path.name,
        "metrics": {
            metric: None
            for metric in metric_names
        },
        "threshold": None,
        "evaluation": {},
    }

    if not file_path.exists():
        return result

    metrics_df = load_csv(file_path)
    if metrics_df.empty:
        return result

    row = metrics_df.iloc[0]
    source_model_name = metric_value(row, "model")
    if source_model_name:
        result["source_model_name"] = str(source_model_name)

    result["available"] = True
    result["metrics"] = {
        metric: metric_value(row, metric)
        for metric in metric_names
    }
    result["threshold"] = metric_value(row, "threshold")
    result["evaluation"] = {
        field: metric_value(row, field)
        for field in evaluation_fields
        if field in metrics_df.columns
    }
    return result


def model_metrics():
    models = [
        read_evaluation_metric(
            source["file"],
            source["name"],
        )
        for source in MODEL_METRIC_SOURCES
    ]

    aml_rules = next(
        model
        for model in models
        if model["key"] == "aml_rules"
    )
    aml_rules["rules"] = []

    if AML_RULE_METRICS_FILE.exists():
        rule_metrics_df = load_csv(AML_RULE_METRICS_FILE)
        rule_metric_columns = (
            "accuracy",
            "precision",
            "recall",
            "f1",
            "roc_auc",
            "pr_auc",
        )
        rule_evaluation_columns = (
            "test_transactions",
            "test_fraud",
        )

        if "rule" in rule_metrics_df.columns:
            for _, row in rule_metrics_df.iterrows():
                aml_rules["rules"].append(
                    {
                        "rule_name": str(row["rule"]),
                        "metrics": {
                            metric: metric_value(row, metric)
                            for metric in rule_metric_columns
                        },
                        "evaluation": {
                            field: metric_value(row, field)
                            for field in rule_evaluation_columns
                            if field in rule_metrics_df.columns
                        },
                    }
                )

    return {"models": models}


def get_activity_pattern_counts(transactions):
    """
    Count transaction-level AML flags across the full dataset.

    Reuse persisted rule results only when every dataset transaction
    has exactly one valid result for every activity rule.
    """
    global _activity_pattern_counts

    if _activity_pattern_counts is not None:
        return _activity_pattern_counts

    if (
        "transaction_id" not in transactions.columns
        or transactions["transaction_id"].isna().any()
        or transactions["transaction_id"].duplicated().any()
    ):
        raise HTTPException(
            status_code=500,
            detail=(
                "The transaction dataset must contain unique, "
                "non-null transaction IDs to calculate activity patterns."
            ),
        )

    transaction_ids = set(
        transactions["transaction_id"].astype(str)
    )
    expected_result_count = (
        len(transaction_ids) * len(ACTIVITY_RULES)
    )

    if AML_RULE_RESULTS_FILE.exists():
        rule_results = load_csv(AML_RULE_RESULTS_FILE)
        required_columns = {
            "transaction_id",
            "rule",
            "triggered",
        }

        if required_columns.issubset(rule_results.columns):
            relevant_results = rule_results[
                rule_results["rule"].isin(ACTIVITY_RULES)
            ].copy()

            relevant_results["transaction_id"] = (
                relevant_results["transaction_id"].astype(str)
            )

            triggered_values = (
                relevant_results["triggered"]
                .astype("string")
                .str.strip()
                .str.lower()
                .map(
                    {
                        "true": 1,
                        "false": 0,
                        "1": 1,
                        "0": 0,
                    }
                )
            )

            complete_results = (
                len(relevant_results) == expected_result_count
                and not relevant_results["transaction_id"].isna().any()
                and not relevant_results["rule"].isna().any()
                and not relevant_results.duplicated(
                    subset=["transaction_id", "rule"]
                ).any()
                and set(relevant_results["transaction_id"])
                == transaction_ids
                and set(relevant_results["rule"]) == set(ACTIVITY_RULES)
                and not triggered_values.isna().any()
            )

            if complete_results:
                _activity_pattern_counts = {
                    rule: int(
                        relevant_results.loc[
                            (relevant_results["rule"] == rule)
                            & (triggered_values == 1),
                            "transaction_id",
                        ].nunique()
                    )
                    for rule in ACTIVITY_RULES
                }
                return _activity_pattern_counts

    rule_input_columns = [
        "transaction_id",
        "timestamp",
        "sender_account",
        "receiver_account",
        "amount",
    ]
    missing_columns = [
        column
        for column in rule_input_columns
        if column not in transactions.columns
    ]

    if missing_columns:
        raise HTTPException(
            status_code=500,
            detail=(
                "Cannot calculate activity patterns; transaction "
                "data is missing required rule inputs: "
                + ", ".join(missing_columns)
            ),
        )

    scored_transactions = score_transactions_chronologically(
        transactions[rule_input_columns].copy()
    )

    scored_ids = scored_transactions["transaction_id"].astype(str)
    if (
        len(scored_transactions) != len(transaction_ids)
        or scored_ids.duplicated().any()
        or set(scored_ids) != transaction_ids
    ):
        raise HTTPException(
            status_code=500,
            detail=(
                "AML rules did not produce flags for every transaction "
                "in the dataset."
            ),
        )

    counts = {}
    for rule in ACTIVITY_RULES:
        if rule not in scored_transactions.columns:
            raise HTTPException(
                status_code=500,
                detail=(
                    "AML rule evaluation did not produce the "
                    f"required activity flag: {rule}"
                ),
            )

        values = pd.to_numeric(
            scored_transactions[rule],
            errors="coerce",
        )

        if values.isna().any() or not values.isin([0, 1]).all():
            raise HTTPException(
                status_code=500,
                detail=(
                    "AML rule evaluation produced missing or invalid "
                    f"values for activity flag: {rule}"
                ),
            )

        counts[rule] = int(
            scored_transactions.loc[
                values == 1,
                "transaction_id",
            ].nunique()
        )

    _activity_pattern_counts = counts
    return _activity_pattern_counts


def dataframe_to_records(df):
    """
    Convert pandas DataFrame values into JSON-safe
    native Python objects.
    """
    records = df.to_dict(orient="records")

    safe_records = []

    for record in records:
        safe_record = {}

        for key, value in record.items():

            if pd.isna(value):
                safe_record[key] = None

            elif hasattr(value, "item"):
                safe_record[key] = value.item()

            else:
                safe_record[key] = value

        safe_records.append(safe_record)

    return safe_records


def safe_float(value, default=0.0):
    """
    Convert a graph attribute into a JSON-safe float.
    """
    try:
        if value is None:
            return default

        return float(value)

    except (TypeError, ValueError):
        return default


def safe_int(value, default=0):
    """
    Convert a graph attribute into a JSON-safe integer.
    """
    try:
        if value is None:
            return default

        return int(float(value))

    except (TypeError, ValueError):
        return default


def safe_string(value, default=None):
    """
    Convert a graph attribute into a clean string.
    """
    if value is None:
        return default

    return str(value)


def edge_transaction_id(edge_data):
    """
    Extract the transaction ID stored on a graph edge.
    """
    return safe_string(
        edge_data.get("transaction_id")
    )


def find_transaction_edge(graph, transaction_id):
    """
    Find the graph edge corresponding to a transaction ID.

    The graph is a MultiDiGraph, but this also handles
    ordinary DiGraph-style edge structures defensively.
    """

    transaction_id = str(transaction_id)

    if graph.is_multigraph():

        for source, target, key, data in graph.edges(
            keys=True,
            data=True,
        ):
            graph_transaction_id = (
                edge_transaction_id(data)
            )

            if graph_transaction_id == transaction_id:
                return {
                    "source": str(source),
                    "target": str(target),
                    "key": key,
                    "data": data,
                }

    else:

        for source, target, data in graph.edges(
            data=True
        ):
            graph_transaction_id = (
                edge_transaction_id(data)
            )

            if graph_transaction_id == transaction_id:
                return {
                    "source": str(source),
                    "target": str(target),
                    "key": None,
                    "data": data,
                }

    return None


def graph_edge_record(
    source,
    target,
    edge_data,
):
    """
    Convert a graph transaction edge into the
    API representation used by the frontend.
    """

    transaction_id = edge_transaction_id(
        edge_data
    )

    return {
        "transaction_id": transaction_id,

        "sender_account": str(source),

        "receiver_account": str(target),

        "amount": safe_float(
            edge_data.get("amount")
        ),

        "timestamp": safe_string(
            edge_data.get("timestamp")
        ),

        "currency": safe_string(
            edge_data.get("currency")
        ),

        "transaction_type": safe_string(
            edge_data.get("transaction_type")
        ),

        "channel": safe_string(
            edge_data.get("channel")
        ),

        "sender_country": safe_string(
            edge_data.get("sender_country")
        ),

        "receiver_country": safe_string(
            edge_data.get("receiver_country")
        ),

        "status": safe_string(
            edge_data.get("status")
        ),

        "is_fraud": safe_int(
            edge_data.get("is_fraud")
        ),

        "typology": safe_string(
            edge_data.get("typology")
        ),
    }


def build_account_node(
    graph,
    account_id,
    account_type,
    central=False,
):
    """
    Create an account node from the real graph.
    """

    account_id = str(account_id)

    if account_id not in graph:
        return {
            "id": f"account:{account_id}",
            "type": account_type,
            "label": account_id,
            "central": central,
            "connections": 0,
        }

    degree = int(
        graph.degree(account_id)
    )

    return {
        "id": f"account:{account_id}",
        "type": account_type,
        "label": account_id,
        "central": central,
        "connections": degree,
    }


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/")
def root():
    return {
        "service": "AegisGraph API",
        "status": "running",
        "version": "1.1.0",
    }


@app.get("/api/health")
def health():
    return {
        "status": "healthy",
        "service": "aegisgraph",
    }


@app.get("/api/model-metrics")
def get_model_metrics():
    """
    Return saved model evaluation metrics without recalculating them.
    """
    return model_metrics()


# ============================================================
# RISK SUMMARY
# ============================================================

@app.get("/api/risk-summary")
def risk_summary():
    df = load_csv(RISK_FILE)

    category_counts = (
        df["risk_category"]
        .value_counts()
        .to_dict()
    )

    return {
        "total_transactions": int(len(df)),

        "risk_categories": {
            str(key): int(value)
            for key, value in category_counts.items()
        },

        "average_risk_score": float(
            df["risk_score"].mean()
        ),

        "maximum_risk_score": float(
            df["risk_score"].max()
        ),
    }


# ============================================================
# ALERTS
# ============================================================

@app.get("/api/alerts")
def get_alerts():
    df = load_csv(ALERTS_FILE)

    return {
        "total_alerts": int(len(df)),
        "alerts": dataframe_to_records(df),
    }


@app.get("/api/alerts/{alert_id}")
def get_alert(alert_id: str):
    df = load_csv(ALERTS_FILE)

    matches = df[
        df["alert_id"].astype(str)
        == str(alert_id)
    ]

    if matches.empty:
        raise HTTPException(
            status_code=404,
            detail=(
                f"Alert not found: {alert_id}"
            ),
        )

    return dataframe_to_records(matches)[0]


# ============================================================
# GRAPH INVESTIGATION
# ============================================================

@app.get("/api/graph/{alert_id}")
def get_investigation_graph(
    alert_id: str,
):
    """
    Return the real transaction neighborhood
    surrounding an investigation alert.

    The graph is derived from the existing
    transaction_graph.graphml file.

    No synthetic graph nodes are created.
    """

    # --------------------------------------------------------
    # Find alert
    # --------------------------------------------------------

    alerts_df = load_csv(ALERTS_FILE)

    alert_matches = alerts_df[
        alerts_df["alert_id"].astype(str)
        == str(alert_id)
    ]

    if alert_matches.empty:
        raise HTTPException(
            status_code=404,
            detail=(
                f"Alert not found: {alert_id}"
            ),
        )

    alert = dataframe_to_records(
        alert_matches
    )[0]

    transaction_id = safe_string(
        alert.get("transaction_id")
    )

    if not transaction_id:
        raise HTTPException(
            status_code=400,
            detail=(
                "Alert does not contain a "
                "transaction_id."
            ),
        )

    # --------------------------------------------------------
    # Load graph
    # --------------------------------------------------------

    graph = load_transaction_graph()

    # --------------------------------------------------------
    # Find investigated transaction
    # --------------------------------------------------------

    investigated_edge = find_transaction_edge(
        graph,
        transaction_id,
    )

    if investigated_edge is None:
        raise HTTPException(
            status_code=404,
            detail=(
                "Transaction was not found in "
                f"the graph: {transaction_id}"
            ),
        )

    investigated_sender = (
        investigated_edge["source"]
    )

    investigated_receiver = (
        investigated_edge["target"]
    )

    investigated_data = (
        investigated_edge["data"]
    )

    # --------------------------------------------------------
    # Build account set
    # --------------------------------------------------------

    central_accounts = {
        investigated_sender,
        investigated_receiver,
    }

    related_transaction_edges = []

    seen_transaction_ids = set()

    # --------------------------------------------------------
    # Collect actual transactions involving
    # either central account.
    #
    # This is a one-hop investigation neighborhood.
    # --------------------------------------------------------

    for account_id in central_accounts:

        if account_id not in graph:
            continue

        if graph.is_multigraph():

            for source, target, key, data in graph.edges(
                account_id,
                keys=True,
                data=True,
            ):

                tx_id = edge_transaction_id(
                    data
                )

                if not tx_id:
                    continue

                if tx_id in seen_transaction_ids:
                    continue

                seen_transaction_ids.add(
                    tx_id
                )

                related_transaction_edges.append(
                    {
                        "source": str(source),
                        "target": str(target),
                        "key": key,
                        "data": data,
                    }
                )

        else:

            for source, target, data in graph.edges(
                account_id,
                data=True,
            ):

                tx_id = edge_transaction_id(
                    data
                )

                if not tx_id:
                    continue

                if tx_id in seen_transaction_ids:
                    continue

                seen_transaction_ids.add(
                    tx_id
                )

                related_transaction_edges.append(
                    {
                        "source": str(source),
                        "target": str(target),
                        "key": None,
                        "data": data,
                    }
                )

    # --------------------------------------------------------
    # Make sure investigated transaction is included
    # --------------------------------------------------------

    if transaction_id not in seen_transaction_ids:

        related_transaction_edges.append(
            investigated_edge
        )

        seen_transaction_ids.add(
            transaction_id
        )

    # --------------------------------------------------------
    # Sort transactions
    #
    # Investigated transaction first.
    # Remaining transactions follow chronologically
    # when timestamps are available.
    # --------------------------------------------------------

    def transaction_sort_key(edge):
        tx_id = edge_transaction_id(
            edge["data"]
        )

        if tx_id == transaction_id:
            return (
                0,
                "",
            )

        timestamp = safe_string(
            edge["data"].get("timestamp"),
            "",
        )

        return (
            1,
            timestamp,
        )

    related_transaction_edges.sort(
        key=transaction_sort_key
    )

    # --------------------------------------------------------
    # Limit the displayed neighborhood.
    #
    # We do not fabricate data. We only limit the
    # amount of real graph data sent to the browser.
    # --------------------------------------------------------

    MAX_TRANSACTIONS = 60

    displayed_edges = (
        related_transaction_edges[
            :MAX_TRANSACTIONS
        ]
    )

    # --------------------------------------------------------
    # Build nodes
    # --------------------------------------------------------

    nodes = {}
    links = []

    # Central account nodes
    nodes[
        f"account:{investigated_sender}"
    ] = build_account_node(
        graph,
        investigated_sender,
        "sender",
        central=True,
    )

    nodes[
        f"account:{investigated_receiver}"
    ] = build_account_node(
        graph,
        investigated_receiver,
        "receiver",
        central=True,
    )

    # --------------------------------------------------------
    # Build transaction nodes and account nodes
    # --------------------------------------------------------

    for edge in displayed_edges:

        source = str(edge["source"])
        target = str(edge["target"])
        data = edge["data"]

        tx_id = edge_transaction_id(
            data
        )

        if not tx_id:
            continue

        is_investigated = (
            tx_id == transaction_id
        )

        # --------------------------------------------
        # Determine account type
        # --------------------------------------------

        if source == investigated_sender:
            source_type = "sender"

        elif source == investigated_receiver:
            source_type = "receiver"

        else:
            source_type = "account"

        if target == investigated_receiver:
            target_type = "receiver"

        elif target == investigated_sender:
            target_type = "sender"

        else:
            target_type = "account"

        # --------------------------------------------
        # Add source account
        # --------------------------------------------

        source_node_id = (
            f"account:{source}"
        )

        if source_node_id not in nodes:

            nodes[source_node_id] = (
                build_account_node(
                    graph,
                    source,
                    source_type,
                    central=False,
                )
            )

        # --------------------------------------------
        # Add target account
        # --------------------------------------------

        target_node_id = (
            f"account:{target}"
        )

        if target_node_id not in nodes:

            nodes[target_node_id] = (
                build_account_node(
                    graph,
                    target,
                    target_type,
                    central=False,
                )
            )

        # --------------------------------------------
        # Transaction node
        # --------------------------------------------

        transaction_node_id = (
            f"transaction:{tx_id}"
        )

        transaction_record = (
            graph_edge_record(
                source,
                target,
                data,
            )
        )

        nodes[
            transaction_node_id
        ] = {
            "id": transaction_node_id,
            "type": "transaction",
            "label": tx_id,
            "transaction_id": tx_id,
            "central": is_investigated,
            "amount": transaction_record[
                "amount"
            ],
            "timestamp": transaction_record[
                "timestamp"
            ],
            "currency": transaction_record[
                "currency"
            ],
            "transaction_type": (
                transaction_record[
                    "transaction_type"
                ]
            ),
            "channel": transaction_record[
                "channel"
            ],
            "sender_account": source,
            "receiver_account": target,
            "sender_country": (
                transaction_record[
                    "sender_country"
                ]
            ),
            "receiver_country": (
                transaction_record[
                    "receiver_country"
                ]
            ),
            "status": transaction_record[
                "status"
            ],
            "is_fraud": transaction_record[
                "is_fraud"
            ],
            "typology": transaction_record[
                "typology"
            ],
        }

        # --------------------------------------------
        # Account -> transaction
        # --------------------------------------------

        links.append(
            {
                "id": (
                    f"{source}"
                    f"->{tx_id}"
                ),
                "source": source_node_id,
                "target": transaction_node_id,
                "type": "transaction",
                "direction": "outgoing",
                "transaction_id": tx_id,
                "amount": transaction_record[
                    "amount"
                ],
                "currency": transaction_record[
                    "currency"
                ],
                "timestamp": transaction_record[
                    "timestamp"
                ],
                "central": is_investigated,
            }
        )

        # --------------------------------------------
        # Transaction -> account
        # --------------------------------------------

        links.append(
            {
                "id": (
                    f"{tx_id}"
                    f"->{target}"
                ),
                "source": transaction_node_id,
                "target": target_node_id,
                "type": "transaction",
                "direction": "incoming",
                "transaction_id": tx_id,
                "amount": transaction_record[
                    "amount"
                ],
                "currency": transaction_record[
                    "currency"
                ],
                "timestamp": transaction_record[
                    "timestamp"
                ],
                "central": is_investigated,
            }
        )

    # --------------------------------------------------------
    # Determine visible transaction count
    # --------------------------------------------------------

    visible_transaction_count = len(
        [
            node
            for node in nodes.values()
            if node["type"] == "transaction"
        ]
    )

    # --------------------------------------------------------
    # Return investigation graph
    # --------------------------------------------------------

    return {
        "alert_id": alert_id,

        "investigated_transaction": {
            "transaction_id": transaction_id,
            "sender_account": (
                investigated_sender
            ),
            "receiver_account": (
                investigated_receiver
            ),
            "amount": safe_float(
                investigated_data.get(
                    "amount"
                )
            ),
            "timestamp": safe_string(
                investigated_data.get(
                    "timestamp"
                )
            ),
        },

        "network": {
            "nodes": list(nodes.values()),
            "links": links,

            "total_nodes": int(
                len(nodes)
            ),

            "total_links": int(
                len(links)
            ),

            "transaction_count": int(
                visible_transaction_count
            ),

            "truncated": (
                len(related_transaction_edges)
                > MAX_TRANSACTIONS
            ),

            "source": (
                "transaction_graph.graphml"
            ),
        },

        "alert": alert,
    }


# ============================================================
# INVESTIGATION CASES
# ============================================================

@app.get("/api/cases")
def get_cases():
    df = load_csv(CASES_FILE)

    return {
        "total_cases": int(len(df)),
        "cases": dataframe_to_records(df),
    }


@app.get("/api/cases/{case_id}")
def get_case(case_id: str):
    df = load_csv(DOSSIERS_FILE)

    matches = df[
        df["case_id"].astype(str)
        == str(case_id)
    ]

    if matches.empty:
        raise HTTPException(
            status_code=404,
            detail=(
                f"Case not found: {case_id}"
            ),
        )

    return dataframe_to_records(matches)[0]


# ============================================================
# CASE REVIEWS
# ============================================================

@app.get("/api/reviews")
def get_reviews():
    df = load_csv(REVIEWS_FILE)

    return {
        "total_reviews": int(len(df)),
        "reviews": dataframe_to_records(df),
    }


# ============================================================
# INVESTIGATION DOSSIERS
# ============================================================

@app.get("/api/dossiers")
def get_dossiers():
    df = load_csv(DOSSIERS_FILE)

    return {
        "total_dossiers": int(len(df)),
        "dossiers": dataframe_to_records(df),
    }


# ============================================================
# PROJECT SUMMARY
# ============================================================

@app.get("/api/summary")
def project_summary():
    risk_df = load_csv(RISK_FILE)
    alerts_df = load_csv(ALERTS_FILE)
    cases_df = load_csv(CASES_FILE)
    dossiers_df = load_csv(DOSSIERS_FILE)
    transactions = load_transaction_dataframe()

    risk_counts = (
        risk_df["risk_category"]
        .value_counts()
        .to_dict()
    )

    priority_counts = (
        cases_df["case_priority"]
        .value_counts()
        .to_dict()
    )
    activity_pattern_counts = get_activity_pattern_counts(
        transactions
    )

    return {
        "dataset": {
            "total_transactions": int(
                transactions["transaction_id"].nunique()
            ),
        },

        "risk": {
            "total_transactions": int(
                len(risk_df)
            ),

            "categories": {
                str(key): int(value)
                for key, value
                in risk_counts.items()
            },
        },

        "activity_patterns": activity_pattern_counts,

        "alerts": {
            "total": int(
                len(alerts_df)
            ),
        },

        "investigation": {
            "total_cases": int(
                len(cases_df)
            ),

            "total_dossiers": int(
                len(dossiers_df)
            ),

            "priority": {
                str(key): int(value)
                for key, value
                in priority_counts.items()
            },
        },
    }

# ============================================================
# TRANSACTIONS
# ============================================================

@app.get("/api/transactions")
def get_transactions(
    page: int = Query(
        1,
        ge=1,
        description="Page number",
    ),
    page_size: int = Query(
        25,
        ge=1,
        le=100,
        description="Number of transactions per page",
    ),
    search: str = Query(
        "",
        description="Search transaction ID or account ID",
    ),
    typology: str = Query(
        "",
        description="Filter by transaction typology",
    ),
    status: str = Query(
        "",
        description="Filter by transaction status",
    ),
    fraud_only: bool = Query(
        False,
        description="Return only transactions marked as fraud",
    ),
):
    """
    Return paginated transaction records from the
    cleaned transaction dataset.
    """

    df = load_transaction_dataframe()

    # Attach model predictions where available.
    risk_df = load_csv(RISK_FILE)

    risk_columns = [
        "transaction_id",
        "rf_probability",
        "tgn_probability",
        "thgnn_probability",
        "rule_score",
        "risk_score",
        "risk_category",
    ]

    if all(col in risk_df.columns for col in risk_columns):
        risk_df = risk_df[risk_columns].drop_duplicates(
            subset=["transaction_id"]
        )

        df = df.merge(
            risk_df,
            on="transaction_id",
            how="left",
            validate="one_to_one",
        )

    filtered = df.copy()

    # --------------------------------------------------------
    # SEARCH
    # --------------------------------------------------------

    search_value = search.strip().lower()

    if search_value:
        searchable_columns = [
            "transaction_id",
            "sender_account",
            "receiver_account",
        ]

        mask = False

        for column in searchable_columns:
            if column in filtered.columns:
                mask = (
                    mask
                    | filtered[column]
                    .astype(str)
                    .str.lower()
                    .str.contains(
                        search_value,
                        regex=False,
                        na=False,
                    )
                )

        filtered = filtered[mask]

    # --------------------------------------------------------
    # TYPOLOGY FILTER
    # --------------------------------------------------------

    typology_value = typology.strip().lower()

    if typology_value and "typology" in filtered.columns:
        filtered = filtered[
            filtered["typology"]
            .astype(str)
            .str.lower()
            == typology_value
        ]

    # --------------------------------------------------------
    # STATUS FILTER
    # --------------------------------------------------------

    status_value = status.strip().lower()

    if status_value and "status" in filtered.columns:
        filtered = filtered[
            filtered["status"]
            .astype(str)
            .str.lower()
            == status_value
        ]

    # --------------------------------------------------------
    # FRAUD FILTER
    # --------------------------------------------------------

    if fraud_only and "is_fraud" in filtered.columns:
        filtered = filtered[
            filtered["is_fraud"].astype(int) == 1
        ]

    # --------------------------------------------------------
    # SORT
    # --------------------------------------------------------

    if "timestamp" in filtered.columns:
        filtered = filtered.sort_values(
            "timestamp",
            ascending=False,
        )

    # --------------------------------------------------------
    # PAGINATION
    # --------------------------------------------------------

    total = len(filtered)

    start = (page - 1) * page_size
    end = start + page_size

    page_data = filtered.iloc[start:end]

    records = dataframe_to_records(page_data)

    total_pages = (
        (total + page_size - 1) // page_size
        if total
        else 0
    )

    return {
        "transactions": records,
        "pagination": {
            "page": page,
            "page_size": page_size,
            "total": total,
            "total_pages": total_pages,
        },
        "filters": {
            "search": search,
            "typology": typology,
            "status": status,
            "fraud_only": fraud_only,
        },
    }