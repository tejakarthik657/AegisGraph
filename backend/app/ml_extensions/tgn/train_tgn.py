import sys
from pathlib import Path
from app.ml_extensions.tgn.message import MessageEncoder
from app.ml_extensions.tgn.memory import TemporalMemory
from app.ml_extensions.tgn.time_encoding import TimeEncoder
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler


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
    RESULTS_DATA_DIR,
    create_directories,
)

from app.ml_extensions.tgn.tgn_model import (
    TGNFraudModel,
)


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------
RANDOM_SEED = 42

TRAIN_RATIO = 0.80

HIDDEN_DIM = 64

LEARNING_RATE = 0.001

EPOCHS = 50

WEIGHT_DECAY = 1e-4


# ---------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# ---------------------------------------------------------
# Device
# ---------------------------------------------------------
DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ---------------------------------------------------------
# Load transactions
# ---------------------------------------------------------
def load_transactions():

    print(
        "Loading transaction data..."
    )

    df = pd.read_csv(
        CLEAN_TRANSACTIONS_FILE
    )

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        errors="coerce",
    )

    df = (
        df.sort_values("timestamp")
        .reset_index(drop=True)
    )

    print(
        f"Loaded {len(df)} transactions."
    )

    return df


# ---------------------------------------------------------
# Create account mapping
# ---------------------------------------------------------
def create_account_mapping(df):

    accounts = sorted(
        set(
            df["sender_account"]
        )
        |
        set(
            df["receiver_account"]
        )
    )

    account_to_index = {
        account: index
        for index, account in enumerate(accounts)
    }

    return account_to_index


# ---------------------------------------------------------
# Create transaction features
# ---------------------------------------------------------
def create_transaction_features(df):

    features = pd.DataFrame(index=df.index)

    # -----------------------------------------------------
    # Amount
    # -----------------------------------------------------

    features["amount"] = pd.to_numeric(
        df["amount"],
        errors="coerce",
    )

    features["log_amount"] = np.log1p(
        features["amount"].clip(lower=0)
    )

    # -----------------------------------------------------
    # Balance features
    # -----------------------------------------------------

    balance_columns = [
        "sender_balance_before",
        "sender_balance_after",
        "receiver_balance_before",
        "receiver_balance_after",
    ]

    for column in balance_columns:

        features[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    # -----------------------------------------------------
    # Amount/balance ratios
    # -----------------------------------------------------

    features[
        "amount_to_sender_balance"
    ] = (
        features["amount"]
        /
        features[
            "sender_balance_before"
        ].replace(0, np.nan)
    )

    features[
        "amount_to_receiver_balance"
    ] = (
        features["amount"]
        /
        features[
            "receiver_balance_before"
        ].replace(0, np.nan)
    )

    # -----------------------------------------------------
    # Time features
    # -----------------------------------------------------

    features["hour"] = (
        df["timestamp"].dt.hour
    )

    features["day_of_week"] = (
        df["timestamp"].dt.dayofweek
    )

    features["day_of_month"] = (
        df["timestamp"].dt.day
    )

    features["month"] = (
        df["timestamp"].dt.month
    )

    features["is_weekend"] = (
        df["timestamp"].dt.dayofweek >= 5
    ).astype(int)

    # -----------------------------------------------------
    # Transaction type
    # -----------------------------------------------------

    categorical = pd.get_dummies(
        df[
            [
                "transaction_type",
                "channel",
                "currency",
            ]
        ],
        dtype=float,
    )

    features = pd.concat(
        [
            features,
            categorical,
        ],  
        axis=1,
    )

    # -----------------------------------------------------
    # Cleanup
    # -----------------------------------------------------

    features = (
        features
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
        .fillna(0)
    )

    return features


# ---------------------------------------------------------
# Create temporal edge graph
# ---------------------------------------------------------
def create_graph(df, account_to_index):

    print(
        "Building temporal transaction graph..."
    )

    source_nodes = (
        df["sender_account"]
        .map(account_to_index)
        .astype(int)
        .to_numpy()
    )

    target_nodes = (
        df["receiver_account"]
        .map(account_to_index)
        .astype(int)
        .to_numpy()
    )

    edge_index = torch.tensor(
        np.vstack(
            [
                source_nodes,
                target_nodes,
            ]
        ),
        dtype=torch.long,
    )

    return edge_index


# ---------------------------------------------------------
# Create temporal values
# ---------------------------------------------------------
# ---------------------------------------------------------
# Create account-level temporal values
# ---------------------------------------------------------
def create_temporal_features(
    df,
    account_to_index,
):

    number_of_accounts = len(
        account_to_index
    )

    account_timestamps = np.zeros(
        number_of_accounts,
        dtype=np.float32,
    )

    # Normalize transaction timestamps
    timestamps = (
        df["timestamp"]
        .astype("int64")
        .to_numpy()
    )

    min_timestamp = timestamps.min()
    max_timestamp = timestamps.max()

    if max_timestamp > min_timestamp:
        normalized_timestamps = (
            timestamps - min_timestamp
        ) / (
            max_timestamp - min_timestamp
        )
    else:
        normalized_timestamps = np.zeros(
            len(timestamps),
            dtype=np.float32,
        )

    # Use the latest transaction timestamp
    # observed for each account.
    for index, row in df.iterrows():

        timestamp_value = float(
            normalized_timestamps[index]
        )

        sender = account_to_index[
            row["sender_account"]
        ]

        receiver = account_to_index[
            row["receiver_account"]
        ]

        account_timestamps[sender] = max(
            account_timestamps[sender],
            timestamp_value,
        )

        account_timestamps[receiver] = max(
            account_timestamps[receiver],
            timestamp_value,
        )

    account_timestamps = (
        account_timestamps.reshape(-1, 1)
    )

    return torch.tensor(
        account_timestamps,
        dtype=torch.float32,
    )


# ---------------------------------------------------------
# Build graph node representation
# ---------------------------------------------------------
def build_node_features(
    transaction_features,
    df,
    account_to_index,
):
    """
    Aggregate transaction information into account-level
    node features.

    Each graph node represents a bank account.

    Transaction features are aggregated over the
    transactions involving that account.
    """

    number_of_accounts = len(
        account_to_index
    )

    feature_array = (
        transaction_features
        .to_numpy(
            dtype=np.float32
        )
    )

    feature_dimension = (
        feature_array.shape[1]
    )

    node_features = np.zeros(
        (
            number_of_accounts,
            feature_dimension,
        ),
        dtype=np.float32,
    )

    node_counts = np.zeros(
        number_of_accounts,
        dtype=np.float32,
    )

    # -----------------------------------------------------
    # Aggregate transaction information
    # -----------------------------------------------------

    for index, row in df.iterrows():

        sender = account_to_index[
            row["sender_account"]
        ]

        receiver = account_to_index[
            row["receiver_account"]
        ]

        values = feature_array[index]

        node_features[sender] += values
        node_features[receiver] += values

        node_counts[sender] += 1
        node_counts[receiver] += 1

    # -----------------------------------------------------
    # Convert sums to averages
    # -----------------------------------------------------

    node_counts[
        node_counts == 0
    ] = 1

    node_features = (
        node_features
        /
        node_counts[:, None]
    )

    return node_features


# ---------------------------------------------------------
# Build transaction labels on nodes
# ---------------------------------------------------------
def build_node_labels(
    df,
    account_to_index,
):
    """
    Assign an account-level fraud label.

    An account is considered fraud-positive if it
    participates in at least one fraudulent transaction.

    This is used only for the initial graph-level
    training experiment.
    """

    labels = np.zeros(
        len(account_to_index),
        dtype=np.float32,
    )

    for _, row in df.iterrows():

        if int(row["is_fraud"]) == 1:

            sender = account_to_index[
                row["sender_account"]
            ]

            receiver = account_to_index[
                row["receiver_account"]
            ]

            labels[sender] = 1
            labels[receiver] = 1

    return labels


# ---------------------------------------------------------
# Main training function
# ---------------------------------------------------------
def train_tgn():

    create_directories()

    print(
        "\n========================================"
    )

    print(
        "       AEGISGRAPH TEMPORAL GRAPH MODEL"
    )

    print(
        "========================================"
    )

    print(
        f"\nUsing device: {DEVICE}"
    )

    # -----------------------------------------------------
    # Load data
    # -----------------------------------------------------

    df = load_transactions()

    # -----------------------------------------------------
    # Chronological split
    # -----------------------------------------------------

    split_index = int(
        len(df) * TRAIN_RATIO
    )

    train_df = df.iloc[
        :split_index
    ].copy()

    test_df = df.iloc[
        split_index:
    ].copy()

    print(
        "\n========== DATA PREPARATION =========="
    )

    print(
        f"Training transactions: "
        f"{len(train_df)}"
    )

    print(
        f"Testing transactions: "
        f"{len(test_df)}"
    )

    print(
        f"Training fraud count: "
        f"{train_df['is_fraud'].sum()}"
    )

    print(
        f"Testing fraud count: "
        f"{test_df['is_fraud'].sum()}"
    )

    # -----------------------------------------------------
    # Account mapping
    # -----------------------------------------------------

    account_to_index = (
        create_account_mapping(df)
    )

    print(
        f"Graph accounts: "
        f"{len(account_to_index)}"
    )

    # -----------------------------------------------------
    # Transaction features
    # -----------------------------------------------------

    transaction_features = (
        create_transaction_features(df)
    )

    # -----------------------------------------------------
    # Scale features
    # -----------------------------------------------------

    scaler = StandardScaler()

    train_features = (
        transaction_features.iloc[
            :split_index
        ]
    )

    scaler.fit(train_features)

    scaled_features = scaler.transform(
        transaction_features
    )

    scaled_features = pd.DataFrame(
        scaled_features,
        columns=transaction_features.columns,
    )

    # -----------------------------------------------------
    # Node features
    # -----------------------------------------------------

    node_features = build_node_features(
        scaled_features,
        df,
        account_to_index,
    )

    # -----------------------------------------------------
    # Node labels
    # -----------------------------------------------------

    node_labels = build_node_labels(
        df,
        account_to_index,
    )

    # -----------------------------------------------------
    # Graph
    # -----------------------------------------------------

    edge_index = create_graph(
        df,
        account_to_index,
    )

    # -----------------------------------------------------
    # Temporal values
    # -----------------------------------------------------

    timestamps = create_temporal_features(
        df,
        account_to_index,
    )

    # -----------------------------------------------------
    # Convert to tensors
    # -----------------------------------------------------

    x = torch.tensor(
        node_features,
        dtype=torch.float32,
    )

    y = torch.tensor(
        node_labels,
        dtype=torch.float32,
    )

    x = x.to(DEVICE)
    y = y.to(DEVICE)

    edge_index = edge_index.to(
        DEVICE
    )

    timestamps = timestamps.to(
        DEVICE
    )

    # -----------------------------------------------------
    # Model
    # -----------------------------------------------------

    model = TGNFraudModel(
        input_dim=x.shape[1],
        hidden_dim=HIDDEN_DIM,
    ).to(DEVICE)

    # -----------------------------------------------------
    # Class weighting
    # -----------------------------------------------------

    positive_count = y.sum().item()

    negative_count = (
        len(y) - positive_count
    )

    if positive_count > 0:

        pos_weight = (
            negative_count
            /
            positive_count
        )

    else:

        pos_weight = 1.0

    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(
            pos_weight,
            device=DEVICE,
        )
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    # -----------------------------------------------------
    # Training
    # -----------------------------------------------------

    print(
        "\n========== TGN TRAINING =========="
    )

    model.train()

    for epoch in range(
        1,
        EPOCHS + 1,
    ):

        optimizer.zero_grad()

        logits = model(
            x,
            edge_index,
            timestamps,
        )

        loss = criterion(
            logits,
            y,
        )

        loss.backward()

        optimizer.step()

        if (
            epoch == 1
            or epoch % 5 == 0
            or epoch == EPOCHS
        ):

            print(
                f"Epoch "
                f"{epoch:03d}/{EPOCHS} "
                f"| Loss: "
                f"{loss.item():.4f}"
            )

    print(
        "\nTGN training complete."
    )

    # -----------------------------------------------------
    # Evaluation
    # -----------------------------------------------------

    print(
        "\n========== TGN EVALUATION =========="
    )

    model.eval()

    with torch.no_grad():

        logits = model(
            x,
            edge_index,
            timestamps,
        )

        probabilities = (
            torch.sigmoid(
                logits
            )
            .cpu()
            .numpy()
        )

    # -----------------------------------------------------
    # IMPORTANT
    # -----------------------------------------------------
    # This first version produces account-level
    # predictions. Therefore we evaluate the account
    # labels rather than pretending these are direct
    # transaction-level predictions.
    # -----------------------------------------------------

    predictions = (
        probabilities >= 0.5
    ).astype(int)

    y_true = (
        y.cpu()
        .numpy()
        .astype(int)
    )

    precision = precision_score(
        y_true,
        predictions,
        zero_division=0,
    )

    recall = recall_score(
        y_true,
        predictions,
        zero_division=0,
    )

    f1 = f1_score(
        y_true,
        predictions,
        zero_division=0,
    )

    try:

        roc_auc = roc_auc_score(
            y_true,
            probabilities,
        )

    except ValueError:

        roc_auc = float("nan")

    try:

        pr_auc = average_precision_score(
            y_true,
            probabilities,
        )

    except ValueError:

        pr_auc = float("nan")

    print(
        f"\nPrecision : {precision:.4f}"
    )

    print(
        f"Recall    : {recall:.4f}"
    )

    print(
        f"F1 Score  : {f1:.4f}"
    )

    print(
        f"ROC-AUC   : {roc_auc:.4f}"
    )

    print(
        f"PR-AUC    : {pr_auc:.4f}"
    )

    print(
        "\nClassification Report:"
    )

    print(
        classification_report(
            y_true,
            predictions,
            zero_division=0,
        )
    )

    print(
        "Confusion Matrix:"
    )

    print(
        confusion_matrix(
            y_true,
            predictions,
        )
    )

    # -----------------------------------------------------
    # Save account predictions
    # -----------------------------------------------------

    index_to_account = {
        index: account
        for account, index
        in account_to_index.items()
    }

    prediction_df = pd.DataFrame(
        {
            "account_id": [
                index_to_account[i]
                for i in range(
                    len(index_to_account)
                )
            ],
            "is_fraud": y_true,
            "fraud_probability": probabilities,
            "prediction": predictions,
        }
    )

    prediction_file = (
        RESULTS_DATA_DIR
        / "tgn_predictions.csv"
    )

    prediction_df.to_csv(
        prediction_file,
        index=False,
    )

    # -----------------------------------------------------
    # Save metrics
    # -----------------------------------------------------

    metrics_df = pd.DataFrame(
        [
            {
                "model": "TGN",
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "roc_auc": roc_auc,
                "pr_auc": pr_auc,
            }
        ]
    )

    metrics_file = (
        RESULTS_DATA_DIR
        / "tgn_metrics.csv"
    )

    metrics_df.to_csv(
        metrics_file,
        index=False,
    )

    # -----------------------------------------------------
    # Save model
    # -----------------------------------------------------

    model_file = (
        RESULTS_DATA_DIR
        / "tgn_model.pt"
    )

    torch.save(
        model.state_dict(),
        model_file,
    )

    print(
        "\nPredictions saved to:"
        f"\n{prediction_file}"
    )

    print(
        "\nMetrics saved to:"
        f"\n{metrics_file}"
    )

    print(
        "\nModel saved to:"
        f"\n{model_file}"
    )

    print(
        "\n========================================"
    )

    print(
        "       TGN TRAINING COMPLETE"
    )

    print(
        "========================================"
    )


# ---------------------------------------------------------
# Script entry point
# ---------------------------------------------------------
if __name__ == "__main__":
    train_tgn()