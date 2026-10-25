import sys
from pathlib import Path
import joblib
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
from torch_geometric.nn import GATConv
import torch.nn.functional as F


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

# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------
RANDOM_SEED = 42

TRAIN_RATIO = 0.80
FORBIDDEN_FEATURE_COLUMNS = {
    "is_fraud",
    "typology",
    "sender_balance_after",
    "receiver_balance_after",
}

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

    if df["timestamp"].isna().any():
        raise ValueError("Invalid transaction timestamps detected.")

    df = (
        df.sort_values("timestamp", kind="mergesort")
        .reset_index(drop=True)
    )

    print(
        f"Loaded {len(df)} transactions."
    )

    return df


def chronological_train_test_split(
    df: pd.DataFrame,
    train_ratio: float = TRAIN_RATIO,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp]:
    """Split chronologically without dividing equal-timestamp batches."""

    if not 0 < train_ratio < 1:
        raise ValueError("train_ratio must be between 0 and 1.")
    if "timestamp" not in df.columns:
        raise ValueError("Transactions require a timestamp column.")

    ordered = df.copy()
    ordered["timestamp"] = pd.to_datetime(
        ordered["timestamp"],
        errors="raise",
    )
    ordered = ordered.sort_values(
        "timestamp",
        kind="mergesort",
    ).reset_index(drop=True)

    split_index = int(len(ordered) * train_ratio)
    if split_index <= 0 or split_index >= len(ordered):
        raise ValueError("Chronological split must produce train and test rows.")

    boundary_timestamp = ordered.loc[split_index, "timestamp"]
    while (
        split_index < len(ordered)
        and ordered.loc[split_index - 1, "timestamp"] == boundary_timestamp
    ):
        split_index -= 1

    # Move the boundary after the whole tied timestamp group, keeping
    # all same-time events together and the split near the requested ratio.
    if split_index == 0:
        split_index = int(len(ordered) * train_ratio)
        while (
            split_index < len(ordered)
            and ordered.loc[split_index, "timestamp"] == boundary_timestamp
        ):
            split_index += 1

    if split_index <= 0 or split_index >= len(ordered):
        raise ValueError(
            "Unable to form chronological train/test partitions "
            "without splitting an equal-timestamp group."
        )

    train = ordered.iloc[:split_index].copy().reset_index(drop=True)
    test = ordered.iloc[split_index:].copy().reset_index(drop=True)
    prediction_timestamp = test["timestamp"].min()
    if train["timestamp"].max() >= prediction_timestamp:
        raise ValueError("Train/test split is not strictly chronological.")
    return train, test, prediction_timestamp


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
        "receiver_balance_before",
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


def fit_transaction_preprocessor(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> tuple[StandardScaler, list[str], pd.DataFrame, pd.DataFrame]:
    """Fit categorical schema and scaling on training transactions only."""

    train_features = create_transaction_features(train_df)
    test_features = create_transaction_features(test_df)
    feature_columns = list(train_features.columns)

    forbidden = FORBIDDEN_FEATURE_COLUMNS.intersection(feature_columns)
    if forbidden:
        raise ValueError(
            f"Forbidden account-TGN model features: {sorted(forbidden)}"
        )

    test_features = test_features.reindex(
        columns=feature_columns,
        fill_value=0.0,
    )

    scaler = StandardScaler()
    scaler.fit(train_features)

    scaled_train = pd.DataFrame(
        scaler.transform(train_features),
        columns=feature_columns,
        index=train_df.index,
    )
    scaled_test = pd.DataFrame(
        scaler.transform(test_features),
        columns=feature_columns,
        index=test_df.index,
    )
    return scaler, feature_columns, scaled_train, scaled_test


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

    # Normalize timestamps using the supplied history only.
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
    prediction_timestamp=None,
):
    """
    Aggregate transaction information into account-level
    node features.

    Each graph node represents a bank account.

    Transaction features are aggregated over the
    transactions involving that account.
    """

    if prediction_timestamp is not None:
        prediction_timestamp = pd.Timestamp(prediction_timestamp)
        eligible = df["timestamp"] < prediction_timestamp
        df = df.loc[eligible].copy()
        transaction_features = transaction_features.loc[
            eligible
        ].copy()
        df = df.reset_index(drop=True)
        transaction_features = transaction_features.reset_index(drop=True)

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

    Labels must be built from the training partition only.
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


def build_test_node_labels(
    test_df: pd.DataFrame,
    account_to_index: dict,
) -> tuple[np.ndarray, np.ndarray]:
    """Build held-out labels and node indices for test-active known accounts."""

    labels_by_account = np.zeros(
        len(account_to_index),
        dtype=np.float32,
    )
    active_nodes = set()

    for _, row in test_df.iterrows():
        for account in (
            row["sender_account"],
            row["receiver_account"],
        ):
            node = account_to_index.get(account)
            if node is not None:
                active_nodes.add(node)
                if int(row["is_fraud"]) == 1:
                    labels_by_account[node] = 1.0

    node_indices = np.asarray(sorted(active_nodes), dtype=np.int64)
    return node_indices, labels_by_account[node_indices]


# ---------------------------------------------------------
# Account-level graph classifier
# ---------------------------------------------------------
class AccountTGNClassifier(nn.Module):
    """Graph-attention classifier for a train-period account snapshot."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = HIDDEN_DIM,
    ):
        super().__init__()
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        self.time_encoder = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.gat1 = GATConv(
            hidden_dim,
            hidden_dim,
            heads=2,
            concat=False,
            dropout=0.2,
        )
        self.gat2 = GATConv(
            hidden_dim,
            hidden_dim,
            heads=2,
            concat=False,
            dropout=0.2,
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x, edge_index, timestamps):
        x = self.input_projection(x) + self.time_encoder(timestamps)
        x = F.relu(self.gat1(x, edge_index))
        x = F.relu(self.gat2(x, edge_index))
        return self.classifier(x).squeeze(-1)


# ---------------------------------------------------------
# Main training function
# ---------------------------------------------------------
def _legacy_train_tgn():

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


def train_tgn():
    """Train the legacy account-level model and evaluate held-out accounts."""

    create_directories()
    df = load_transactions()
    train_df, test_df, prediction_timestamp = (
        chronological_train_test_split(df)
    )

    print(
        "\n========== CHRONOLOGICAL ACCOUNT-TGN SPLIT =========="
    )
    print(
        f"Training transactions: {len(train_df)} "
        f"({train_df['timestamp'].min()} to {train_df['timestamp'].max()})"
    )
    print(
        f"Testing transactions: {len(test_df)} "
        f"({test_df['timestamp'].min()} to {test_df['timestamp'].max()})"
    )
    print(f"Prediction snapshot: {prediction_timestamp}")

    # All learned graph structure, account IDs, and node features come
    # from the training period. Test labels are created separately below.
    account_to_index = create_account_mapping(train_df)
    scaler, feature_columns, scaled_train, _ = (
        fit_transaction_preprocessor(train_df, test_df)
    )
    node_features_array = build_node_features(
        scaled_train,
        train_df,
        account_to_index,
        prediction_timestamp=prediction_timestamp,
    )
    train_labels_array = build_node_labels(
        train_df,
        account_to_index,
    )
    edge_index = create_graph(train_df, account_to_index)
    account_timestamps = create_temporal_features(
        train_df,
        account_to_index,
    )

    x = torch.tensor(node_features_array, dtype=torch.float32, device=DEVICE)
    y_train = torch.tensor(
        train_labels_array,
        dtype=torch.float32,
        device=DEVICE,
    )
    edge_index = edge_index.to(DEVICE)
    account_timestamps = account_timestamps.to(DEVICE)

    model = AccountTGNClassifier(
        input_dim=x.shape[1],
        hidden_dim=HIDDEN_DIM,
    ).to(DEVICE)
    positive_count = float(y_train.sum().item())
    negative_count = float(len(y_train) - positive_count)
    pos_weight = (
        negative_count / positive_count
        if positive_count > 0
        else 1.0
    )
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(pos_weight, device=DEVICE)
    )
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    model.train()
    for epoch in range(1, EPOCHS + 1):
        optimizer.zero_grad()
        logits = model(x, edge_index, account_timestamps)
        loss = criterion(logits, y_train)
        loss.backward()
        optimizer.step()
        if epoch == 1 or epoch % 5 == 0 or epoch == EPOCHS:
            print(f"Epoch {epoch:03d}/{EPOCHS} | Loss: {loss.item():.4f}")

    model.eval()
    with torch.no_grad():
        all_probabilities = torch.sigmoid(
            model(x, edge_index, account_timestamps)
        ).cpu().numpy()

    # Test labels are first constructed after fitting and inference;
    # they are used exclusively to evaluate the held-out predictions.
    test_node_indices, test_labels_array = build_test_node_labels(
        test_df,
        account_to_index,
    )
    if len(test_node_indices) == 0:
        raise ValueError(
            "No test-period accounts were observed during training; "
            "account-level evaluation is not possible."
        )

    test_probabilities = all_probabilities[test_node_indices]
    test_predictions = (test_probabilities >= 0.5).astype(int)
    y_true = test_labels_array.astype(int)
    tn, fp, fn, tp = confusion_matrix(
        y_true,
        test_predictions,
        labels=[0, 1],
    ).ravel()
    precision = precision_score(y_true, test_predictions, zero_division=0)
    recall = recall_score(y_true, test_predictions, zero_division=0)
    f1 = f1_score(y_true, test_predictions, zero_division=0)
    if len(np.unique(y_true)) == 2:
        roc_auc = roc_auc_score(y_true, test_probabilities)
        pr_auc = average_precision_score(y_true, test_probabilities)
    else:
        roc_auc = float("nan")
        pr_auc = float("nan")

    print("\n========== HELD-OUT ACCOUNT EVALUATION ==========")
    print(f"Test accounts: {len(y_true)}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1:        {f1:.4f}")
    print(f"ROC-AUC:   {roc_auc:.4f}")
    print(f"PR-AUC:    {pr_auc:.4f}")
    print("Confusion matrix [TN FP; FN TP]:")
    print(np.array([[tn, fp], [fn, tp]]))
    print(
        classification_report(
            y_true,
            test_predictions,
            zero_division=0,
        )
    )

    index_to_account = {
        index: account
        for account, index in account_to_index.items()
    }
    prediction_df = pd.DataFrame(
        {
            "account_id": [
                index_to_account[int(index)]
                for index in test_node_indices
            ],
            "is_fraud": y_true,
            "fraud_probability": test_probabilities,
            "prediction": test_predictions,
        }
    )
    prediction_file = RESULTS_DATA_DIR / "tgn_predictions.csv"
    prediction_df.to_csv(prediction_file, index=False)

    metrics_df = pd.DataFrame(
        [
            {
                "model": "TGN",
                "train_transactions": len(train_df),
                "test_transactions": len(test_df),
                "test_accounts": len(y_true),
                "threshold": 0.5,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "roc_auc": roc_auc,
                "pr_auc": pr_auc,
                "true_negatives": int(tn),
                "false_positives": int(fp),
                "false_negatives": int(fn),
                "true_positives": int(tp),
            }
        ]
    )
    metrics_file = RESULTS_DATA_DIR / "tgn_metrics.csv"
    metrics_df.to_csv(metrics_file, index=False)

    torch.save(
        {
            "state_dict": model.state_dict(),
            "input_dim": len(feature_columns),
            "hidden_dim": HIDDEN_DIM,
            "account_to_index": account_to_index,
        },
        RESULTS_DATA_DIR / "tgn_model.pt",
    )
    joblib.dump(
        {
            "scaler": scaler,
            "feature_columns": feature_columns,
            "train_end_timestamp": train_df["timestamp"].max(),
            "prediction_timestamp": prediction_timestamp,
        },
        RESULTS_DATA_DIR / "account_tgn_preprocessor.joblib",
    )
    print(f"Predictions saved to: {prediction_file}")
    print(f"Metrics saved to: {metrics_file}")


# ---------------------------------------------------------
# Script entry point
# ---------------------------------------------------------
if __name__ == "__main__":
    train_tgn()