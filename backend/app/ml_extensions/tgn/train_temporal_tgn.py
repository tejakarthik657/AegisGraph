import sys
from pathlib import Path

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
    TemporalMemoryTGN,
)

from app.ml_extensions.tgn.temporal_features import (
    create_temporal_features,
)

from app.ml_extensions.tgn.preprocessing import (
    fit_preprocessor,
    save_preprocessor,
)


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

RANDOM_SEED = 42

TRAIN_RATIO = 0.80

HIDDEN_DIM = 64

LEARNING_RATE = 0.001

EPOCHS = 10

WEIGHT_DECAY = 1e-4

# Number of chronological events per backpropagation window.
TBPTT_WINDOW_SIZE = 32

# Maximum gradient norm for stability.
MAX_GRAD_NORM = 1.0


PREPROCESSOR_FILE = (
    RESULTS_DATA_DIR
    / "temporal_tgn_preprocessor.joblib"
)

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

        raise ValueError(
            "Invalid timestamps detected."
        )

    df = (
        df
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    print(
        f"Loaded {len(df)} transactions."
    )

    return df


# ---------------------------------------------------------
# Account mapping
# ---------------------------------------------------------

def create_account_mapping(df):

    accounts = sorted(
        set(df["sender_account"])
        |
        set(df["receiver_account"])
    )

    return {
        account: index
        for index, account
        in enumerate(accounts)
    }



# ---------------------------------------------------------
# Convert transaction features
# ---------------------------------------------------------

def prepare_features(
    df,
    split_index,
):
    """
    Prepare TGN transaction features using the shared
    preprocessing module.

    The scaler is fitted ONLY on the chronological training
    portion and then applied to the complete dataset.
    """

    (
        scaler,
        feature_columns,
        scaled_features,
    ) = fit_preprocessor(
        df=df,
        split_index=split_index,
    )

    # -----------------------------------------------------
    # Save fitted preprocessing artifacts
    # -----------------------------------------------------

    save_preprocessor(
        scaler=scaler,
        feature_columns=feature_columns,
        output_path=PREPROCESSOR_FILE,
    )

    print(
        "TGN preprocessing:"
    )

    print(
        f"  Feature count: {len(feature_columns)}"
    )

    print(
        f"  Preprocessor saved to: "
        f"{PREPROCESSOR_FILE}"
    )

    # -----------------------------------------------------
    # Convert to PyTorch tensor
    # -----------------------------------------------------

    return torch.tensor(
        scaled_features.values,
        dtype=torch.float32,
        
    )

# ---------------------------------------------------------
# Build event tensors
# ---------------------------------------------------------

def build_event_tensors(
    df,
    account_to_index,
):

    sender_ids = torch.tensor(
        df["sender_account"]
        .map(account_to_index)
        .to_numpy(),
        dtype=torch.long,
    )

    receiver_ids = torch.tensor(
        df["receiver_account"]
        .map(account_to_index)
        .to_numpy(),
        dtype=torch.long,
    )

    # -----------------------------------------------------
    # Temporal features
    # -----------------------------------------------------

    temporal_df = create_temporal_features(
        df
    )

    # Use log time delta because raw seconds can be large.
    delta_t = torch.tensor(
        temporal_df[
            "log_time_delta"
        ].to_numpy(
            dtype=np.float32
        ),
        dtype=torch.float32,
    ).reshape(-1, 1)

    # -----------------------------------------------------
    # Absolute event time
    # -----------------------------------------------------

    timestamps = (
        df["timestamp"]
        .astype("int64")
        .to_numpy()
        .astype(np.float64)
    )

    # Convert nanoseconds to seconds.
    timestamps = timestamps / 1_000_000_000.0

    timestamps = (
        timestamps
        - timestamps.min()
    )

    timestamps = torch.tensor(
        timestamps,
        dtype=torch.float32,
    )

    return (
        sender_ids,
        receiver_ids,
        delta_t,
        timestamps,
    )


# ---------------------------------------------------------
# Calculate class weight
# ---------------------------------------------------------

def calculate_pos_weight(
    train_df
):

    positives = int(
        train_df["is_fraud"].sum()
    )

    negatives = (
        len(train_df)
        - positives
    )

    if positives == 0:
        return 1.0

    return negatives / positives


# ---------------------------------------------------------
# Train one chronological pass
# ---------------------------------------------------------


def train_one_epoch(
    model,
    optimizer,
    criterion,
    features,
    sender_ids,
    receiver_ids,
    delta_t,
    timestamps,
    labels,
    start_index,
    end_index,
):
    """
    Train chronologically using truncated backpropagation through
    time (TBPTT).

    Each event is predicted before its memory update.
    Gradients are accumulated over TBPTT_WINDOW_SIZE events.
    Memory history is detached at each window boundary.
    """

    model.train()

    total_loss = 0.0
    count = 0

    window_losses = []
    window_count = 0

    optimizer.zero_grad(set_to_none=True)

    for index in range(start_index, end_index):
        sender = sender_ids[index:index + 1]
        receiver = receiver_ids[index:index + 1]
        transaction = features[index:index + 1]
        time_delta = delta_t[index:index + 1]
        timestamp = timestamps[index:index + 1]
        target = labels[index:index + 1]

        # -----------------------------------------------------
        # 1. Predict BEFORE updating account memory.
        # -----------------------------------------------------

        logits = model.predict_event(
            sender_ids=sender,
            receiver_ids=receiver,
            transaction_features=transaction,
            delta_t=time_delta,
        )

        loss = criterion(logits, target)

        # Retain this event's loss until the window backward pass.
        window_losses.append(loss)
        window_count += 1

        total_loss += loss.detach().item()
        count += 1

        # -----------------------------------------------------
        # 2. Observe the event and update memory WITH gradients.
        # -----------------------------------------------------

        model.update_event_memory(
            sender_ids=sender,
            receiver_ids=receiver,
            transaction_features=transaction,
            delta_t=time_delta,
            timestamps=timestamp,
            track_grad=True,
        )

        # -----------------------------------------------------
        # 3. Backpropagate at the window boundary.
        # -----------------------------------------------------

        is_window_end = (
            window_count >= TBPTT_WINDOW_SIZE
            or index == end_index - 1
        )

        if is_window_end:
            window_loss = torch.stack(window_losses).mean()

            window_loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=MAX_GRAD_NORM,
            )

            optimizer.step()

            # The next window starts from the current memory values,
            # but does not backpropagate through the previous window.
            model.memory.detach_memory()

            optimizer.zero_grad(set_to_none=True)

            window_losses.clear()
            window_count = 0

    if count == 0:
        return 0.0

    return total_loss / count


# ---------------------------------------------------------
# Evaluate chronologically
# ---------------------------------------------------------

def evaluate(
    model,
    features,
    sender_ids,
    receiver_ids,
    delta_t,
    timestamps,
    labels,
    start_index,
    end_index,
):

    model.eval()

    probabilities = []

    true_labels = []

    with torch.no_grad():

        for index in range(
            start_index,
            end_index,
        ):

            sender = sender_ids[
                index:index + 1
            ]

            receiver = receiver_ids[
                index:index + 1
            ]

            transaction = features[
                index:index + 1
            ]

            time_delta = delta_t[
                index:index + 1
            ]

            timestamp = timestamps[
                index:index + 1
            ]

            target = labels[
                index:index + 1
            ]

            # -------------------------------------------------
            # Predict BEFORE updating memory.
            # -------------------------------------------------

            logits = model.predict_event(
                sender_ids=sender,
                receiver_ids=receiver,
                transaction_features=transaction,
                delta_t=time_delta,
            )

            probability = torch.sigmoid(
                logits
            )

            probabilities.append(
                probability.item()
            )

            true_labels.append(
                target.item()
            )

            # -------------------------------------------------
            # After prediction, expose event to memory.
            # -------------------------------------------------

            model.update_event_memory(
                sender_ids=sender,
                receiver_ids=receiver,
                transaction_features=transaction,
                delta_t=time_delta,
                timestamps=timestamp,
            )

    probabilities = np.array(
        probabilities
    )

    true_labels = np.array(
        true_labels
    ).astype(int)

    predictions = (
        probabilities >= 0.5
    ).astype(int)

    precision = precision_score(
        true_labels,
        predictions,
        zero_division=0,
    )

    recall = recall_score(
        true_labels,
        predictions,
        zero_division=0,
    )

    f1 = f1_score(
        true_labels,
        predictions,
        zero_division=0,
    )

    try:

        roc_auc = roc_auc_score(
            true_labels,
            probabilities,
        )

    except ValueError:

        roc_auc = float("nan")

    try:

        pr_auc = average_precision_score(
            true_labels,
            probabilities,
        )

    except ValueError:

        pr_auc = float("nan")

    return {
        "probabilities": probabilities,
        "labels": true_labels,
        "predictions": predictions,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
    }


# ---------------------------------------------------------
# Main training function
# ---------------------------------------------------------

def train_temporal_tgn():

    create_directories()

    print(
        "\n========================================"
    )

    print(
        "      AEGISGRAPH TEMPORAL MEMORY TGN"
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

    num_nodes = len(
        account_to_index
    )

    print(
        f"Graph accounts: {num_nodes}"
    )

    # -----------------------------------------------------
    # Transaction features
    # -----------------------------------------------------

    features = prepare_features(
        df,
        split_index,
    )

    # -----------------------------------------------------
    # Event tensors
    # -----------------------------------------------------

    (
        sender_ids,
        receiver_ids,
        delta_t,
        timestamps,
    ) = build_event_tensors(
        df,
        account_to_index,
    )

    # -----------------------------------------------------
    # Labels
    # -----------------------------------------------------

    labels = torch.tensor(
        df["is_fraud"].to_numpy(
            dtype=np.float32
        ),
        dtype=torch.float32,
    )

    # -----------------------------------------------------
    # Move to device
    # -----------------------------------------------------

    features = features.to(DEVICE)

    sender_ids = sender_ids.to(
        DEVICE
    )

    receiver_ids = receiver_ids.to(
        DEVICE
    )

    delta_t = delta_t.to(DEVICE)

    timestamps = timestamps.to(
        DEVICE
    )

    labels = labels.to(DEVICE)

    # -----------------------------------------------------
    # Model
    # -----------------------------------------------------

    model = TemporalMemoryTGN(
        input_dim=features.shape[1],
        num_nodes=num_nodes,
        hidden_dim=HIDDEN_DIM,
        message_dim=HIDDEN_DIM,
    ).to(DEVICE)

    # -----------------------------------------------------
    # Class weighting
    # -----------------------------------------------------

    pos_weight = calculate_pos_weight(
        train_df
    )

    print(
        f"Positive class weight: "
        f"{pos_weight:.4f}"
    )

    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(
            pos_weight,
            dtype=torch.float32,
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
        "\n========== TEMPORAL TGN TRAINING =========="
    )

    for epoch in range(
        1,
        EPOCHS + 1,
    ):

        # Reset temporal state before each
        # chronological training pass.
        model.reset_memory()

        loss = train_one_epoch(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            features=features,
            sender_ids=sender_ids,
            receiver_ids=receiver_ids,
            delta_t=delta_t,
            timestamps=timestamps,
            labels=labels,
            start_index=0,
            end_index=split_index,
        )

        if (
            epoch == 1
            or epoch % 2 == 0
            or epoch == EPOCHS
        ):

            print(
                f"Epoch "
                f"{epoch:03d}/{EPOCHS} "
                f"| Loss: {loss:.4f}"
            )

    print(
        "\nTemporal TGN training complete."
    )

    # -----------------------------------------------------
    # Test
    # -----------------------------------------------------

    print(
        "\n========== TEMPORAL TGN EVALUATION =========="
    )

    # Rebuild memory using training history only.
    model.reset_memory()

    with torch.no_grad():

        for index in range(
            0,
            split_index,
        ):

            sender = sender_ids[
                index:index + 1
            ]

            receiver = receiver_ids[
                index:index + 1
            ]

            transaction = features[
                index:index + 1
            ]

            time_delta = delta_t[
                index:index + 1
            ]

            timestamp = timestamps[
                index:index + 1
            ]

            # Observe training event.
            model.update_event_memory(
                sender_ids=sender,
                receiver_ids=receiver,
                transaction_features=transaction,
                delta_t=time_delta,
                timestamps=timestamp,
            )

    results = evaluate(
        model=model,
        features=features,
        sender_ids=sender_ids,
        receiver_ids=receiver_ids,
        delta_t=delta_t,
        timestamps=timestamps,
        labels=labels,
        start_index=split_index,
        end_index=len(df),
    )

    # -----------------------------------------------------
    # Print metrics
    # -----------------------------------------------------

    print(
        f"\nPrecision : "
        f"{results['precision']:.4f}"
    )

    print(
        f"Recall    : "
        f"{results['recall']:.4f}"
    )

    print(
        f"F1 Score  : "
        f"{results['f1']:.4f}"
    )

    print(
        f"ROC-AUC   : "
        f"{results['roc_auc']:.4f}"
    )

    print(
        f"PR-AUC    : "
        f"{results['pr_auc']:.4f}"
    )

    print(
        "\nClassification Report:"
    )

    print(
        classification_report(
            results["labels"],
            results["predictions"],
            zero_division=0,
        )
    )

    print(
        "Confusion Matrix:"
    )

    print(
        confusion_matrix(
            results["labels"],
            results["predictions"],
        )
    )

    # -----------------------------------------------------
    # Save transaction predictions
    # -----------------------------------------------------

    test_transactions = df.iloc[
        split_index:
    ].copy()

    prediction_df = pd.DataFrame(
        {
            "transaction_id":
                test_transactions[
                    "transaction_id"
                ].values,

            "timestamp":
                test_transactions[
                    "timestamp"
                ].values,

            "sender_account":
                test_transactions[
                    "sender_account"
                ].values,

            "receiver_account":
                test_transactions[
                    "receiver_account"
                ].values,

            "amount":
                test_transactions[
                    "amount"
                ].values,

            "is_fraud":
                results["labels"],

            "fraud_probability":
                results["probabilities"],

            "prediction":
                results["predictions"],
        }
    )

    prediction_file = (
        RESULTS_DATA_DIR
        / "temporal_tgn_predictions.csv"
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
                "model":
                    "TemporalMemoryTGN",

                "precision":
                    results["precision"],

                "recall":
                    results["recall"],

                "f1":
                    results["f1"],

                "roc_auc":
                    results["roc_auc"],

                "pr_auc":
                    results["pr_auc"],
            }
        ]
    )

    metrics_file = (
        RESULTS_DATA_DIR
        / "temporal_tgn_metrics.csv"
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
        / "temporal_tgn_model.pt"
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
        "    TEMPORAL TGN TRAINING COMPLETE"
    )

    print(
        "========================================"
    )


# ---------------------------------------------------------
# Entry point
# ---------------------------------------------------------

if __name__ == "__main__":
    train_temporal_tgn()