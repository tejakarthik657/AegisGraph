"""
Inference utilities for the TemporalMemoryTGN model.

This module loads the trained TemporalMemoryTGN checkpoint and
provides transaction-level fraud probability inference.

It does not train the model.

Shared preprocessing is loaded from the saved TGN preprocessor so
that inference uses the same feature schema and StandardScaler
fitted during training.
"""

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch

from app.ml_extensions.tgn.tgn_model import TemporalMemoryTGN

from app.ml_extensions.tgn.preprocessing import (
    load_preprocessor,
    transform_transaction_features,
)


# ---------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[4]

RESULTS_DIR = (
    PROJECT_ROOT
    / "data"
    / "results"
)

MODEL_FILE = (
    RESULTS_DIR
    / "temporal_tgn_model.pt"
)

PREPROCESSOR_FILE = (
    RESULTS_DIR
    / "temporal_tgn_preprocessor.joblib"
)


# ---------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------

def load_tgn_model(
    model_path: Optional[Path] = None,
    device: Optional[str] = None,
) -> TemporalMemoryTGN:
    """
    Load the trained TemporalMemoryTGN checkpoint.

    The current project checkpoint is a raw OrderedDict state_dict.
    Model dimensions are inferred directly from its parameters.
    """

    if model_path is None:
        model_path = MODEL_FILE

    if device is None:
        device = "cpu"

    model_path = Path(model_path)

    if not model_path.exists():
        raise FileNotFoundError(
            f"TGN model checkpoint not found:\n"
            f"{model_path}"
        )

    checkpoint = torch.load(
        model_path,
        map_location=device,
        weights_only=False,
    )

    if not isinstance(checkpoint, dict):
        raise ValueError(
            "Expected the TGN checkpoint to be "
            "a state dictionary."
        )

    state_dict = checkpoint

    # -------------------------------------------------------------
    # Infer dimensions from the saved model
    # -------------------------------------------------------------

    if "input_projection.weight" not in state_dict:
        raise ValueError(
            "Checkpoint is missing "
            "input_projection.weight."
        )

    if "memory.memory" not in state_dict:
        raise ValueError(
            "Checkpoint is missing "
            "memory.memory."
        )

    input_projection_weight = (
        state_dict[
            "input_projection.weight"
        ]
    )

    memory_tensor = (
        state_dict[
            "memory.memory"
        ]
    )

    # input_projection.weight shape:
    # [hidden_dim, input_dim]

    hidden_dim = (
        input_projection_weight.shape[0]
    )

    input_dim = (
        input_projection_weight.shape[1]
    )

    # memory.memory shape:
    # [num_nodes, memory_dim]

    num_nodes = (
        memory_tensor.shape[0]
    )

    memory_dim = (
        memory_tensor.shape[1]
    )

    if hidden_dim != memory_dim:
        raise ValueError(
            "Checkpoint dimension mismatch: "
            f"hidden_dim={hidden_dim}, "
            f"memory_dim={memory_dim}"
        )

    # -------------------------------------------------------------
    # Construct the same model architecture
    # -------------------------------------------------------------

    model = TemporalMemoryTGN(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_nodes=num_nodes,
    )

    # -------------------------------------------------------------
    # Load trained parameters
    # -------------------------------------------------------------

    missing_keys, unexpected_keys = (
        model.load_state_dict(
            state_dict,
            strict=False,
        )
    )

    if missing_keys:
        raise ValueError(
            "Missing model parameters in checkpoint:\n"
            + "\n".join(
                str(key)
                for key in missing_keys
            )
        )

    if unexpected_keys:
        raise ValueError(
            "Unexpected parameters in checkpoint:\n"
            + "\n".join(
                str(key)
                for key in unexpected_keys
            )
        )

    model.to(device)

    model.eval()

    return model


# ---------------------------------------------------------------------
# Preprocessor loading
# ---------------------------------------------------------------------

def load_tgn_preprocessor(
    preprocessor_path: Optional[Path] = None,
):
    """
    Load the fitted TGN preprocessing artifacts.

    Returns
    -------
    scaler:
        StandardScaler fitted on the chronological training data.

    feature_columns:
        Exact feature schema used during TGN training.
    """

    if preprocessor_path is None:
        preprocessor_path = PREPROCESSOR_FILE

    preprocessor_path = Path(
        preprocessor_path
    )

    if not preprocessor_path.exists():
        raise FileNotFoundError(
            "TGN preprocessor not found:\n"
            f"{preprocessor_path}\n\n"
            "Run train_temporal_tgn.py first so that "
            "the fitted preprocessing artifacts are saved."
        )

    scaler, feature_columns = (
        load_preprocessor(
            preprocessor_path
        )
    )

    return scaler, feature_columns


# ---------------------------------------------------------------------
# Raw transaction preprocessing
# ---------------------------------------------------------------------

def preprocess_transaction(
    transaction: pd.DataFrame,
    scaler=None,
    feature_columns=None,
    preprocessor_path: Optional[Path] = None,
) -> torch.Tensor:
    """
    Convert raw transaction data into the tensor expected by TGN.

    Parameters
    ----------
    transaction:
        Raw transaction DataFrame.

    scaler:
        Optional previously loaded StandardScaler.

    feature_columns:
        Optional training-time feature schema.

    preprocessor_path:
        Optional path to the saved preprocessor.

    Returns
    -------
    torch.Tensor
        Float32 tensor containing the scaled TGN features.
    """

    if transaction.empty:
        raise ValueError(
            "Cannot preprocess an empty transaction DataFrame."
        )

    # -------------------------------------------------------------
    # Load saved preprocessing artifacts when not supplied
    # -------------------------------------------------------------

    if (
        scaler is None
        or feature_columns is None
    ):
        (
            scaler,
            feature_columns,
        ) = load_tgn_preprocessor(
            preprocessor_path
        )

    # -------------------------------------------------------------
    # Apply the exact training-time feature schema and scaler
    # -------------------------------------------------------------

    scaled_features = (
        transform_transaction_features(
            df=transaction,
            scaler=scaler,
            feature_columns=feature_columns,
        )
    )

    # -------------------------------------------------------------
    # Convert to PyTorch tensor
    # -------------------------------------------------------------

    return torch.tensor(
        scaled_features.values,
        dtype=torch.float32,
    )


# ---------------------------------------------------------------------
# Single-event inference
# ---------------------------------------------------------------------

@torch.no_grad()
def predict_event(
    model: TemporalMemoryTGN,
    sender_id: int,
    receiver_id: int,
    transaction_features: torch.Tensor,
    time_delta: torch.Tensor,
    timestamp: torch.Tensor,
) -> float:
    """
    Predict fraud probability for one transaction event.

    The prediction is made BEFORE the current event updates
    temporal memory.
    """

    model.eval()

    sender_id = torch.as_tensor(
        sender_id,
        dtype=torch.long,
    )

    receiver_id = torch.as_tensor(
        receiver_id,
        dtype=torch.long,
    )

    if sender_id.dim() == 0:
        sender_id = sender_id.unsqueeze(0)

    if receiver_id.dim() == 0:
        receiver_id = receiver_id.unsqueeze(0)

    transaction_features = (
        transaction_features.float()
    )

    if transaction_features.dim() == 1:
        transaction_features = (
            transaction_features.unsqueeze(0)
        )

    time_delta = time_delta.float()

    if time_delta.dim() == 0:
        time_delta = time_delta.unsqueeze(0)

    timestamp = timestamp.float()

    if timestamp.dim() == 0:
        timestamp = timestamp.unsqueeze(0)

    logits = model.predict_event(
        sender_ids=sender_id,
        receiver_ids=receiver_id,
        transaction_features=transaction_features,
        delta_t=time_delta,
    )

    probability = torch.sigmoid(
        logits
    )

    return float(
        probability
        .detach()
        .cpu()
        .reshape(-1)[0]
    )


# ---------------------------------------------------------------------
# Batch prediction helper
# ---------------------------------------------------------------------

def attach_fraud_probabilities(
    predictions: pd.DataFrame,
    probabilities: np.ndarray,
) -> pd.DataFrame:
    """
    Attach fraud probabilities and binary predictions
    to a DataFrame.
    """

    predictions = predictions.copy()

    probabilities = np.asarray(
        probabilities,
        dtype=float,
    )

    if len(predictions) != len(
        probabilities
    ):
        raise ValueError(
            "Number of predictions does not match "
            "number of probability values."
        )

    predictions[
        "fraud_probability"
    ] = probabilities

    predictions[
        "prediction"
    ] = (
        predictions[
            "fraud_probability"
        ]
        >= 0.5
    ).astype(int)

    return predictions


# ---------------------------------------------------------------------
# Checkpoint information
# ---------------------------------------------------------------------

def get_model_dimensions(
    model_path: Optional[Path] = None,
):
    """
    Return the dimensions stored in the trained checkpoint.

    Useful for validating the inference setup.
    """

    if model_path is None:
        model_path = MODEL_FILE

    model_path = Path(
        model_path
    )

    if not model_path.exists():
        raise FileNotFoundError(
            f"TGN model checkpoint not found:\n"
            f"{model_path}"
        )

    checkpoint = torch.load(
        model_path,
        map_location="cpu",
        weights_only=False,
    )

    if not isinstance(checkpoint, dict):
        raise ValueError(
            "Expected a state dictionary."
        )

    if (
        "input_projection.weight"
        not in checkpoint
    ):
        raise ValueError(
            "Checkpoint is missing "
            "input_projection.weight."
        )

    if (
        "memory.memory"
        not in checkpoint
    ):
        raise ValueError(
            "Checkpoint is missing "
            "memory.memory."
        )

    input_projection_weight = (
        checkpoint[
            "input_projection.weight"
        ]
    )

    memory_tensor = (
        checkpoint[
            "memory.memory"
        ]
    )

    return {
        "input_dim": int(
            input_projection_weight.shape[1]
        ),
        "hidden_dim": int(
            input_projection_weight.shape[0]
        ),
        "num_nodes": int(
            memory_tensor.shape[0]
        ),
        "memory_dim": int(
            memory_tensor.shape[1]
        ),
    }


# ---------------------------------------------------------------------
# Preprocessor information
# ---------------------------------------------------------------------

def get_preprocessor_information(
    preprocessor_path: Optional[Path] = None,
):
    """
    Return information about the saved TGN preprocessor.
    """

    scaler, feature_columns = (
        load_tgn_preprocessor(
            preprocessor_path
        )
    )

    return {
        "feature_count": len(
            feature_columns
        ),
        "feature_columns": list(
            feature_columns
        ),
        "scaler_type": type(
            scaler
        ).__name__,
    }


# ---------------------------------------------------------------------
# Complete inference setup validation
# ---------------------------------------------------------------------

def validate_inference_setup(
    model_path: Optional[Path] = None,
    preprocessor_path: Optional[Path] = None,
):
    """
    Validate that the trained model and preprocessing artifacts
    are compatible.

    Returns
    -------
    dict
        Validation information.
    """

    model_dimensions = (
        get_model_dimensions(
            model_path
        )
    )

    (
        scaler,
        feature_columns,
    ) = load_tgn_preprocessor(
        preprocessor_path
    )

    feature_count = len(
        feature_columns
    )

    if model_dimensions[
        "input_dim"
    ] != feature_count:
        raise ValueError(
            "TGN model/preprocessor mismatch:\n"
            f"Model input_dim: "
            f"{model_dimensions['input_dim']}\n"
            f"Preprocessor feature count: "
            f"{feature_count}"
        )

    if feature_count != 22:
        raise ValueError(
            "Unexpected TGN feature count:\n"
            f"Expected: 22\n"
            f"Actual: {feature_count}"
        )

    return {
        "model_dimensions": model_dimensions,
        "feature_count": feature_count,
        "feature_columns": list(
            feature_columns
        ),
        "scaler_type": type(
            scaler
        ).__name__,
        "compatible": True,
    }


# ---------------------------------------------------------------------
# Script entry point
# ---------------------------------------------------------------------

if __name__ == "__main__":

    print("=" * 70)
    print("AEGISGRAPH TGN INFERENCE CHECK")
    print("=" * 70)

    # -------------------------------------------------------------
    # Model checkpoint
    # -------------------------------------------------------------

    print()
    print("Checkpoint dimensions:")

    dimensions = get_model_dimensions()

    for key, value in dimensions.items():
        print(
            f"  {key}: {value}"
        )

    # -------------------------------------------------------------
    # Preprocessor
    # -------------------------------------------------------------

    print()
    print("Loading TGN preprocessor...")

    (
        scaler,
        feature_columns,
    ) = load_tgn_preprocessor()

    print(
        "Preprocessor loaded successfully."
    )

    print(
        f"  Feature count: "
        f"{len(feature_columns)}"
    )

    print(
        f"  Scaler: "
        f"{type(scaler).__name__}"
    )

    # -------------------------------------------------------------
    # Compatibility check
    # -------------------------------------------------------------

    print()
    print(
        "Validating model/preprocessor compatibility..."
    )

    validation = (
        validate_inference_setup()
    )

    print(
        "Model/preprocessor compatibility: "
        "OK"
    )

    print(
        f"  Model input_dim: "
        f"{validation['model_dimensions']['input_dim']}"
    )

    print(
        f"  Preprocessor features: "
        f"{validation['feature_count']}"
    )

    # -------------------------------------------------------------
    # Model
    # -------------------------------------------------------------

    print()
    print("Loading trained model...")

    model = load_tgn_model()

    print(
        "Model loaded successfully."
    )

    # -------------------------------------------------------------
    # Final status
    # -------------------------------------------------------------

    print()
    print(
        "Inference module check complete."
    )

    print("=" * 70)