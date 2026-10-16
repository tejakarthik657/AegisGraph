import sys
from pathlib import Path

import pandas as pd

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
    RESULTS_DATA_DIR,
    create_directories,
)

from app.baseline_model.baseline_model import (
    BaselineFraudModel,
)


# ---------------------------------------------------------
# Files
# ---------------------------------------------------------
FEATURES_FILE = (
    RESULTS_DATA_DIR
    / "baseline_features.csv"
)

PREDICTIONS_FILE = (
    RESULTS_DATA_DIR
    / "baseline_predictions.csv"
)

METRICS_FILE = (
    RESULTS_DATA_DIR
    / "model_metrics.csv"
)

FEATURE_IMPORTANCE_FILE = (
    RESULTS_DATA_DIR
    / "baseline_feature_importance.csv"
)


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------
TRAIN_RATIO = 0.80


# ---------------------------------------------------------
# Load feature dataset
# ---------------------------------------------------------
def load_dataset():
    """
    Load the engineered feature dataset.
    """

    print(
        "\nLoading feature dataset..."
    )

    df = pd.read_csv(
        FEATURES_FILE
    )

    print(
        f"Loaded {len(df)} transactions."
    )

    return df


# ---------------------------------------------------------
# Prepare train/test data
# ---------------------------------------------------------
def prepare_data(
    df: pd.DataFrame,
):
    """
    Prepare chronological train/test datasets.

    The first 80% of transactions are used for training.
    The final 20% are used for testing.
    """

    print(
        "\n========== DATA PREPARATION =========="
    )

    # -----------------------------------------------------
    # Make a copy
    # -----------------------------------------------------
    df = df.copy()

    # -----------------------------------------------------
    # Target
    # -----------------------------------------------------
    target_column = "is_fraud"

    # -----------------------------------------------------
    # Remove labels and non-model columns
    # -----------------------------------------------------
    forbidden_columns = [
        "transaction_id",
        "timestamp",
        "is_fraud",
        "typology",
    ]

    feature_columns = [
        column
        for column in df.columns
        if column not in forbidden_columns
    ]

    # -----------------------------------------------------
    # Make sure all features are numeric
    # -----------------------------------------------------
    X = df[
        feature_columns
    ].copy()

    y = df[
        target_column
    ].astype(int)

    # -----------------------------------------------------
    # Handle any remaining missing values
    # -----------------------------------------------------
    X = X.fillna(0)

    # -----------------------------------------------------
    # Chronological split
    # -----------------------------------------------------
    split_index = int(
        len(df) * TRAIN_RATIO
    )

    X_train = X.iloc[
        :split_index
    ].copy()

    X_test = X.iloc[
        split_index:
    ].copy()

    y_train = y.iloc[
        :split_index
    ].copy()

    y_test = y.iloc[
        split_index:
    ].copy()

    print(
        f"Training transactions: "
        f"{len(X_train)}"
    )

    print(
        f"Testing transactions: "
        f"{len(X_test)}"
    )

    print(
        f"\nTraining fraud count: "
        f"{y_train.sum()}"
    )

    print(
        f"Testing fraud count: "
        f"{y_test.sum()}"
    )

    print(
        f"\nNumber of model features: "
        f"{len(feature_columns)}"
    )

    return (
        X_train,
        X_test,
        y_train,
        y_test,
        feature_columns,
    )


# ---------------------------------------------------------
# Train model
# ---------------------------------------------------------
def train_model(
    X_train,
    y_train,
):
    """
    Train Random Forest baseline.
    """

    print(
        "\n========== MODEL TRAINING =========="
    )

    model = BaselineFraudModel(
        n_estimators=200,
        max_depth=12,
        min_samples_leaf=2,
        random_state=42,
    )

    model.fit(
        X_train,
        y_train,
    )

    print(
        "Random Forest training complete."
    )

    return model


# ---------------------------------------------------------
# Evaluate model
# ---------------------------------------------------------
def evaluate_model(
    model,
    X_test,
    y_test,
):
    """
    Evaluate the trained model.
    """

    print(
        "\n========== MODEL EVALUATION =========="
    )

    # -----------------------------------------------------
    # Predictions
    # -----------------------------------------------------
    predictions = model.predict(
        X_test
    )

    fraud_probabilities = (
        model.fraud_probability(
            X_test
        )
    )

    # -----------------------------------------------------
    # Metrics
    # -----------------------------------------------------
    precision = precision_score(
        y_test,
        predictions,
        zero_division=0,
    )

    recall = recall_score(
        y_test,
        predictions,
        zero_division=0,
    )

    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0,
    )

    # ROC-AUC requires both classes
    if y_test.nunique() == 2:

        roc_auc = roc_auc_score(
            y_test,
            fraud_probabilities,
        )

        pr_auc = average_precision_score(
            y_test,
            fraud_probabilities,
        )

    else:

        roc_auc = float("nan")
        pr_auc = float("nan")

    # -----------------------------------------------------
    # Print metrics
    # -----------------------------------------------------
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

    # -----------------------------------------------------
    # Classification report
    # -----------------------------------------------------
    print(
        "\nClassification Report:"
    )

    print(
        classification_report(
            y_test,
            predictions,
            zero_division=0,
        )
    )

    # -----------------------------------------------------
    # Confusion matrix
    # -----------------------------------------------------
    matrix = confusion_matrix(
        y_test,
        predictions,
    )

    print(
        "Confusion Matrix:"
    )

    print(matrix)

    metrics = {
        "model": "RandomForest_Baseline",
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
    }

    return (
        predictions,
        fraud_probabilities,
        metrics,
    )


# ---------------------------------------------------------
# Save predictions
# ---------------------------------------------------------
def save_predictions(
    df,
    predictions,
    fraud_probabilities,
    test_start_index,
):
    """
    Save test-set predictions.
    """

    test_df = df.iloc[
        test_start_index:
    ].copy()

    result = pd.DataFrame(
        {
            "transaction_index":
                test_df.index,

            "actual_is_fraud":
                test_df["is_fraud"].values,

            "actual_typology":
                test_df["typology"].values,

            "predicted_is_fraud":
                predictions,

            "fraud_probability":
                fraud_probabilities,
        }
    )

    result.to_csv(
        PREDICTIONS_FILE,
        index=False,
    )

    print(
        f"\nPredictions saved to:"
        f"\n{PREDICTIONS_FILE}"
    )

    return result


# ---------------------------------------------------------
# Save metrics
# ---------------------------------------------------------
def save_metrics(
    metrics,
):
    """
    Save evaluation metrics.
    """

    metrics_df = pd.DataFrame(
        [metrics]
    )

    metrics_df.to_csv(
        METRICS_FILE,
        index=False,
    )

    print(
        f"Metrics saved to:"
        f"\n{METRICS_FILE}"
    )


# ---------------------------------------------------------
# Save feature importance
# ---------------------------------------------------------
def save_feature_importance(
    model,
    feature_columns,
):
    """
    Save Random Forest feature importance.
    """

    importance = (
        model.feature_importance(
            feature_columns
        )
    )

    importance.to_csv(
        FEATURE_IMPORTANCE_FILE,
        index=False,
    )

    print(
        f"Feature importance saved to:"
        f"\n{FEATURE_IMPORTANCE_FILE}"
    )

    print(
        "\nTop 10 features:"
    )

    print(
        importance.head(10)
    )


# ---------------------------------------------------------
# Main training pipeline
# ---------------------------------------------------------
def run_training():

    print(
        "\n========================================"
    )

    print(
        "       AEGISGRAPH BASELINE MODEL"
    )

    print(
        "========================================"
    )

    create_directories()

    # -----------------------------------------------------
    # Load data
    # -----------------------------------------------------
    df = load_dataset()

    # -----------------------------------------------------
    # Prepare data
    # -----------------------------------------------------
    (
        X_train,
        X_test,
        y_train,
        y_test,
        feature_columns,
    ) = prepare_data(df)

    # -----------------------------------------------------
    # Train
    # -----------------------------------------------------
    model = train_model(
        X_train,
        y_train,
    )

    # -----------------------------------------------------
    # Evaluate
    # -----------------------------------------------------
    (
        predictions,
        fraud_probabilities,
        metrics,
    ) = evaluate_model(
        model,
        X_test,
        y_test,
    )

    # -----------------------------------------------------
    # Calculate test starting index
    # -----------------------------------------------------
    test_start_index = int(
        len(df) * TRAIN_RATIO
    )

    # -----------------------------------------------------
    # Save predictions
    # -----------------------------------------------------
    save_predictions(
        df,
        predictions,
        fraud_probabilities,
        test_start_index,
    )

    # -----------------------------------------------------
    # Save metrics
    # -----------------------------------------------------
    save_metrics(
        metrics
    )

    # -----------------------------------------------------
    # Save feature importance
    # -----------------------------------------------------
    save_feature_importance(
        model,
        feature_columns,
    )

    print(
        "\n========================================"
    )

    print(
        "       BASELINE TRAINING COMPLETE"
    )

    print(
        "========================================"
    )

    return model


# ---------------------------------------------------------
# Script entry point
# ---------------------------------------------------------
if __name__ == "__main__":
    run_training()