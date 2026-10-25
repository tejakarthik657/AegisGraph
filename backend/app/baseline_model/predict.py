"""
AegisGraph — Baseline Model Prediction.

Load a trained Random Forest model and generate fraud
predictions on the test set.
"""

from pathlib import Path

import joblib
import pandas as pd

from app.baseline_model.baseline_model import BaselineFraudModel
from app.baseline_model.features.feature_engineering import (
    build_features,
)
from app.baseline_model.features.feature_engineering import (
    NON_MODEL_COLUMNS,
    chronological_split_index,
)
from app.utils.config import (
    CLEAN_TRANSACTIONS_FILE,
    RESULTS_DATA_DIR,
)


MODEL_FILE = RESULTS_DATA_DIR / "baseline_rf_model.joblib"
OUTPUT_FILE = RESULTS_DATA_DIR / "baseline_predictions.csv"


def load_model() -> BaselineFraudModel:
    """Load the trained Random Forest model from disk."""

    if not MODEL_FILE.exists():
        raise FileNotFoundError(
            f"Trained model not found: {MODEL_FILE}"
        )

    artifact = joblib.load(MODEL_FILE)
    if isinstance(artifact, dict):
        model = artifact.get("model")
        feature_columns = artifact.get("feature_columns")
        if model is None or not isinstance(feature_columns, list):
            raise ValueError(
                f"Invalid Random Forest model artifact: {MODEL_FILE}"
            )
        model.feature_columns = feature_columns
        return model

    if not isinstance(artifact, BaselineFraudModel):
        raise ValueError(
            f"Invalid Random Forest model artifact: {MODEL_FILE}"
        )
    artifact.feature_columns = list(
        artifact.model.feature_names_in_
    )
    if NON_MODEL_COLUMNS.intersection(artifact.feature_columns):
        raise ValueError(
            "Saved Random Forest model includes excluded feature columns."
        )
    return artifact


def prepare_test_data(
    feature_columns: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load transactions_clean.csv and split into the
    same 80/20 chronological test set used during training.

    Returns
    -------
    test_features:
        Engineered feature matrix for the test set.

    test_df:
        The raw test dataframe (with transaction_id, is_fraud, etc.).
    """

    df = pd.read_csv(CLEAN_TRANSACTIONS_FILE)

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = (
        df.sort_values("timestamp", kind="mergesort")
        .reset_index(drop=True)
    )

    split_index = chronological_split_index(df)
    test_df = df.iloc[split_index:].copy()

    all_features = build_features(
        df,
        feature_columns=feature_columns,
    )
    test_features = all_features.iloc[
        split_index:
    ].copy()

    return test_features, test_df


def predict(
    model: BaselineFraudModel | None = None,
) -> pd.DataFrame:
    """
    Generate fraud predictions on the test set.

    Returns
    -------
    A DataFrame with columns:
        transaction_index, transaction_id,
        predicted_fraud, fraud_probability
    """

    if model is None:
        model = load_model()

    feature_columns = getattr(
        model,
        "feature_columns",
        list(model.model.feature_names_in_),
    )
    test_features, test_df = prepare_test_data(feature_columns)

    predictions = model.predict(test_features)
    probabilities = model.fraud_probability(test_features)

    results = pd.DataFrame({
        "transaction_index": test_df.index,
        "transaction_id": test_df["transaction_id"].values,
        "predicted_fraud": predictions,
        "fraud_probability": probabilities,
    })

    RESULTS_DATA_DIR.mkdir(parents=True, exist_ok=True)

    results.to_csv(OUTPUT_FILE, index=False)

    print(f"Predictions saved to: {OUTPUT_FILE}")
    print(f"Total predictions: {len(results)}")
    print(
        f"Predicted fraud: "
        f"{int(results['predicted_fraud'].sum())}"
    )

    return results


if __name__ == "__main__":
    predict()
