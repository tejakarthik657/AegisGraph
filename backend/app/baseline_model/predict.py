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

    return joblib.load(MODEL_FILE)


def prepare_test_data() -> tuple[pd.DataFrame, pd.DataFrame]:
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
    df = df.sort_values("timestamp")

    split_index = int(len(df) * 0.80)
    test_df = df.iloc[split_index:].copy()

    test_features = build_features(test_df)

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

    test_features, test_df = prepare_test_data()

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
