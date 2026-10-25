"""Tests for required THGNN integration in the main risk-fusion pipeline."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from app.risk import risk_fusion


class TestRiskFusionTHGNNIntegration(unittest.TestCase):
    def _write_csv(self, folder: Path, name: str, df: pd.DataFrame) -> Path:
        path = folder / name
        df.to_csv(path, index=False)
        return path

    def _valid_paths(self, folder: Path) -> dict[str, Path]:
        transactions = pd.DataFrame({
            "transaction_id": [f"TX{i}" for i in range(10)],
            "timestamp": pd.date_range("2025-01-01", periods=10, freq="h"),
            "sender_account": [f"A{i}" for i in range(10)],
            "receiver_account": [f"B{i}" for i in range(10)],
            "amount": [100.0] * 10,
            "is_fraud": [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            "typology": ["none"] * 8 + ["layering", "none"],
        })
        return {
            "TRANSACTIONS_FILE": self._write_csv(
                folder,
                "transactions.csv",
                transactions,
            ),
            "RF_PREDICTIONS_FILE": self._write_csv(
                folder,
                "rf.csv",
                pd.DataFrame({
                    "transaction_index": [8, 9],
                    "fraud_probability": [0.2, 0.2],
                }),
            ),
            "TGN_PREDICTIONS_FILE": self._write_csv(
                folder,
                "tgn.csv",
                pd.DataFrame({
                    "transaction_id": ["TX8", "TX9"],
                    "fraud_probability": [0.4, 0.4],
                }),
            ),
            "THGNN_PREDICTIONS_FILE": self._write_csv(
                folder,
                "thgnn.csv",
                pd.DataFrame({
                    "transaction_id": ["TX8", "TX9"],
                    "fraud_probability": [1.0, 0.0],
                }),
            ),
            "RULE_PREDICTIONS_FILE": self._write_csv(
                folder,
                "rules.csv",
                pd.DataFrame({
                    "transaction_id": ["TX8", "TX9"],
                    "large_transaction": [0, 0],
                    "structuring": [0, 0],
                    "rapid_movement": [0, 0],
                    "fan_out": [0, 0],
                    "circular_movement": [0, 0],
                    "rule_prediction": [0, 0],
                }),
            ),
            "OUTPUT_FILE": folder / "risk.csv",
            "METRICS_FILE": folder / "risk_metrics.csv",
        }

    def _check_alignment_error(
        self,
        source_attribute: str,
        error_fragment: str,
        replace_key,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            paths = self._valid_paths(folder)
            source_path = paths[source_attribute]
            predictions = pd.read_csv(source_path)
            key_column = (
                "transaction_index"
                if source_attribute == "RF_PREDICTIONS_FILE"
                else "transaction_id"
            )
            predictions = replace_key(predictions, key_column)
            predictions.to_csv(source_path, index=False)

            with patch.multiple(risk_fusion, **paths):
                with self.assertRaisesRegex(ValueError, error_fragment):
                    risk_fusion.build_risk_fusion()

    def test_fourth_model_is_merged_and_contributes_to_risk(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            paths = self._valid_paths(folder)
            with patch.multiple(risk_fusion, **paths):
                output = risk_fusion.build_risk_fusion()

            self.assertIn("thgnn_probability", output.columns)
            self.assertAlmostEqual(output.loc[output.transaction_id == "TX8", "risk_score"].iloc[0], 0.4)
            self.assertAlmostEqual(output.loc[output.transaction_id == "TX9", "risk_score"].iloc[0], 0.15)
            metrics = pd.read_csv(paths["METRICS_FILE"]).iloc[0]
            self.assertEqual(metrics["threshold"], 0.5)
            self.assertEqual(metrics["test_transactions"], 2)
            self.assertEqual(metrics["true_positives"], 0)
            self.assertEqual(metrics["false_positives"], 0)
            self.assertEqual(metrics["true_negatives"], 1)
            self.assertEqual(metrics["false_negatives"], 1)

    def test_missing_thgnn_file_raises_error(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            paths = self._valid_paths(folder)
            paths["THGNN_PREDICTIONS_FILE"] = folder / "missing_thgnn.csv"
            with patch.multiple(risk_fusion, **paths):
                with self.assertRaises(FileNotFoundError):
                    risk_fusion.build_risk_fusion()

    def test_missing_ids_raise_error(self):
        sources = (
            "RF_PREDICTIONS_FILE",
            "TGN_PREDICTIONS_FILE",
            "THGNN_PREDICTIONS_FILE",
            "RULE_PREDICTIONS_FILE",
        )
        for source in sources:
            with self.subTest(source=source):
                self._check_alignment_error(
                    source,
                    "missing",
                    lambda predictions, _: predictions.iloc[:-1].copy(),
                )

    def test_unexpected_ids_raise_error(self):
        sources = (
            "RF_PREDICTIONS_FILE",
            "TGN_PREDICTIONS_FILE",
            "THGNN_PREDICTIONS_FILE",
            "RULE_PREDICTIONS_FILE",
        )
        for source in sources:
            with self.subTest(source=source):
                def add_unexpected(predictions, key_column):
                    predictions = predictions.copy()
                    predictions.loc[predictions.index[-1], key_column] = (
                        10 if key_column == "transaction_index" else "TX_EXTRA"
                    )
                    return predictions

                self._check_alignment_error(
                    source,
                    "unexpected",
                    add_unexpected,
                )

    def test_duplicate_ids_raise_error(self):
        sources = (
            "RF_PREDICTIONS_FILE",
            "TGN_PREDICTIONS_FILE",
            "THGNN_PREDICTIONS_FILE",
            "RULE_PREDICTIONS_FILE",
        )
        for source in sources:
            with self.subTest(source=source):
                self._check_alignment_error(
                    source,
                    "duplicate",
                    lambda predictions, _: pd.concat(
                        [predictions, predictions.iloc[[0]]],
                        ignore_index=True,
                    ),
                )

    def test_single_label_test_set_has_safe_metrics(self):
        for fraud_label in (0, 1):
            with self.subTest(fraud_label=fraud_label):
                with tempfile.TemporaryDirectory() as temp:
                    folder = Path(temp)
                    paths = self._valid_paths(folder)
                    transactions = pd.read_csv(paths["TRANSACTIONS_FILE"])
                    transactions.loc[
                        transactions["transaction_id"].isin(["TX8", "TX9"]),
                        "is_fraud",
                    ] = fraud_label
                    transactions.to_csv(
                        paths["TRANSACTIONS_FILE"],
                        index=False,
                    )

                    with patch.multiple(risk_fusion, **paths):
                        risk_fusion.build_risk_fusion()

                    metrics = pd.read_csv(paths["METRICS_FILE"]).iloc[0]
                    self.assertTrue(pd.isna(metrics["roc_auc"]))
                    self.assertEqual(metrics["pr_auc"], fraud_label)
                    self.assertTrue(pd.notna(metrics["precision"]))
                    self.assertTrue(pd.notna(metrics["recall"]))
                    self.assertTrue(pd.notna(metrics["f1"]))


if __name__ == "__main__":
    unittest.main()
