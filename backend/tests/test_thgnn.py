"""Tests for the temporal heterogeneous GNN prototype."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import joblib
import pandas as pd

try:
    import torch
    from torch_geometric.data import HeteroData
    TORCH_GEOMETRIC_AVAILABLE = True
except (ImportError, OSError):
    TORCH_GEOMETRIC_AVAILABLE = False


@unittest.skipUnless(TORCH_GEOMETRIC_AVAILABLE, "PyTorch/PyG unavailable")
class TestTHGNN(unittest.TestCase):
    @staticmethod
    def _transactions(rows):
        return pd.DataFrame(
            [
                {
                    "transaction_id": transaction_id,
                    "timestamp": pd.Timestamp(timestamp),
                    "sender_account": sender,
                    "receiver_account": receiver,
                    "amount": amount,
                    "currency": "USD",
                    "transaction_type": "TRANSFER",
                    "channel": "ONLINE",
                    "sender_country": "US",
                    "receiver_country": "US",
                    "is_fraud": label,
                    "typology": "synthetic" if label else "none",
                }
                for (
                    transaction_id,
                    timestamp,
                    sender,
                    receiver,
                    amount,
                    label,
                ) in rows
            ]
        )

    def test_model_outputs_one_logit_per_transaction(self):
        from app.ml_extensions.thgnn.model import TemporalHeterogeneousFraudGNN

        graph = HeteroData()
        graph["account"].x = torch.randn(3, 5)
        graph["transaction"].x = torch.randn(2, 7)
        graph["transaction"].time_features = torch.tensor([[0.0, 0.0], [0.1, 1.0]])
        graph["account", "sends", "transaction"].edge_index = torch.tensor([[0, 1], [0, 1]])
        graph["transaction", "sent_by", "account"].edge_index = torch.tensor([[0, 1], [0, 1]])
        graph["transaction", "received_by", "account"].edge_index = torch.tensor([[0, 1], [1, 2]])
        graph["account", "receives", "transaction"].edge_index = torch.tensor([[1, 2], [0, 1]])

        model = TemporalHeterogeneousFraudGNN(5, 7, hidden_dim=16)
        output = model(graph)
        self.assertEqual(tuple(output.shape), (2,))

    def test_model_requires_temporal_features(self):
        from app.ml_extensions.thgnn.model import TemporalHeterogeneousFraudGNN

        graph = HeteroData()
        graph["account"].x = torch.randn(2, 5)
        graph["transaction"].x = torch.randn(1, 7)
        for relation in TemporalHeterogeneousFraudGNN.RELATIONS:
            graph[relation].edge_index = torch.tensor([[0], [0]])
        model = TemporalHeterogeneousFraudGNN(5, 7, hidden_dim=16)
        with self.assertRaises(ValueError):
            model(graph)

    def test_future_history_does_not_change_earlier_target_features(self):
        from app.ml_extensions.thgnn.preprocessing import build_thgnn_graph

        history = self._transactions([
            ("H1", "2025-01-01 09:00", "A", "B", 20.0, 0),
        ])
        target = self._transactions([
            ("T1", "2025-01-01 10:00", "A", "C", 30.0, 1),
        ])
        future = self._transactions([
            ("F1", "2025-01-01 11:00", "A", "D", 900.0, 0),
            ("F2", "2025-01-01 12:00", "D", "E", 1500.0, 1),
        ])

        _, account_scaler, transaction_scaler, _, feature_columns = (
            build_thgnn_graph(target, fit_scalers=True, history_transactions=history)
        )
        earlier_graph, _, _, earlier_labels, _ = build_thgnn_graph(
            target,
            fit_scalers=False,
            account_scaler=account_scaler,
            transaction_scaler=transaction_scaler,
            feature_columns=feature_columns,
            history_transactions=history,
        )
        with_future_graph, _, _, _, _ = build_thgnn_graph(
            target,
            fit_scalers=False,
            account_scaler=account_scaler,
            transaction_scaler=transaction_scaler,
            feature_columns=feature_columns,
            history_transactions=pd.concat([history, future], ignore_index=True),
        )

        self.assertTrue(torch.equal(
            earlier_graph["account"].x,
            with_future_graph["account"].x,
        ))
        self.assertTrue(torch.equal(
            earlier_graph["transaction"].x,
            with_future_graph["transaction"].x,
        ))
        self.assertTrue(torch.equal(
            earlier_graph["transaction"].time_features,
            with_future_graph["transaction"].time_features,
        ))
        self.assertEqual(earlier_labels.tolist(), [1.0])
        self.assertNotIn("is_fraud", feature_columns)

    def test_equal_timestamp_history_is_excluded(self):
        from app.ml_extensions.thgnn.preprocessing import build_thgnn_graph

        target = self._transactions([
            ("T1", "2025-01-01 10:00", "A", "C", 30.0, 1),
        ])
        prior = self._transactions([
            ("H1", "2025-01-01 09:00", "A", "B", 20.0, 0),
        ])
        equal_time = self._transactions([
            ("H2", "2025-01-01 10:00", "A", "D", 900.0, 1),
        ])
        _, account_scaler, transaction_scaler, _, feature_columns = (
            build_thgnn_graph(target, fit_scalers=True, history_transactions=prior)
        )

        prior_graph, _, _, _, _ = build_thgnn_graph(
            target,
            fit_scalers=False,
            account_scaler=account_scaler,
            transaction_scaler=transaction_scaler,
            feature_columns=feature_columns,
            history_transactions=prior,
        )
        equal_timestamp_graph, _, _, _, _ = build_thgnn_graph(
            target,
            fit_scalers=False,
            account_scaler=account_scaler,
            transaction_scaler=transaction_scaler,
            feature_columns=feature_columns,
            history_transactions=pd.concat(
                [prior, equal_time],
                ignore_index=True,
            ),
        )

        self.assertTrue(torch.equal(
            prior_graph["account"].x,
            equal_timestamp_graph["account"].x,
        ))
        self.assertTrue(torch.equal(
            prior_graph["transaction"].time_features,
            equal_timestamp_graph["transaction"].time_features,
        ))

    def test_empty_history_has_zero_account_aggregates(self):
        from app.ml_extensions.thgnn.preprocessing import build_thgnn_graph

        target = self._transactions([
            ("T1", "2025-01-01 10:00", "A", "C", 30.0, 1),
        ])
        graph, account_scaler, transaction_scaler, labels, feature_columns = build_thgnn_graph(
            target,
            fit_scalers=True,
            history_transactions=pd.DataFrame(),
        )
        changed_label = target.copy()
        changed_label["is_fraud"] = 0
        changed_label["typology"] = "different"
        changed_graph, _, _, changed_labels, _ = build_thgnn_graph(
            changed_label,
            fit_scalers=False,
            account_scaler=account_scaler,
            transaction_scaler=transaction_scaler,
            feature_columns=feature_columns,
            history_transactions=pd.DataFrame(),
        )

        self.assertTrue(torch.equal(
            graph["account"].x,
            torch.zeros_like(graph["account"].x),
        ))
        self.assertTrue(torch.equal(
            graph["transaction"].x,
            changed_graph["transaction"].x,
        ))
        self.assertEqual(labels.tolist(), [1.0])
        self.assertEqual(changed_labels.tolist(), [0.0])
        self.assertNotIn("is_fraud", feature_columns)

    def test_target_transaction_is_never_its_own_history(self):
        from app.ml_extensions.thgnn.preprocessing import build_thgnn_graph

        target = self._transactions([
            ("T1", "2025-01-01 10:00", "A", "C", 30.0, 1),
        ])
        _, account_scaler, transaction_scaler, _, feature_columns = (
            build_thgnn_graph(target, fit_scalers=True)
        )
        graph, _, _, _, _ = build_thgnn_graph(
            target,
            fit_scalers=False,
            account_scaler=account_scaler,
            transaction_scaler=transaction_scaler,
            feature_columns=feature_columns,
            history_transactions=target,
        )

        self.assertTrue(torch.equal(
            graph["account"].x,
            torch.zeros_like(graph["account"].x),
        ))
        self.assertEqual(graph["transaction"].x.shape[0], 1)
        self.assertEqual(graph["account"].x.shape[0], 2)

    def test_same_timestamp_peer_does_not_change_target_prediction(self):
        from app.ml_extensions.thgnn.model import TemporalHeterogeneousFraudGNN
        from app.ml_extensions.thgnn.preprocessing import build_thgnn_graph

        history = self._transactions([
            ("H1", "2025-01-01 09:00", "A", "B", 20.0, 0),
        ])
        target = self._transactions([
            ("T1", "2025-01-01 10:00", "A", "C", 30.0, 1),
            ("T2", "2025-01-01 10:00", "A", "D", 900.0, 0),
        ])
        seed_graph, account_scaler, transaction_scaler, _, feature_columns = (
            build_thgnn_graph(
                self._transactions([
                    ("S1", "2025-01-01 08:00", "X", "Y", 10.0, 0),
                ]),
                fit_scalers=True,
                history_transactions=pd.DataFrame(),
            )
        )
        model = TemporalHeterogeneousFraudGNN(
            account_input_dim=seed_graph["account"].x.shape[1],
            transaction_input_dim=seed_graph["transaction"].x.shape[1],
            hidden_dim=8,
            dropout=0.0,
        )
        model.eval()

        def predict(batch):
            graph, _, _, _, _ = build_thgnn_graph(
                batch,
                fit_scalers=False,
                account_scaler=account_scaler,
                transaction_scaler=transaction_scaler,
                feature_columns=feature_columns,
                history_transactions=history,
            )
            return model(graph)

        target_only = predict(target.iloc[[0]])
        both_targets = predict(target)
        reordered = predict(target.iloc[::-1])

        self.assertAlmostEqual(
            target_only[0].item(),
            both_targets[0].item(),
            places=6,
        )
        self.assertAlmostEqual(
            both_targets[0].item(),
            reordered[1].item(),
            places=6,
        )

    def test_graph_predictions_are_independent_of_label_values(self):
        from app.ml_extensions.thgnn.model import TemporalHeterogeneousFraudGNN
        from app.ml_extensions.thgnn.preprocessing import build_thgnn_graph

        target = self._transactions([
            ("T1", "2025-01-01 10:00", "A", "C", 30.0, 1),
            ("T2", "2025-01-01 10:00", "B", "D", 40.0, 0),
        ])
        _, account_scaler, transaction_scaler, _, feature_columns = (
            build_thgnn_graph(target, fit_scalers=True)
        )
        original_graph, _, _, labels, _ = build_thgnn_graph(
            target,
            fit_scalers=False,
            account_scaler=account_scaler,
            transaction_scaler=transaction_scaler,
            feature_columns=feature_columns,
        )
        relabeled = target.copy()
        relabeled["is_fraud"] = 1 - relabeled["is_fraud"]
        relabeled["typology"] = "different_label"
        changed_graph, _, _, changed_labels, _ = build_thgnn_graph(
            relabeled,
            fit_scalers=False,
            account_scaler=account_scaler,
            transaction_scaler=transaction_scaler,
            feature_columns=feature_columns,
        )
        model = TemporalHeterogeneousFraudGNN(
            account_input_dim=original_graph["account"].x.shape[1],
            transaction_input_dim=original_graph["transaction"].x.shape[1],
            hidden_dim=8,
            dropout=0.0,
        )
        model.eval()

        self.assertTrue(torch.equal(original_graph["account"].x, changed_graph["account"].x))
        self.assertTrue(torch.equal(original_graph["transaction"].x, changed_graph["transaction"].x))
        self.assertTrue(torch.equal(model(original_graph), model(changed_graph)))
        self.assertNotEqual(labels.tolist(), changed_labels.tolist())
        self.assertNotIn("typology", feature_columns)

    def test_training_runs_when_test_partition_has_one_class(self):
        from app.ml_extensions.thgnn import train

        transactions = self._transactions([
            (f"TX{i}", f"2025-01-01 {i:02d}:00", f"A{i % 3}", f"B{i % 4}", 10.0 + i, i % 2)
            for i in range(8)
        ] + [
            ("TX8", "2025-01-01 08:00", "A0", "B0", 18.0, 0),
            ("TX9", "2025-01-01 09:00", "A1", "B1", 19.0, 0),
        ])

        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            data_file = folder / "transactions.csv"
            transactions.to_csv(data_file, index=False)
            with (
                patch.object(train, "CLEAN_TRANSACTIONS_FILE", data_file),
                patch.object(train, "RESULTS_DATA_DIR", folder),
                patch.object(train, "MODEL_FILE", folder / "model.pt"),
                patch.object(train, "PREPROCESSOR_FILE", folder / "preprocessor.joblib"),
                patch.object(train, "PREDICTIONS_FILE", folder / "predictions.csv"),
                patch.object(train, "METRICS_FILE", folder / "metrics.csv"),
                patch.object(train, "EPOCHS", 1),
                patch.object(train.torch.cuda, "is_available", return_value=False),
            ):
                train.main()
                metrics = pd.read_csv(folder / "metrics.csv")

        self.assertTrue(metrics["roc_auc"].isna().all())
        self.assertTrue(metrics["pr_auc"].isna().all())

    def test_chronological_split_does_not_divide_timestamp_batch(self):
        from app.ml_extensions.thgnn import train

        transactions = self._transactions([
            (f"TX{i}", f"2025-01-01 {i:02d}:00", f"A{i}", f"B{i}", 10.0, i % 2)
            for i in range(8)
        ] + [
            ("TX8", "2025-01-01 08:00", "A8", "B8", 18.0, 0),
            ("TX9", "2025-01-01 08:00", "A9", "B9", 19.0, 0),
            ("TX10", "2025-01-01 09:00", "A10", "B10", 20.0, 1),
        ])

        with tempfile.TemporaryDirectory() as temp:
            data_file = Path(temp) / "transactions.csv"
            transactions.to_csv(data_file, index=False)
            with patch.object(train, "CLEAN_TRANSACTIONS_FILE", data_file):
                train_df, test_df = train._load_split()

        self.assertEqual(len(train_df), 8)
        self.assertLess(
            train_df["timestamp"].max(),
            test_df["timestamp"].min(),
        )

    def test_inference_ignores_future_history_and_advances_causally(self):
        from app.ml_extensions.thgnn.inference import score_transactions
        from app.ml_extensions.thgnn.model import TemporalHeterogeneousFraudGNN
        from app.ml_extensions.thgnn.preprocessing import build_thgnn_graph

        history = self._transactions([
            ("H1", "2025-01-01 09:00", "A", "B", 20.0, 0),
        ])
        future = self._transactions([
            ("F1", "2025-01-01 13:00", "D", "E", 1500.0, 1),
        ])
        targets = self._transactions([
            ("T1", "2025-01-01 10:00", "A", "C", 30.0, 1),
            ("T2", "2025-01-01 11:00", "C", "D", 40.0, 0),
        ]).drop(columns="is_fraud")
        seed_graph, account_scaler, transaction_scaler, _, feature_columns = (
            build_thgnn_graph(
                self._transactions([
                    ("S1", "2025-01-01 08:00", "X", "Y", 10.0, 0),
                ]),
                fit_scalers=True,
                history_transactions=pd.DataFrame(),
            )
        )
        model = TemporalHeterogeneousFraudGNN(
            account_input_dim=seed_graph["account"].x.shape[1],
            transaction_input_dim=seed_graph["transaction"].x.shape[1],
            hidden_dim=8,
            dropout=0.0,
        )

        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            model_path = folder / "model.pt"
            preprocessor_path = folder / "preprocessor.joblib"
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "account_input_dim": seed_graph["account"].x.shape[1],
                    "transaction_input_dim": seed_graph["transaction"].x.shape[1],
                    "hidden_dim": 8,
                    "dropout": 0.0,
                    "threshold": 0.5,
                },
                model_path,
            )
            joblib.dump(
                {
                    "account_scaler": account_scaler,
                    "transaction_scaler": transaction_scaler,
                    "feature_columns": feature_columns,
                },
                preprocessor_path,
            )

            with_future = score_transactions(
                targets,
                pd.concat([history, future], ignore_index=True),
                model_path=model_path,
                preprocessor_path=preprocessor_path,
                device="cpu",
            )
            without_future = score_transactions(
                targets,
                history,
                model_path=model_path,
                preprocessor_path=preprocessor_path,
                device="cpu",
            )

        self.assertTrue(torch.allclose(
            torch.tensor(with_future["thgnn_score"].to_numpy()),
            torch.tensor(without_future["thgnn_score"].to_numpy()),
        ))


if __name__ == "__main__":
    unittest.main()
