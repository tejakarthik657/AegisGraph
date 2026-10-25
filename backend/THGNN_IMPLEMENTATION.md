# AegisGraph THGNN implementation (first integration)

## What was added

- `app/ml_extensions/thgnn/model.py`: typed account/transaction message passing with an explicit time-feature encoder.
- `app/ml_extensions/thgnn/preprocessing.py`: reuses the existing heterogeneous graph builder and adds relative event time and log time-gap features.
- `app/ml_extensions/thgnn/train.py`: chronological 80/20 split, training, test scoring, and Precision/Recall/F1/ROC-AUC/PR-AUC output.
- `app/ml_extensions/thgnn/inference.py`: loads a trained checkpoint and scores new rows against earlier historical transactions.
- `GET /api/thgnn/predictions` and `GET /api/thgnn/metrics`: expose saved predictions and evaluation metrics after training.
- The main `app/risk/risk_fusion.py` pipeline optionally merges `thgnn_predictions.csv` by transaction ID. When that file exists, THGNN contributes 25% of the risk score and the existing RF/TGN/AML-rule blend contributes the remaining 75%. Without the file, the original three-signal fusion is unchanged.
- The lower-level `app/detection/fusion/risk_fusion.py` also accepts an optional `thgnn_score` input column.
- `requirements-gnn.txt`: optional PyTorch and PyTorch Geometric dependencies.

## Install and run

From the `backend` directory, first install the normal project requirements. Then install the optional GNN dependencies:

```bash
pip install -r requirements.txt
pip install -r requirements-gnn.txt
```

For a CUDA-enabled machine, install the matching PyTorch build for your CUDA version using the official PyTorch installation selector before installing PyG.

Generate the cleaned transaction data if it does not already exist, then train:

```bash
python -m app.data.run_pipeline
python -m app.ml_extensions.thgnn.train
```

Outputs are saved under the project-level `data/results/` directory:

- `thgnn_model.pt`
- `thgnn_preprocessor.joblib`
- `thgnn_predictions.csv`
- `thgnn_metrics.csv`

Start the API from the backend directory with the project's usual command, for example:

```bash
uvicorn app.main:app --reload
```

After training, check `/api/thgnn/metrics` and `/api/thgnn/predictions`.

## Important limitations

This is an initial, temporal-feature-aware heterogeneous GNN prototype, not a reproduction of one specific published THGNN paper. It represents account and transaction node types and adds time-gap features. Its test evaluation uses a fixed training-history snapshot for each test event; it does not update graph memory after each test prediction. Metrics must be measured on your generated data and compared with the baseline before claiming an improvement.

To include THGNN in the main risk-fusion result, run the THGNN trainer before running `python -m app.risk.risk_fusion`. The main pipeline detects `data/results/thgnn_predictions.csv` and merges by transaction ID. Existing runs without that file retain the original three-signal behavior. The lower-level fusion helper requires a `thgnn_score` column to use its optional THGNN signal.
