# AegisGraph — Model Evaluation

## Overview

AegisGraph evaluates fraud detection models using a consistent 80/20 chronological train/test split. This document describes the evaluation methodology and metrics.

## Models

| Model | Type | Description |
|-------|------|-------------|
| Random Forest (RF) | Baseline | Sklearn RF classifier with engineered features |
| Temporal GNN (TGN) | Deep Learning | Temporal Graph Network with memory modules |
| Heterogeneous GNN | Deep Learning | Entity-type-aware graph neural network |
| Hybrid | Ensemble | Weighted combination of TGN + HGNN |
| Risk Fusion | Ensemble | RF + TGN + Rules combined score |

## Evaluation Metrics

### Classification Metrics
- **Accuracy**: Overall correct predictions / total predictions
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1 Score**: Harmonic mean of precision and recall
- **AUC-ROC**: Area under the Receiver Operating Characteristic curve
- **AUC-PR**: Area under the Precision-Recall curve

### Business Metrics
- **Alert Volume**: Number of alerts generated
- **False Positive Rate**: Non-fraud alerts / total alerts
- **Detection Rate**: Caught fraud / total fraud transactions
- **Processing Time**: Inference latency per transaction

## Evaluation Methodology

### Chronological Split
```
[── Training Data (80%) ──|── Test Data (20%) ──]
       Older transactions      Newer transactions
```

### Class Imbalance
- Fraud rate: ~10% of transactions (injected)
- Mitigation: Balanced class weights, SMOTE-aware metrics
- Primary metric: F1 Score (better for imbalanced data)

## Risk Fusion Formula

```
risk_score = w_rf × rf_probability
           + w_tgn × tgn_probability
           + w_rules × rule_score

where w_rf = w_tgn = w_rules = 1/3
```

### Risk Categories

| Category | Score Range | Action |
|----------|------------|--------|
| HIGH | ≥ 0.75 | Generate alert, priority review |
| MEDIUM | 0.50 – 0.74 | Generate alert, standard review |
| LOW | 0.25 – 0.49 | Monitor, no alert |
| VERY_LOW | < 0.25 | No action required |

## Output Files

| File | Description |
|------|-------------|
| `baseline_predictions.csv` | RF model predictions |
| `temporal_tgn_predictions.csv` | TGN model predictions |
| `heterogeneous_gnn_predictions.csv` | HGNN model predictions |
| `hybrid_predictions.csv` | Hybrid ensemble predictions |
| `risk_fusion_predictions.csv` | Final fused risk scores |
| `*_metrics.csv` | Per-model evaluation metrics |
