# AegisGraph Architecture

## System Overview

AegisGraph is a full-stack Anti-Money Laundering (AML) and fraud detection platform built with a modular pipeline architecture. The system combines rule-based detection, machine learning models, graph neural networks, and an AI-powered investigation agent.

## Architecture Diagram

```
┌──────────────────────────────────────────────────────────┐
│                    FRONTEND (React + Vite)                │
│  Dashboard │ Alerts │ Cases │ Graph │ Transactions       │
└──────────────────────┬───────────────────────────────────┘
                       │ REST API (HTTP)
┌──────────────────────┴───────────────────────────────────┐
│                  BACKEND (FastAPI + Python)               │
│                                                           │
│  ┌─────────────┐  ┌──────────────┐  ┌──────────────────┐ │
│  │   MODULE 1  │  │   MODULE 2   │  │    MODULE 3      │ │
│  │ Data Pipeline│  │ Baseline RF  │  │  ML Extensions   │ │
│  │ Generation  │  │   Model      │  │  TGN / HGNN /    │ │
│  │ Ingestion   │  │ Features     │  │  Hybrid          │ │
│  │ Graph Build │  │ Training     │  │                   │ │
│  └─────────────┘  └──────────────┘  └──────────────────┘ │
│                                                           │
│  ┌─────────────┐  ┌──────────────┐  ┌──────────────────┐ │
│  │   MODULE 4  │  │   MODULE 5   │  │    MODULE 6      │ │
│  │  Detection  │  │   Alerts     │  │   Agent          │ │
│  │  Rules      │  │  Generation  │  │   Investigation  │ │
│  │  Patterns   │  │  Service     │  │   Tools / RAG    │ │
│  │  Fusion     │  │  Queue       │  │   Guardrails     │ │
│  └─────────────┘  └──────────────┘  └──────────────────┘ │
│                                                           │
│  ┌─────────────┐  ┌──────────────┐  ┌──────────────────┐ │
│  │   MODULE 7  │  │ Investigation│  │ Case Management  │ │
│  │  Evidence   │  │  Build Cases │  │  State Machine   │ │
│  │  Reporting  │  │  Dossiers    │  │  Reviews         │ │
│  │  Narrative  │  │  Evidence    │  │  Decisions       │ │
│  └─────────────┘  └──────────────┘  └──────────────────┘ │
│                                                           │
│  ┌───────────────────────────────────────────────────────┐│
│  │                  RISK FUSION                          ││
│  │   RF + TGN + Rules → Combined Risk Score → Category   ││
│  └───────────────────────────────────────────────────────┘│
└──────────────────────────────────────────────────────────┘
                       │
         ┌─────────────┴─────────────┐
         │   DATA LAYER (CSV/GraphML) │
         │  raw/ processed/ results/  │
         └───────────────────────────┘
```

## Module Descriptions

### Module 1: Data Pipeline
- **Generator**: Synthetic customer, account, and transaction data with injected fraud typologies
- **Ingestion**: Validation, normalization, and deduplication
- **Graph Builder**: NetworkX graph construction from transaction data

### Module 2: Baseline Model
- **Model**: Random Forest classifier with balanced class weights
- **Features**: Engineered features from transaction attributes
- **Training**: 80/20 chronological train/test split

### Module 3: ML Extensions
- **TGN**: Temporal Graph Network with memory and message passing
- **Heterogeneous GNN**: Entity-type-aware graph neural network
- **Hybrid**: Ensemble combining TGN and HGNN outputs
- **Evaluation**: Model comparison and metrics

### Module 4: Detection
- **Rules**: Velocity break, structuring, rapid holding, high-risk country
- **Patterns**: Smurfing, layering, funnel, fan-out, cycle detection
- **Graph Scoring**: Neighborhood risk assessment
- **Fusion**: Weighted combination of all detection signals

### Module 5: Alerts
- **Generation**: Alert creation from risk-scored transactions
- **Service**: Alert lifecycle management
- **Queue**: Priority queue for alert processing

### Module 6: Investigation Agent
- **State**: Agent state tracking through investigation workflow
- **Graph Definition**: Directed workflow graph of investigation steps
- **Tools**: Account lookup, graph analysis, velocity, pattern, risk, RAG
- **Guardrails**: Safety constraints on agent decisions

### Module 7: Evidence Reporting
- **Evidence Chain**: Structured evidence collection
- **Report Generator**: Investigation report creation
- **Narrative Generator**: AI-generated investigation summaries

## Technology Stack

| Layer | Technology |
|-------|-----------|
| Frontend | React 19, Vite 8, React Router, Axios |
| Backend | FastAPI, Python 3.13, Pydantic |
| ML | scikit-learn, PyTorch, NetworkX |
| Data | Pandas, NumPy, Faker |
| Graph Viz | react-force-graph-2d/3d |

## Data Flow

1. **Generate** → Synthetic data creation with fraud injection
2. **Ingest** → Validate, normalize, deduplicate
3. **Build Graph** → Create NetworkX transaction graph
4. **Train Models** → RF baseline, TGN, HGNN training
5. **Evaluate Rules** → AML rule-based detection
6. **Risk Fusion** → Combine ML + rules into risk scores
7. **Generate Alerts** → Create alerts from high-risk transactions
8. **Build Cases** → Investigation case creation
9. **Investigate** → Agent-assisted investigation
10. **Report** → Evidence-based reporting
