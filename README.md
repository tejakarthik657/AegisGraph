# AegisGraph

**Agentic AI Financial-Crime Investigation System**

A fully functional, end-to-end prototype of an autonomous financial-crime investigation platform: **detect → investigate → reason → retrieve regulations → explain → draft → human approve → audit**.

This is a working reference implementation of every module described in the project proposal — not a mockup. It runs entirely offline, with no external services, GPUs, or API keys required.

---

## What It Does

AegisGraph provides a complete pipeline for financial-crime investigation:

1. **Synthetic data generation** with injected typologies (smurfing, layering, funnel, fan-out, cycles)
2. **Graph construction** using NetworkX for transaction network analysis
3. **Multi-signal detection**: deterministic rules, ML baselines, and GNN-lite propagation
4. **Pattern detection** for known money-laundering typologies
5. **Agentic investigation** — an LLM-driven agent that plans and executes its own investigation path
6. **Regulatory RAG** — retrieves relevant compliance notes using TF-IDF
7. **Explainability** — evidence chains and feature deviation analysis
8. **Report generation** — structured markdown drafts for human review
9. **Human-in-the-loop review** — approve, reject, or escalate with full audit trail

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Backend API | FastAPI + Python 3.13 |
| Frontend | Vite 8 + React 19 + Vanilla CSS |
| Graph Analysis | NetworkX + react-force-graph |
| ML / Risk Engine | scikit-learn (RandomForest baseline) |
| Deep Learning | PyTorch + TGN + Heterogeneous GNN |
| Data Storage | CSV / GraphML (MVP) → PostgreSQL (production) |
| Vector Search | TF-IDF (MVP) → Qdrant (production) |
| Testing | pytest + unittest |

---

## Project Structure

```
aegisgraph/
├── README.md                      ← you are here
├── docker-compose.yml              ← local dev stack
├── backend/
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── app/
│   │   ├── main.py                      # FastAPI application
│   │   ├── pipeline.py                  # End-to-end orchestration
│   │   ├── config.py                    # Tunable parameters
│   │   ├── data_generator.py            # Synthetic data + typology injection
│   │   ├── graph_store.py               # Graph construction (NetworkX)
│   │   ├── feature_engineering.py       # Behavioral feature computation
│   │   ├── rule_engine.py               # Deterministic rule checks
│   │   ├── ml_risk_engine.py            # ML baseline (RandomForest)
│   │   ├── graph_propagation_model.py   # GNN-lite (message passing)
│   │   ├── pattern_detectors.py         # Typology detection algorithms
│   │   ├── investigation_agent.py       # Agentic LLM planner
│   │   ├── regulatory_rag.py            # TF-IDF retrieval over compliance notes
│   │   ├── explainability.py            # Evidence chain + feature deviation
│   │   ├── report_generator.py          # Structured report + markdown draft
│   │   ├── db.py                        # SQLite persistence (cases/alerts/audit)
│   │   └── risk_fusion.py               # Combines rule + ML + graph scores
│   ├── data/regulations/             # Compliance reference notes for RAG
│   └── tests/                         # Unit and integration tests
├── frontend/
│   ├── app/                           # Next.js app directory
│   ├── components/                    # React components
│   ├── package.json
│   └── next.config.mjs
└── scripts/
    └── run_pipeline_demo.py           # CLI demo, no server required
```

---

## Quick Start

### Option A — CLI Demo (fastest)

Run the entire pipeline from the command line:

```bash
cd backend
pip install -r requirements.txt
python3 ../scripts/run_pipeline_demo.py
```

This prints the full pipeline: data generation → graph construction → detection → agent investigation → markdown report → audit trail.

### Option B — Full Web Application

```bash
cd backend
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

Open **http://localhost:8000** to access the dashboard.

### Option C — Docker

```bash
docker compose up --build
```

Then open **http://localhost:8000**.

---

## Running Tests

```bash
cd backend
pytest tests/ -v
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        FastAPI Gateway                          │
├──────────┬──────────┬──────────┬──────────┬──────────┬──────────┤
│  Data    │   Graph  │ Feature  │  Rule +  │   ML +   │ Pattern  │
│ Generator│  Store   │Engineering│  Graph   │ GNN-lite │Detectors │
├──────────┴──────────┴──────────┤ Fusion   ├──────────┴──────────┤
│        Investigation Agent       │  Engine  │   Regulatory RAG   │
├─────────────────────────────────┴──────────┴───────────────────┤
│              Explainability + Report Generator                   │
├─────────────────────────────────────────────────────────────────┤
│         Human Review (Approve / Reject / Escalate)               │
├─────────────────────────────────────────────────────────────────┤
│                    Audit Ledger (SQLite)                         │
└─────────────────────────────────────────────────────────────────┘
```

---

## Agentic AI Investigation

**Without an API key (default):** The Investigation Agent runs a fixed, deterministic sequence of tool calls — reproducible, auditable, and requires no external service.

**With `ANTHROPIC_API_KEY` set:** The agent hands its tool registry to Claude and lets the model decide, turn by turn, which tool to call next based on what it has learned. Two different accounts can trigger two completely different investigation paths.

Safety features:
- Hard cap on tool calls prevents runaway loops
- Deterministic backfill for required fields the agent skipped
- Automatic fallback to deterministic pipeline on any LLM failure
- The model can only call tools within a fixed, auditable surface

Every agent decision is logged to the audit ledger in real time.

---

## Typical Run Output

On a fresh run, the synthetic generator produces:

- ~400 accounts, ~2,350 transactions
- ~70-80 HIGH-risk alerts
- ML model benchmarks (held-out test set): Precision ≈ 0.6, Recall ≈ 0.4, F1 ≈ 0.5, ROC-AUC ≈ 0.8, PR-AUC ≈ 0.7

These are real results from a RandomForest baseline on noisy, imbalanced synthetic data — not polished numbers.

---

## Production Upgrade Path

| Component | This Prototype | Production |
|-----------|---------------|------------|
| Graph Database | NetworkX (in-memory) | Neo4j |
| Primary Database | SQLite | PostgreSQL |
| Vector Search | TF-IDF + cosine | Qdrant + neural embeddings |
| GNN | NumPy message passing | PyTorch Geometric |
| Streaming | Batch + REST | Kafka |
| Orchestration | Docker Compose | Kubernetes |

Each module exposes a stable public interface, so upgrades are drop-in replacements without redesign.

---

## Disclaimer

All data in this project is synthetic. No real customer, account, or transaction data is used. The regulatory reference notes are original illustrative text and are **not** verbatim excerpts of any actual law or regulation. Any generated report is explicitly labeled a draft for human compliance review and is not, and must never be treated as, a filed Suspicious Activity Report (SAR).

py
cd backend
.\venv312\Scripts\Activate.ps1
python -m uvicorn app.main:app --reload