# AegisGraph — Agentic AI Financial-Crime Investigation System

A working, end-to-end prototype of an autonomous financial-crime investigation
platform: **detect → investigate → reason → retrieve regulations → explain →
draft → human approve → audit.**

This is a fully functional reference implementation of every module described
in the project proposal — not a mockup. It runs entirely offline, with no
external services, GPUs, or API keys required.

---

## 1. What's actually in this zip

```
aegisgraph/
├── README.md                      ← you are here
├── docker-compose.yml              ← local dev stack (+ commented production stack)
├── backend/
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── app/
│   │   ├── config.py                    # all tunable parameters, one place
│   │   ├── data_generator.py            # Module 01 — synthetic data + injected typologies
│   │   ├── graph_store.py               # Module 02 — graph construction (NetworkX)
│   │   ├── feature_engineering.py       # Module 03 — behavioral feature computation
│   │   ├── rule_engine.py               # Module 04 — deterministic rule checks
│   │   ├── ml_risk_engine.py            # Module 05 — ML baseline (RandomForest)
│   │   ├── graph_propagation_model.py   # Module 06 — GNN-lite (NumPy message passing)
│   │   ├── pattern_detectors.py         # Module 07 — smurfing/layering/funnel/fan-out/cycles
│   │   ├── investigation_agent.py       # Module 08 — agentic LLM planner + deterministic fallback
│   │   ├── regulatory_rag.py            # Module 09 — TF-IDF retrieval over compliance notes
│   │   ├── explainability.py            # Module 10 — evidence chain + feature deviation
│   │   ├── report_generator.py          # Module 11 — structured report + markdown draft
│   │   ├── db.py                        # Module 12 — SQLite persistence (cases/alerts/audit)
│   │   ├── risk_fusion.py               # combines rule + ML + graph scores
│   │   ├── pipeline.py                  # orchestrates all modules end-to-end
│   │   └── main.py                      # FastAPI application (the API Gateway)
│   ├── data/
│   │   └── regulations/             # 5 original sample compliance reference notes (for RAG)
│   └── tests/
│       ├── test_pattern_detectors.py    # unit tests for each typology detector
│       ├── test_pipeline.py             # full pipeline integration test
│       └── test_agentic_investigation.py # agentic-mode tests (mocked LLM client)
├── frontend/
│   ├── index.html                   # Dashboard (alert queue, ML metrics)
│   ├── investigation.html           # Investigation workspace (the "killer demo" screen)
│   ├── app.js                       # all frontend logic, vanilla JS, no build step
│   └── styles.css
└── scripts/
    └── run_pipeline_demo.py         # run the whole system from the command line, no server
```

Every one of the 12 modules from the project's methodology section exists as
real, runnable, tested code — not a stub.

---

## 2. Quick start

### Option A — no Docker, fastest way to see it work

```bash
cd aegisgraph/backend
pip install -r requirements.txt --break-system-packages   # or use a venv
python3 ../scripts/run_pipeline_demo.py
```

This prints the entire pipeline running end-to-end in your terminal: data
generation → graph construction → rule/ML/graph detection → benchmarked ML
metrics → the agent investigating the top alert → a full markdown
investigation report → the complete audit trail.

### Option B — run the full web app (API + dashboard)

```bash
cd aegisgraph/backend
pip install -r requirements.txt --break-system-packages
uvicorn app.main:app --reload --port 8000
```

Then open **http://localhost:8000** in a browser. You'll see the dashboard
with the alert queue and benchmarked ML metrics; click "Investigate" on any
alert to open the full investigation workspace (graph visualization, agent
activity log, evidence panel, regulatory context, draft report, and
approve/reject/escalate buttons).

### Option C — Docker

```bash
cd aegisgraph
docker compose up --build
```

Then open **http://localhost:8000**.

### Running the tests

```bash
cd aegisgraph/backend
python3 tests/test_pattern_detectors.py
python3 tests/test_pipeline.py
python3 tests/test_agentic_investigation.py
# or, if pytest is installed:
pytest tests/ -v
```

---

## 3. How to read this as "the whole project"

Every section from the presentation deck maps directly onto working code:

| Presentation section | Where it lives |
|---|---|
| Dataset / data sources | `app/data_generator.py` — synthetic, with ground-truth typology labels |
| System architecture (graph → rules/ML/GNN → agent → RAG → LLM → report → human → audit) | `app/pipeline.py` orchestrates every module in that exact order |
| Rule engine | `app/rule_engine.py` |
| ML risk engine | `app/ml_risk_engine.py` (benchmarked, not assumed, metrics) |
| GNN multi-hop detector | `app/graph_propagation_model.py` |
| Pattern detectors (smurfing, layering, funnel, fan-out, cycles) | `app/pattern_detectors.py`, each independently unit-tested |
| Investigation Agent (agentic tool-calling) | `app/investigation_agent.py` |
| Regulatory RAG | `app/regulatory_rag.py` + `backend/data/regulations/*.txt` |
| Explainability | `app/explainability.py` |
| Report generation | `app/report_generator.py` |
| Human-in-the-loop review | `POST /api/cases/{case_id}/review` + the Approve/Reject/Escalate buttons in `investigation.html` |
| Audit ledger | `app/db.py` (`audit_log` table) + the Audit Ledger panel in the UI |

---

## 4. What you'll actually see when you run it

On a fresh run, the synthetic generator creates a randomized universe, so
exact numbers will vary slightly, but a typical run produces:

- ~400 accounts, ~2,350 transactions
- ~70-80 accounts raised as HIGH-risk alerts
- ML risk model benchmarked on a held-out test set at roughly:
  precision ≈ 0.6, recall ≈ 0.4, F1 ≈ 0.5, ROC-AUC ≈ 0.8, PR-AUC ≈ 0.7

**These numbers are intentionally imperfect.** This is a genuine classifier
trained on genuinely noisy, imbalanced synthetic data — not a number picked
to look impressive. That's the honest result of a RandomForest baseline on
this dataset, and it gives you a real starting point to improve from (better
features, the GNN-lite propagation model, more training data, hyperparameter
tuning) rather than an invented figure you'd have to defend under questioning.

Clicking "Investigate" on an alert triggers the real Investigation Agent,
which:
1. Loads the account profile
2. Expands the transaction network 2 hops out (capped at 60 nodes for a
   readable visualization)
3. Recomputes velocity/behavioral features
4. Runs the deterministic pattern detectors and reports what it actually found
5. Reports the rule + ML + graph fusion score
6. Retrieves the most relevant compliance reference notes for the detected
   pattern types (real TF-IDF retrieval, not hardcoded)
7. Generates an evidence chain and a structured, human-readable draft report
8. Logs every single one of these steps to the audit ledger, visible in the UI

---

## 5. Honest scope notes — what's simplified and why

This prototype makes deliberate engineering trade-offs to be **runnable
anywhere with zero external infrastructure**, while preserving the exact
interface you'd swap in for the production version described in the original
proposal. Every substitution below is called out explicitly in the code
(`Production Upgrade Path` comments in each module):

| Proposal component | This prototype uses | Why | Production upgrade |
|---|---|---|---|
| Neo4j graph database | NetworkX (in-memory graph) | Zero external services to run the demo | Swap `GraphStore` internals for Cypher queries against Neo4j; same public methods |
| PostgreSQL | SQLite | Zero setup | Change the connection string + add Alembic migrations |
| Qdrant vector DB + neural embeddings | scikit-learn TF-IDF + cosine similarity | No model downloads needed (sandboxed/offline-friendly) | Swap in `sentence-transformers` embeddings + Qdrant client behind the same `retrieve()` interface |
| PyTorch Geometric (GraphSAGE/GAT/HAN) | NumPy-based message-passing + LogisticRegression classifier ("GNN-lite") | No heavy ML framework installs, runs on CPU instantly | Replace `GraphPropagationModel` with a real `torch_geometric` model; same `fit`/`predict_proba` contract |
| Kafka streaming ingestion | Batch generation + REST API | Simpler to demo; the proposal itself recommends deferring Kafka to a later phase | Add a Kafka consumer that calls the same ingestion/feature functions per-message |
| Kubernetes | Docker Compose | The proposal itself recommends deferring K8s | Containerize each module as a separate service behind the same APIs |
| LLM-generated narrative | Deterministic template-based summary (LLM optional) | Works without any API key; deterministic and reproducible for grading/demo | Set the `ANTHROPIC_API_KEY` environment variable — `investigation_agent.py` will automatically call Claude to phrase the executive summary from the *already-gathered* structured evidence (it still can't invent facts) |

None of these substitutions change what the system *demonstrates*: real
graph-based multi-hop detection, real deterministic typology algorithms, a
real benchmarked ML model, a real agentic tool-calling investigation loop,
real retrieval-augmented regulatory grounding, real explainability, and a
real audited human-in-the-loop workflow.

---

## 6. Real agentic AI — how it actually works now

This is the part worth reading carefully, because it's the difference
between "an AI-sounding pipeline" and genuine agentic behavior.

**Without an API key (default):** the Investigation Agent runs a fixed,
deterministic sequence of tool calls — reproducible, auditable, and requires
no external service. This is what ships and runs out of the box.

**With `ANTHROPIC_API_KEY` set:** the Investigation Agent hands its entire
tool registry to Claude and lets the model decide, turn by turn, which tool
to call next based on what it has learned so far — including skipping tools
it judges unnecessary, calling tools in a different order, expanding the
network further only when it judges the case warrants it, and deciding for
itself when it has gathered enough evidence to stop (by calling a
`finish_investigation` tool). **Two different accounts can trigger two
completely different investigation paths** — that's the actual definition of
agentic behavior, not just "an LLM writes the final paragraph."

```bash
export ANTHROPIC_API_KEY=your_key_here
pip install anthropic --break-system-packages   # already in requirements.txt
uvicorn app.main:app --reload --port 8000
```

Every decision the model makes is written to the audit ledger in real time —
open the Audit Ledger panel on any investigated case and you'll see entries
like `agent_reasoning`, `agent_tool_call:expand_network`, and
`agent_decision:finish_investigation`, in whatever order the model actually
chose them, not a hardcoded order.

**Safety rails baked into the design (all verified by
`tests/test_agentic_investigation.py`):**
- A hard cap (`config.AGENT_MAX_TOOL_CALLS`, default 10) prevents a runaway
  loop if the model never calls `finish_investigation`.
- If the model's chosen path skips a field the report absolutely needs (e.g.
  it never calls `run_risk_models`), the system deterministically backfills
  it — and explicitly logs this as a `system_fallback:*` audit entry, so you
  can always see exactly what the AI itself investigated versus what the
  system had to supply afterward.
- If the `anthropic` package isn't installed, or the API call fails for any
  reason (network issue, rate limit, bad response), the agent automatically
  and silently falls back to the deterministic pipeline — the system never
  crashes or produces an incomplete investigation because of an LLM failure.
- The model can only ever call the tools in `AGENT_TOOL_SCHEMAS` — it cannot
  take any action outside that fixed, auditable surface, and every tool
  still executes the same deterministic Python function used in the
  non-agentic path. The LLM plans; it never fabricates evidence.

**Since this sandbox has no live API key, the agentic loop's control logic
was validated with a mocked Anthropic client** that returns the exact
response shapes the real API returns (`tests/test_agentic_investigation.py`)
— it proves the short-path-vs-long-path adaptive behavior, the budget cap,
the fallback logging, and the graceful degradation all work correctly. The
moment you add a real key, the exact same code calls the real model instead
of the mock.

## 7. Enabling only the narrative phrasing (no full agentic planning)

If you'd rather keep the fixed deterministic pipeline but still want Claude
to phrase the final executive-summary paragraph, that behavior has been
superseded by full agentic mode — setting `ANTHROPIC_API_KEY` now enables
real planning, which is strictly more capable. There is no separate
"narrative-only" toggle anymore; see section 6 above.

## 8. Extending the project

Some natural next steps, each of which slots into the existing module
boundaries without a redesign:

- **Adversarial evaluation**: build a second generator that tries to design
  transaction networks that evade the current detectors, and measure how
  detection performance degrades — a strong basis for a research contribution.
- **Unknown-pattern discovery**: add an unsupervised anomaly detector (e.g.
  graph community detection + outlier scoring) alongside the known-typology
  detectors, for structures that don't match any predefined pattern.
- **Real GNN**: swap `graph_propagation_model.py` for a PyTorch Geometric
  GraphSAGE/GAT model once you have GPU access.
- **Multi-agent specialization**: split `InvestigationAgent` into a
  Supervisor + Graph/Behavior/Regulatory sub-agents, as described in the
  original design's hierarchical agent section.
- **Streaming ingestion**: add a Kafka consumer that feeds `data_generator`
  -style transactions into `graph_store.build()` incrementally instead of in
  one batch.

---

## 9. Important disclaimer

All data in this project is synthetic. No real customer, account, or
transaction data is used anywhere. The regulatory reference notes in
`backend/data/regulations/` are original illustrative text written for this
project — they are **not** verbatim excerpts of any actual law or regulation
and must not be treated as legal guidance. Any report generated by this
system is explicitly labeled a draft for human compliance review and is not,
and must never be treated as, a filed Suspicious Activity Report (SAR).
#   A e g i s G r a p h  
 