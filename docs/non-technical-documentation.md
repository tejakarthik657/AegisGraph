# Non-Technical Documentation

> **AegisGraph — Agentic AI Financial-Crime Investigation System**
> Plain-language guide for stakeholders, analysts, and compliance teams

---

## Table of Contents

1. [What Is AegisGraph?](#1-what-is-aegisgraph)
2. [The Problem We Solve](#2-the-problem-we-solve)
3. [How It Works — The Big Picture](#3-how-it-works--the-big-picture)
4. [The Investigation Workflow](#4-the-investigation-workflow)
5. [Understanding Risk Levels](#5-understanding-risk-levels)
6. [Human-in-the-Loop](#6-human-in-the-loop)
7. [The Audit Trail](#7-the-audit-trail)
8. [What the AI Actually Does](#8-what-the-ai-actually-does)
9. [Safety and Controls](#9-safety-and-controls)
10. [Frequently Asked Questions](#10-frequently-questions)

---

## 1. What Is AegisGraph?

AegisGraph is an intelligent system that helps financial institutions detect, investigate, and document suspicious financial activity — think money laundering, structuring, and other financial crimes.

It works like a highly trained compliance analyst that never sleeps:

- **Scans** millions of transactions to find suspicious patterns
- **Investigates** flagged accounts automatically
- **Researches** relevant regulations and compliance rules
- **Drafts** investigation reports
- **Logs** every step for auditors
- **Presents** findings to a human reviewer for final decision

---

## 2. The Problem We Solve

Financial crime investigators face an impossible volume of work:

- **Millions of transactions** flow through banks daily
- **Thousands of accounts** may need review
- **Manual investigation** takes hours per case
- **Patterns are hidden** across complex transaction networks
- **Regulations change** constantly and are hard to track

Traditional systems flag transactions but leave the hard investigative work to humans. AegisGraph automates the entire workflow — from detection to report drafting — while keeping humans in control of final decisions.

---

## 3. How It Works — The Big Picture

```
DETECT          INVESTIGATE          REASON          RETRIEVE          EXPLAIN          DRAFT          APPROVE          AUDIT
   │                │                 │                │                │              │               │               │
   ▼                ▼                 ▼                ▼                ▼              ▼               ▼               ▼
Scan all     AI agent examines     LLM reasons     System finds     System builds   Markdown       Human makes     Every action
accounts     the flagged account   about findings   relevant rules   evidence chain  report draft   approve/reject   is recorded
for risk     automatically                    from a knowledge                    ready for      or escalate      permanently
```

### Six Core Capabilities

| Capability | What It Means |
|------------|--------------|
| **Smart Detection** | Catches known money-laundering patterns like smurfing, layering, and funnel accounts |
| **Graph Intelligence** | Understands how accounts connect through transactions, not just individual transactions |
| **AI Investigation** | An AI agent plans and executes its own investigation strategy for each case |
| **Regulatory Knowledge** | Automatically finds the right compliance rules for each situation |
| **Explainable Findings** | Shows exactly *why* an account was flagged, with evidence |
| **Complete Audit Trail** | Records every decision, tool call, and finding for regulators |

---

## 4. The Investigation Workflow

When AegisGraph identifies a suspicious account, it automatically:

### Step 1: Alert Generation
The system evaluates every account against multiple risk signals and assigns a risk score. Accounts above the threshold become **alerts**.

### Step 2: Case Creation
A human reviewer clicks "Investigate" on an alert. The system creates a **case** with a unique ID.

### Step 3: Automated Investigation
The Investigation Agent takes over and:

1. **Loads the account profile** — Who owns it? Where? What type?
2. **Expands the network** — Maps all connected accounts (up to 3 hops away)
3. **Analyzes behavior** — How fast is money moving? How many connections?
4. **Runs pattern detectors** — Checks for known laundering typologies
5. **Computes risk scores** — Combines rule-based, ML, and graph signals
6. **Retrieves regulations** — Finds relevant compliance notes
7. **Builds evidence chain** — Documents every finding with reasoning
8. **Drafts a report** — Creates a structured markdown report

### Step 4: Human Review
The drafted report is presented to a human reviewer who can:

- **Approve** — The investigation is complete and accurate
- **Reject** — The findings are incorrect or insufficient
- **Escalate** — Immediate attention needed from senior compliance

### Step 5: Audit Recording
Every step — from alert creation to final decision — is recorded in an immutable audit log.

---

## 5. Understanding Risk Levels

Every account receives a risk score from 0% to 100%:

| Risk Level | Score Range | Action |
|------------|-------------|--------|
| **HIGH** | 60% - 100% | Immediate investigation required |
| **MEDIUM** | 30% - 59% | Review recommended |
| **LOW** | 0% - 29% | Standard monitoring |

### How Risk Is Calculated

AegisGraph uses three independent risk signals and combines them:

```
Final Risk = (30% Rule Score) + (40% ML Score) + (30% Graph Score)
```

- **Rule Score**: Does the account violate any compliance rules? (velocity, structuring, rapid movement)
- **ML Score**: What does the machine learning model predict? (trained on historical data)
- **Graph Score**: How connected is this account to other suspicious accounts?

This triple-signal approach reduces false positives and catches sophisticated schemes that single-method systems miss.

---

## 6. Human-in-the-Loop

AegisGraph is designed to **augment** human investigators, not replace them.

### What the System Does Automatically
- Data processing and feature computation
- Pattern detection and risk scoring
- Network expansion and visualization
- Report drafting and regulatory lookup

### What Humans Control
- **Case triage**: Which alerts to investigate first
- **Investigation review**: Approving, rejecting, or escalating findings
- **Final SAR filing**: Only humans can file Suspicious Activity Reports
- **Model oversight**: Monitoring ML performance and retraining decisions

### Role-Based Access

| Role | Capabilities |
|------|-------------|
| **Analyst** | View dashboard, investigate alerts, draft reports |
| **Senior Reviewer** | Review high-risk cases, approve escalations, four-eyes verification |

---

## 7. The Audit Trail

Every action in AegisGraph is permanently recorded:

| What's Logged | Example |
|---------------|---------|
| Pipeline events | "Pipeline rebuilt with adversarial data" |
| Alert events | "Alert created for ACC-001, risk level: HIGH" |
| Investigation steps | "Tool called: expand_network (3 hops)" |
| Agent decisions | "Agent chose to skip regulatory search" |
| Human decisions | "SeniorReviewer: APPROVED case CASE-A1B2C3" |
| System fallbacks | "Backfilled run_risk_models (agent did not call)" |

Audit records include:
- **Actor** (who or what performed the action)
- **Action** (what was done)
- **Timestamp** (when it happened)
- **Details** (relevant context)

This creates a complete, regulator-ready record of every investigation.

---

## 8. What the AI Actually Does

AegisGraph's AI has two modes:

### Mode 1: Deterministic (Default)
- Runs the same investigation steps for every case
- No external API keys needed
- Fully reproducible and auditable
- Works offline

### Mode 2: Agentic (With API Key)
When an LLM API key is configured, the AI becomes truly autonomous:

- **Plans its own investigation**: Decides which tools to call and in what order
- **Adapts per case**: A simple case gets a quick investigation; a complex case gets deep analysis
- **Skips unnecessary steps**: Doesn't waste time on low-risk accounts
- **Reasons about its choices**: Logs why it chose each step
- **Knows when to stop**: Calls `finish_investigation` when evidence is sufficient

**Example**: For a clear-cut structuring case, the agent might call just 3 tools. For a complex layered scheme, it might call 8+ tools, expanding the network further and searching multiple regulation categories.

### What the AI Never Does
- **Never fabricates evidence**: Every fact comes from an actual data query
- **Never makes final decisions**: Humans approve all findings
- **Never operates outside bounds**: Can only use approved tools
- **Never crashes the system**: Automatically falls back to deterministic mode on any failure

---

## 9. Safety and Controls

### Built-in Safeguards

1. **Tool Budget Cap**: Maximum 10 tool calls per investigation prevents runaway loops
2. **Required Field Backfill**: If the AI skips a critical check, the system fills it in automatically and logs the fallback
3. **Graceful Degradation**: If the LLM service fails, the system automatically switches to deterministic mode
4. **Fixed Tool Surface**: The AI can only call pre-approved tools — no arbitrary actions
5. **Evidence-Only Summaries**: The AI can only state facts it actually retrieved through tools
6. **Senior Review Pass**: High-risk cases trigger an additional AI review before human review

### Adversarial Testing

AegisGraph includes an adversarial simulator that attempts to create transaction networks that evade detection. This helps measure how robust the system is against sophisticated laundering techniques.

---

## 10. Frequently Asked Questions

### Does AegisGraph use real customer data?
No. All data is synthetically generated for demonstration and development purposes. No real customer information is used.

### Can the AI make decisions without human approval?
No. The AI investigates and drafts reports, but a human must approve, reject, or escalate every case. The AI never files reports or makes enforcement decisions.

### What happens if the AI service goes down?
The system automatically falls back to deterministic mode. Every investigation still completes successfully with the same tools — just without LLM-guided planning.

### Is this a production system?
AegisGraph is a fully functional reference implementation demonstrating the complete workflow. It is designed to be extended into production by swapping components (NetworkX → Neo4j, SQLite → PostgreSQL, etc.) without changing the overall architecture.

### How accurate is the ML model?
The RandomForest baseline achieves approximately:
- Precision: 60%
- Recall: 40%
- F1: 50%
- ROC-AUC: 80%

These are honest results on noisy, imbalanced synthetic data — a realistic starting point for improvement.

### What regulations does it check?
The system includes sample compliance notes covering structuring, smurfing, layering, rapid movement, and reporting requirements. In production, these would be replaced with actual regulatory texts.

### Is the audit trail regulator-ready?
The audit trail captures every investigation step with timestamps, actor identification, and detailed context. It is designed to meet standard compliance audit requirements, but you should consult your legal and compliance team for specific regulatory needs.

---

*This document is intended for non-technical stakeholders. For technical details, see [Technical Documentation](./technical-documentation.md).*
