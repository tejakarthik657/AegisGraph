"""AegisGraph — Regulatory Corpus Retriever."""

from __future__ import annotations
from pathlib import Path

CORPUS_DIR = Path(__file__).parent / "regulatory_corpus"

# Pre-built regulatory knowledge base (simplified)
REGULATORY_KNOWLEDGE = [
    {"id": "BSA-001", "source": "BSA/AML Manual",
     "topic": "SAR Filing",
     "content": "Financial institutions must file SARs for suspicious transactions over $5,000."},
    {"id": "BSA-002", "source": "BSA/AML Manual",
     "topic": "CTR Filing",
     "content": "CTRs are required for cash transactions exceeding $10,000."},
    {"id": "FATF-001", "source": "FATF Recommendations",
     "topic": "Customer Due Diligence",
     "content": "Enhanced due diligence required for high-risk customers and jurisdictions."},
    {"id": "FATF-002", "source": "FATF Recommendations",
     "topic": "Wire Transfers",
     "content": "Originator and beneficiary information must accompany wire transfers."},
]


def retrieve(query: str, top_k: int = 3) -> list[dict]:
    """Retrieve relevant regulatory documents (keyword-based)."""
    query_lower = query.lower()
    scored = []
    for doc in REGULATORY_KNOWLEDGE:
        score = sum(1 for word in query_lower.split() if word in doc["content"].lower())
        scored.append((score, doc))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [doc for _, doc in scored[:top_k]]
