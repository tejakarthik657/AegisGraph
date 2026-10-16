"""AegisGraph — Account Lookup Tool."""

from __future__ import annotations
import pandas as pd
from app.utils.config import CLEAN_TRANSACTIONS_FILE


def lookup_account(account_id: str) -> dict:
    """Look up account transaction history and stats."""
    df = pd.read_csv(CLEAN_TRANSACTIONS_FILE)
    sent = df[df["sender_account"] == account_id]
    received = df[df["receiver_account"] == account_id]
    return {
        "account_id": account_id,
        "total_sent": len(sent),
        "total_received": len(received),
        "total_amount_sent": float(sent["amount"].sum()) if not sent.empty else 0.0,
        "total_amount_received": float(received["amount"].sum()) if not received.empty else 0.0,
        "unique_counterparties_sent": int(sent["receiver_account"].nunique()) if not sent.empty else 0,
        "unique_counterparties_received": int(received["sender_account"].nunique()) if not received.empty else 0,
    }
