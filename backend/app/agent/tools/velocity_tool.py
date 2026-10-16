"""AegisGraph — Velocity Analysis Tool."""

from __future__ import annotations
import pandas as pd
from app.utils.config import CLEAN_TRANSACTIONS_FILE


def analyze_velocity(account_id: str, window_hours: int = 24) -> dict:
    """Analyze transaction velocity for an account."""
    df = pd.read_csv(CLEAN_TRANSACTIONS_FILE)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    sent = df[df["sender_account"] == account_id].sort_values("timestamp")
    if sent.empty:
        return {"account_id": account_id, "transaction_count": 0, "max_velocity": 0}
    window = pd.Timedelta(hours=window_hours)
    max_velocity = 0
    for i, row in sent.iterrows():
        count = len(sent[(sent["timestamp"] >= row["timestamp"] - window) & (sent["timestamp"] <= row["timestamp"])])
        max_velocity = max(max_velocity, count)
    return {
        "account_id": account_id,
        "window_hours": window_hours,
        "total_outgoing": len(sent),
        "max_velocity": max_velocity,
    }
