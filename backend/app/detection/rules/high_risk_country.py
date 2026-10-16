"""
AegisGraph — High Risk Country Detection Rule.

Flags transactions involving jurisdictions known
for elevated AML risk.
"""

from __future__ import annotations

import pandas as pd


# FATF and high-risk jurisdictions list (simplified)
HIGH_RISK_COUNTRIES = {
    "IR",   # Iran
    "KP",   # North Korea
    "MM",   # Myanmar
    "SY",   # Syria
    "YE",   # Yemen
    "AF",   # Afghanistan
    "PK",   # Pakistan
    "NG",   # Nigeria
    "VU",   # Vanuatu
    "PA",   # Panama
}


def detect_high_risk_country(
    transactions: pd.DataFrame,
    high_risk_set: set[str] | None = None,
) -> pd.DataFrame:
    """
    Flag transactions where sender or receiver country
    is in the high-risk set.

    Parameters
    ----------
    transactions:
        Must contain: sender_country, receiver_country.

    high_risk_set:
        Optional override of the default country set.

    Returns
    -------
    DataFrame with an added `high_risk_country` boolean column.
    """

    df = transactions.copy()

    if high_risk_set is None:
        high_risk_set = HIGH_RISK_COUNTRIES

    # Normalize country codes
    for col in ["sender_country", "receiver_country"]:
        if col in df.columns:
            df[col] = (
                df[col]
                .astype(str)
                .str.strip()
                .str.upper()
            )

    sender_match = False
    receiver_match = False

    if "sender_country" in df.columns:
        sender_match = df["sender_country"].isin(
            high_risk_set
        )

    if "receiver_country" in df.columns:
        receiver_match = df["receiver_country"].isin(
            high_risk_set
        )

    df["high_risk_country"] = (
        (sender_match | receiver_match)
    ).astype(int)

    return df
