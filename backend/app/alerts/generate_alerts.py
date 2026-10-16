from pathlib import Path

import pandas as pd


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]

RESULTS_DIR = PROJECT_ROOT / "data" / "results"

INPUT_FILE = RESULTS_DIR / "risk_fusion_predictions.csv"
OUTPUT_FILE = RESULTS_DIR / "alerts.csv"


# ============================================================
# ALERT CONFIGURATION
# ============================================================

HIGH_RISK_THRESHOLD = 0.75
MEDIUM_RISK_THRESHOLD = 0.50


# ============================================================
# LOAD RISK FUSION RESULTS
# ============================================================

def load_risk_predictions():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Risk fusion file not found:\n{INPUT_FILE}"
        )

    df = pd.read_csv(INPUT_FILE)

    required_columns = [
        "transaction_id",
        "timestamp",
        "sender_account",
        "receiver_account",
        "amount",
        "is_fraud",
        "typology",
        "rf_probability",
        "tgn_probability",
        "rule_score",
        "large_transaction",
        "structuring",
        "rapid_movement",
        "fan_out",
        "circular_movement",
        "risk_score",
        "risk_category",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "Missing required columns:\n"
            + "\n".join(missing_columns)
        )

    return df


# ============================================================
# GENERATE ALERT ID
# ============================================================

def generate_alert_id(index):
    return f"ALT{index:06d}"


# ============================================================
# DETERMINE ALERT STATUS
# ============================================================

def determine_alert_status(row):
    """
    Initial prototype alert logic.

    HIGH risk transactions become alerts.

    MEDIUM risk transactions are also included so that the
    investigation layer can later decide how they should be
    handled.

    VERY_LOW and LOW transactions do not become alerts.
    """

    if row["risk_category"] in {"HIGH", "MEDIUM"}:
        return "OPEN"

    return "NO_ALERT"


# ============================================================
# GET TRIGGERED RULES
# ============================================================

def get_triggered_rules(row):
    rules = []

    if bool(row["large_transaction"]):
        rules.append("large_transaction")

    if bool(row["structuring"]):
        rules.append("structuring")

    if bool(row["rapid_movement"]):
        rules.append("rapid_movement")

    if bool(row["fan_out"]):
        rules.append("fan_out")

    if bool(row["circular_movement"]):
        rules.append("circular_movement")

    if not rules:
        return ""

    return "|".join(rules)


# ============================================================
# GENERATE ALERTS
# ============================================================

def generate_alerts(df):
    alerts = df.copy()

    alerts["alert_status"] = alerts.apply(
        determine_alert_status,
        axis=1,
    )

    alerts["triggered_rules"] = alerts.apply(
        get_triggered_rules,
        axis=1,
    )

    alerts["alert_id"] = [
        generate_alert_id(index)
        for index in range(1, len(alerts) + 1)
    ]

    return alerts


# ============================================================
# BUILD FINAL ALERT TABLE
# ============================================================

def build_alert_table(df):
    alert_df = df[
        df["alert_status"] == "OPEN"
    ].copy()

    alert_columns = [
        "alert_id",
        "transaction_id",
        "timestamp",
        "sender_account",
        "receiver_account",
        "amount",
        "risk_score",
        "risk_category",
        "rf_probability",
        "tgn_probability",
        "rule_score",
        "triggered_rules",
        "large_transaction",
        "structuring",
        "rapid_movement",
        "fan_out",
        "circular_movement",
        "alert_status",
        "is_fraud",
        "typology",
    ]

    alert_df = alert_df[alert_columns]

    return alert_df


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 70)
    print("AEGISGRAPH ALERT GENERATION")
    print("=" * 70)

    risk_df = load_risk_predictions()

    print(f"Risk fusion rows loaded: {len(risk_df)}")

    generated_df = generate_alerts(risk_df)

    alert_df = build_alert_table(generated_df)

    print()
    print("Alert generation summary:")
    print(f"  Total transactions: {len(generated_df)}")
    print(f"  Open alerts: {len(alert_df)}")
    print(
        f"  Non-alert transactions: "
        f"{len(generated_df) - len(alert_df)}"
    )

    print()
    print("Alert risk distribution:")

    if len(alert_df) > 0:
        print(
            alert_df["risk_category"]
            .value_counts()
            .sort_index()
        )
    else:
        print("No alerts generated.")

    print()
    print("Triggered-rule summary:")

    if len(alert_df) > 0:
        rule_counts = {
            "large_transaction": int(
                alert_df["large_transaction"].sum()
            ),
            "structuring": int(
                alert_df["structuring"].sum()
            ),
            "rapid_movement": int(
                alert_df["rapid_movement"].sum()
            ),
            "fan_out": int(
                alert_df["fan_out"].sum()
            ),
            "circular_movement": int(
                alert_df["circular_movement"].sum()
            ),
        }

        for rule_name, count in rule_counts.items():
            print(f"  {rule_name}: {count}")
    else:
        print("No triggered rules.")

    # --------------------------------------------------------
    # Save alerts
    # --------------------------------------------------------

    alert_df.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print()
    print("Saved to:")
    print(OUTPUT_FILE)

    print("=" * 70)


if __name__ == "__main__":
    main()