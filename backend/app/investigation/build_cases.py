from pathlib import Path

import pandas as pd


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]

RESULTS_DIR = PROJECT_ROOT / "data" / "results"

INPUT_FILE = RESULTS_DIR / "alerts.csv"
OUTPUT_FILE = RESULTS_DIR / "investigation_cases.csv"


# ============================================================
# LOAD ALERTS
# ============================================================

def load_alerts():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Alert file not found:\n{INPUT_FILE}"
        )

    df = pd.read_csv(INPUT_FILE)

    required_columns = [
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

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "Missing required alert columns:\n"
            + "\n".join(missing_columns)
        )

    return df


# ============================================================
# DETERMINE CASE PRIORITY
# ============================================================

def determine_case_priority(row):
    if row["risk_category"] == "HIGH":
        return "HIGH"

    if row["risk_category"] == "MEDIUM":
        return "MEDIUM"

    return "LOW"


# ============================================================
# BUILD EVIDENCE SUMMARY
# ============================================================

def build_evidence_summary(row):
    evidence = []

    evidence.append(
        f"Risk score: {row['risk_score']:.4f}"
    )

    evidence.append(
        f"RF probability: {row['rf_probability']:.4f}"
    )

    evidence.append(
        f"TGN probability: {row['tgn_probability']:.4f}"
    )

    evidence.append(
        f"Rule score: {row['rule_score']:.4f}"
    )

    if row["triggered_rules"]:
        evidence.append(
            f"Triggered rules: {row['triggered_rules']}"
        )
    else:
        evidence.append(
            "Triggered rules: none"
        )

    return " | ".join(evidence)


# ============================================================
# BUILD INVESTIGATION CASES
# ============================================================

def build_cases(alerts):
    cases = alerts.copy()

    cases["case_id"] = [
        f"CASE{index:06d}"
        for index in range(1, len(cases) + 1)
    ]

    cases["case_priority"] = cases.apply(
        determine_case_priority,
        axis=1,
    )

    cases["case_status"] = "OPEN"

    cases["evidence_summary"] = cases.apply(
        build_evidence_summary,
        axis=1,
    )

    return cases


# ============================================================
# BUILD FINAL CASE TABLE
# ============================================================

def build_case_table(cases):
    case_columns = [
        "case_id",
        "alert_id",
        "transaction_id",
        "timestamp",
        "sender_account",
        "receiver_account",
        "amount",
        "risk_score",
        "risk_category",
        "case_priority",
        "case_status",
        "rf_probability",
        "tgn_probability",
        "rule_score",
        "triggered_rules",
        "large_transaction",
        "structuring",
        "rapid_movement",
        "fan_out",
        "circular_movement",
        "evidence_summary",
        "is_fraud",
        "typology",
    ]

    return cases[case_columns]


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 70)
    print("AEGISGRAPH INVESTIGATION CASE BUILDER")
    print("=" * 70)

    alerts = load_alerts()

    print(f"Alerts loaded: {len(alerts)}")

    cases = build_cases(alerts)

    case_df = build_case_table(cases)

    print()
    print("Investigation case summary:")
    print(f"  Total cases: {len(case_df)}")

    print()
    print("Case priority distribution:")
    print(
        case_df["case_priority"]
        .value_counts()
        .sort_index()
    )

    print()
    print("Case status distribution:")
    print(
        case_df["case_status"]
        .value_counts()
        .sort_index()
    )

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    duplicate_case_ids = case_df["case_id"].duplicated().sum()
    duplicate_alert_ids = case_df["alert_id"].duplicated().sum()

    print()
    print("Validation:")
    print(f"  Duplicate case IDs: {duplicate_case_ids}")
    print(f"  Duplicate alert IDs: {duplicate_alert_ids}")

    if duplicate_case_ids > 0:
        raise ValueError("Duplicate case IDs detected.")

    if duplicate_alert_ids > 0:
        raise ValueError("Duplicate alert IDs detected.")

    # --------------------------------------------------------
    # Save investigation cases
    # --------------------------------------------------------

    case_df.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print()
    print("Saved to:")
    print(OUTPUT_FILE)

    print("=" * 70)


if __name__ == "__main__":
    main()