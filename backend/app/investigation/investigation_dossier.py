from pathlib import Path

import pandas as pd


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]

RESULTS_DIR = PROJECT_ROOT / "data" / "results"

CASE_REVIEW_FILE = RESULTS_DIR / "case_reviews.csv"
GRAPH_EVIDENCE_FILE = RESULTS_DIR / "case_graph_evidence.csv"

OUTPUT_FILE = RESULTS_DIR / "investigation_dossiers.csv"


# ============================================================
# LOAD CASE REVIEWS
# ============================================================

def load_case_reviews():
    if not CASE_REVIEW_FILE.exists():
        raise FileNotFoundError(
            f"Case review file not found:\n{CASE_REVIEW_FILE}"
        )

    df = pd.read_csv(CASE_REVIEW_FILE)

    required_columns = [
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
        "review_status",
        "investigator_decision",
        "investigator_notes",
        "investigator_summary",
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
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "Missing case-review columns:\n"
            + "\n".join(missing_columns)
        )

    return df


# ============================================================
# LOAD GRAPH EVIDENCE
# ============================================================

def load_graph_evidence():
    if not GRAPH_EVIDENCE_FILE.exists():
        raise FileNotFoundError(
            f"Graph evidence file not found:\n"
            f"{GRAPH_EVIDENCE_FILE}"
        )

    df = pd.read_csv(GRAPH_EVIDENCE_FILE)

    required_columns = [
        "case_id",
        "alert_id",
        "transaction_id",
        "sender_account",
        "receiver_account",
        "sender_in_degree",
        "sender_out_degree",
        "sender_total_degree",
        "receiver_in_degree",
        "receiver_out_degree",
        "receiver_total_degree",
        "sender_neighbor_count",
        "receiver_neighbor_count",
        "sender_neighbors",
        "receiver_neighbors",
        "related_transaction_count",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "Missing graph-evidence columns:\n"
            + "\n".join(missing_columns)
        )

    return df


# ============================================================
# MERGE INVESTIGATION DATA
# ============================================================

def merge_investigation_data(
    case_reviews,
    graph_evidence,
):
    if case_reviews["case_id"].duplicated().any():
        raise ValueError(
            "Duplicate case IDs found in case reviews."
        )

    if graph_evidence["case_id"].duplicated().any():
        raise ValueError(
            "Duplicate case IDs found in graph evidence."
        )

    dossiers = case_reviews.merge(
        graph_evidence,
        on=[
            "case_id",
            "alert_id",
            "transaction_id",
            "sender_account",
            "receiver_account",
        ],
        how="left",
        validate="one_to_one",
        suffixes=("", "_graph"),
    )

    return dossiers


# ============================================================
# BUILD INVESTIGATION SUMMARY
# ============================================================

def build_dossier_summary(row):
    graph_summary = (
        f"Sender degree={int(row['sender_total_degree'])}, "
        f"receiver degree={int(row['receiver_total_degree'])}, "
        f"related transactions="
        f"{int(row['related_transaction_count'])}"
    )

    model_summary = (
        f"RF={row['rf_probability']:.4f}, "
        f"TGN={row['tgn_probability']:.4f}, "
        f"rules={row['rule_score']:.4f}"
    )

    return (
        f"Case {row['case_id']} | "
        f"Transaction {row['transaction_id']} | "
        f"Risk={row['risk_category']} "
        f"({row['risk_score']:.4f}) | "
        f"{model_summary} | "
        f"{graph_summary}"
    )


# ============================================================
# SELECT FINAL DOSSIER COLUMNS
# ============================================================

def build_dossier_table(dossiers):
    dossiers["dossier_summary"] = dossiers.apply(
        build_dossier_summary,
        axis=1,
    )

    dossier_columns = [
        # Case identity
        "case_id",
        "alert_id",
        "transaction_id",

        # Transaction
        "timestamp",
        "sender_account",
        "receiver_account",
        "amount",

        # Risk
        "risk_score",
        "risk_category",
        "case_priority",

        # Workflow
        "case_status",
        "review_status",
        "investigator_decision",
        "investigator_notes",

        # Model evidence
        "rf_probability",
        "tgn_probability",
        "rule_score",

        # Rule evidence
        "triggered_rules",
        "large_transaction",
        "structuring",
        "rapid_movement",
        "fan_out",
        "circular_movement",

        # Graph evidence
        "sender_in_degree",
        "sender_out_degree",
        "sender_total_degree",
        "receiver_in_degree",
        "receiver_out_degree",
        "receiver_total_degree",
        "sender_neighbor_count",
        "receiver_neighbor_count",
        "sender_neighbors",
        "receiver_neighbors",
        "related_transaction_count",

        # Existing evidence summaries
        "evidence_summary",
        "investigator_summary",
        "dossier_summary",
    ]

    return dossiers[dossier_columns]


# ============================================================
# VALIDATE DOSSIERS
# ============================================================

def validate_dossiers(dossiers):
    if dossiers["case_id"].duplicated().any():
        raise ValueError(
            "Duplicate case IDs detected."
        )

    if dossiers["alert_id"].duplicated().any():
        raise ValueError(
            "Duplicate alert IDs detected."
        )

    if dossiers["transaction_id"].duplicated().any():
        raise ValueError(
            "Duplicate transaction IDs detected."
        )

    required_graph_columns = [
        "sender_total_degree",
        "receiver_total_degree",
        "sender_neighbor_count",
        "receiver_neighbor_count",
        "related_transaction_count",
    ]

    for column in required_graph_columns:
        if dossiers[column].isna().any():
            raise ValueError(
                f"Missing graph evidence detected in: {column}"
            )


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 70)
    print("AEGISGRAPH INVESTIGATION DOSSIER BUILDER")
    print("=" * 70)

    case_reviews = load_case_reviews()

    print(
        f"Case reviews loaded: {len(case_reviews)}"
    )

    graph_evidence = load_graph_evidence()

    print(
        f"Graph evidence rows loaded: "
        f"{len(graph_evidence)}"
    )

    dossiers = merge_investigation_data(
        case_reviews,
        graph_evidence,
    )

    print()
    print(
        f"Merged investigation rows: {len(dossiers)}"
    )

    dossier_df = build_dossier_table(
        dossiers
    )

    validate_dossiers(
        dossier_df
    )

    print()
    print("Investigation dossier summary:")
    print(
        f"  Total dossiers: {len(dossier_df)}"
    )

    print(
        f"  HIGH priority: "
        f"{(dossier_df['case_priority'] == 'HIGH').sum()}"
    )

    print(
        f"  MEDIUM priority: "
        f"{(dossier_df['case_priority'] == 'MEDIUM').sum()}"
    )

    print()
    print("Validation:")

    print(
        f"  Duplicate case IDs: "
        f"{dossier_df['case_id'].duplicated().sum()}"
    )

    print(
        f"  Duplicate alert IDs: "
        f"{dossier_df['alert_id'].duplicated().sum()}"
    )

    print(
        f"  Duplicate transaction IDs: "
        f"{dossier_df['transaction_id'].duplicated().sum()}"
    )

    print()
    print(
        "Graph evidence missing values:"
    )

    graph_columns = [
        "sender_total_degree",
        "receiver_total_degree",
        "sender_neighbor_count",
        "receiver_neighbor_count",
        "related_transaction_count",
    ]

    for column in graph_columns:
        print(
            f"  {column}: "
            f"{dossier_df[column].isna().sum()}"
        )

    # --------------------------------------------------------
    # SAVE DOSSIERS
    # --------------------------------------------------------

    dossier_df.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print()
    print("Saved to:")
    print(OUTPUT_FILE)

    print("=" * 70)


if __name__ == "__main__":
    main()