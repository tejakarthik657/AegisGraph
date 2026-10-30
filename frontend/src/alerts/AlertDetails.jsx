import { useEffect, useState } from "react";
import { Link, useNavigate, useParams } from "react-router-dom";

import RiskEvidenceCards from "../components/models/RiskEvidenceCards";
import {
  getAlert,
  getTransactionById,
} from "../services/api";
import {
  formatScorePercentage,
  mergeTransactionRiskScores,
} from "../utils/riskScores";

function formatAmount(amount) {
  return new Intl.NumberFormat("en-IN", {
    style: "currency",
    currency: "INR",
    maximumFractionDigits: 2,
  }).format(amount);
}

function formatDate(timestamp) {
  if (!timestamp) {
    return "—";
  }

  const date = new Date(timestamp.replace(" ", "T"));

  if (Number.isNaN(date.getTime())) {
    return timestamp;
  }

  return date.toLocaleString("en-IN", {
    day: "2-digit",
    month: "long",
    year: "numeric",
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  });
}

function formatPattern(value) {
  if (!value) {
    return "Risk signal";
  }

  return value
    .replaceAll("_", " ")
    .replace(/\b\w/g, (letter) => letter.toUpperCase());
}

function RiskBadge({ category }) {
  return (
    <span
      className={`detail-risk-badge detail-risk-${category.toLowerCase()}`}
    >
      <span className="detail-risk-dot" />
      {category} RISK
    </span>
  );
}

function RuleItem({
  label,
  triggered,
  description,
}) {
  return (
    <div className="detail-rule-item">
      <div
        className={
          triggered
            ? "detail-rule-indicator triggered"
            : "detail-rule-indicator"
        }
      >
        {triggered ? "✓" : "—"}
      </div>

      <div className="detail-rule-content">
        <div className="detail-rule-label">
          {label}
        </div>

        <div className="detail-rule-description">
          {description}
        </div>
      </div>

      <div
        className={
          triggered
            ? "detail-rule-status triggered"
            : "detail-rule-status"
        }
      >
        {triggered ? "Triggered" : "Not triggered"}
      </div>
    </div>
  );
}

function AlertDetailsLoading() {
  return (
    <section className="page-content">
      <div className="alert-details-loading">
        <div className="loading-spinner" />

        <div>
          <div className="loading-title">
            Loading alert
          </div>

          <div className="loading-description">
            Retrieving alert evidence...
          </div>
        </div>
      </div>
    </section>
  );
}

function AlertDetailsError({ message }) {
  return (
    <section className="page-content">
      <div className="alert-details-error">
        <div className="error-title">
          Alert unavailable
        </div>

        <div className="error-description">
          AegisGraph could not retrieve this alert.
        </div>

        <div className="error-details">
          {message}
        </div>

        <Link
          to="/alerts"
          className="back-to-alerts"
        >
          ← Back to Alerts
        </Link>
      </div>
    </section>
  );
}

export default function AlertDetails() {
  const { alertId } = useParams();
  const navigate = useNavigate();

  const [alert, setAlert] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  useEffect(() => {
    let mounted = true;

    async function loadAlert() {
      try {
        setLoading(true);
        setError("");

        const data = await getAlert(alertId);
        const transaction = await getTransactionById(
          data.transaction_id
        );

        if (mounted) {
          setAlert(
            mergeTransactionRiskScores(data, transaction)
          );
        }
      } catch (err) {
        if (mounted) {
          setError(
            err instanceof Error
              ? err.message
              : "Unknown API error"
          );
        }
      } finally {
        if (mounted) {
          setLoading(false);
        }
      }
    }

    if (alertId) {
      loadAlert();
    }

    return () => {
      mounted = false;
    };
  }, [alertId]);

  if (loading) {
    return <AlertDetailsLoading />;
  }

  if (error || !alert) {
    return (
      <AlertDetailsError
        message={
          error || "No alert data was returned."
        }
      />
    );
  }

  const triggeredRules = [
    {
      label: "Large Transaction",
      triggered: Boolean(alert.large_transaction),
      description:
        "The transaction exceeded the configured large-transaction threshold.",
    },
    {
      label: "Structuring",
      triggered: Boolean(alert.structuring),
      description:
        "The transaction is associated with a possible structuring pattern.",
    },
    {
      label: "Rapid Movement",
      triggered: Boolean(alert.rapid_movement),
      description:
        "Funds moved rapidly through the transaction network.",
    },
    {
      label: "Fan-out",
      triggered: Boolean(alert.fan_out),
      description:
        "The account activity shows funds distributed across multiple receivers.",
    },
    {
      label: "Circular Movement",
      triggered: Boolean(alert.circular_movement),
      description:
        "The transaction is associated with a circular movement pattern.",
    },
  ];

  return (
    <section className="page-content alert-details-page">
      <div className="alert-details-topbar">
        <button
          type="button"
          className="back-button"
          onClick={() => navigate("/alerts")}
        >
          ← Back to Alerts
        </button>
      </div>

      <div className="detail-header">
        <div>
          <div className="page-eyebrow">
            ALERT INVESTIGATION
          </div>

          <div className="detail-title-row">
            <h1>{alert.alert_id}</h1>

            <RiskBadge
              category={alert.risk_category}
            />
          </div>

          <p>
            Transaction {alert.transaction_id} was
            identified as requiring analyst attention.
          </p>
        </div>

        <div className="detail-header-status">
          <span className="detail-status-label">
            STATUS
          </span>

          <span className="detail-status-value">
            {alert.alert_status}
          </span>
        </div>
      </div>

      <section className="detail-hero">
        <div className="detail-hero-main">
          <div className="detail-hero-label">
            Overall Risk
          </div>

          <div className="detail-hero-score">
            {formatScorePercentage(alert.risk_score)}
          </div>

          <div className="detail-hero-description">
            Combined risk assessment from machine learning
            and AML detection signals.
          </div>
        </div>

        <div className="detail-hero-transaction">
          <div className="detail-hero-label">
            Transaction Amount
          </div>

          <div className="detail-hero-amount">
            {formatAmount(alert.amount)}
          </div>

          <div className="detail-route">
            <span>{alert.sender_account}</span>

            <span className="detail-route-arrow">
              →
            </span>

            <span>{alert.receiver_account}</span>
          </div>
        </div>

        <div className="detail-hero-pattern">
          <div className="detail-hero-label">
            Detected Pattern
          </div>

          <div className="detail-hero-pattern-value">
            {formatPattern(alert.typology)}
          </div>

          <div className="detail-hero-description">
            Primary suspicious activity classification.
          </div>
        </div>
      </section>

      <div className="detail-grid">
        <div className="detail-main-column">
          <section className="detail-section">
            <div className="detail-section-heading">
              <div>
                <h2>Why was this alert created?</h2>

                <p>
                  The following detection signals contributed
                  to the alert.
                </p>
              </div>
            </div>

            <RiskEvidenceCards scores={alert} />
          </section>

          <section className="detail-section">
            <div className="detail-section-heading">
              <div>
                <h2>Detection Rules</h2>

                <p>
                  Rule-based signals evaluated for this
                  transaction.
                </p>
              </div>
            </div>

            <div className="detail-rules-panel">
              {triggeredRules.map((rule) => (
                <RuleItem
                  key={rule.label}
                  label={rule.label}
                  triggered={rule.triggered}
                  description={rule.description}
                />
              ))}
            </div>
          </section>

          <section className="detail-section">
            <div className="detail-section-heading">
              <div>
                <h2>Triggered Signals</h2>

                <p>
                  Detection rules that directly contributed
                  to this alert.
                </p>
              </div>
            </div>

            <div className="triggered-signal-list">
              {alert.triggered_rules
                ? alert.triggered_rules
                    .split("|")
                    .map((rule) => (
                      <span
                        className="triggered-signal"
                        key={rule}
                      >
                        {formatPattern(rule)}
                      </span>
                    ))
                : (
                  <span className="no-signals">
                    No rule signals recorded.
                  </span>
                )}
            </div>
          </section>
        </div>

        <aside className="detail-side-column">
          <section className="detail-section detail-information-card">
            <div className="detail-section-heading">
              <div>
                <h2>Transaction Details</h2>
              </div>
            </div>

            <div className="detail-information-list">
              <div className="detail-information-row">
                <span>Transaction ID</span>
                <strong>{alert.transaction_id}</strong>
              </div>

              <div className="detail-information-row">
                <span>Sender Account</span>
                <strong>{alert.sender_account}</strong>
              </div>

              <div className="detail-information-row">
                <span>Receiver Account</span>
                <strong>{alert.receiver_account}</strong>
              </div>

              <div className="detail-information-row">
                <span>Amount</span>
                <strong>
                  {formatAmount(alert.amount)}
                </strong>
              </div>

              <div className="detail-information-row">
                <span>Timestamp</span>
                <strong>
                  {formatDate(alert.timestamp)}
                </strong>
              </div>

              <div className="detail-information-row">
                <span>Risk Category</span>
                <strong>
                  {alert.risk_category}
                </strong>
              </div>
            </div>
          </section>

          <section className="investigation-action-card">
            <div className="action-card-eyebrow">
              NEXT STEP
            </div>

            <h2>Investigate this alert</h2>

            <p>
              Review account relationships, graph evidence,
              detection signals and supporting evidence in
              the investigation workspace.
            </p>

            <button
              type="button"
              className="investigate-button"
              onClick={() =>
                navigate(
                  `/workspace/${alert.alert_id}`
                )
              }
            >
              Investigate Case
              <span>→</span>
            </button>
          </section>
        </aside>
      </div>
    </section>
  );
}