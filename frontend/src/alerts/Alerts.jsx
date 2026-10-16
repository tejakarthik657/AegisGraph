import {
  useEffect,
  useMemo,
  useState,
} from "react";

import { useNavigate } from "react-router-dom";

import { getAlerts } from "../services/api";

const ITEMS_PER_PAGE = 10;

function formatAmount(amount) {
  return new Intl.NumberFormat("en-IN", {
    style: "currency",
    currency: "INR",
    maximumFractionDigits: 2,
  }).format(amount);
}

function formatRiskScore(score) {
  return `${Math.round(score * 100)}%`;
}

function formatDate(timestamp) {
  if (!timestamp) {
    return "—";
  }

  const date = new Date(
    timestamp.replace(" ", "T")
  );

  if (Number.isNaN(date.getTime())) {
    return timestamp;
  }

  return date.toLocaleString("en-IN", {
    day: "2-digit",
    month: "short",
    year: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}

function getPatternName(alert) {
  if (alert.typology) {
    return alert.typology
      .replaceAll("_", " ")
      .replace(/\b\w/g, (letter) =>
        letter.toUpperCase()
      );
  }

  if (alert.triggered_rules) {
    return alert.triggered_rules
      .split("|")[0]
      .replaceAll("_", " ")
      .replace(/\b\w/g, (letter) =>
        letter.toUpperCase()
      );
  }

  return "Risk signal";
}

function RiskBadge({ category }) {
  return (
    <span
      className={`alert-risk-badge alert-risk-${category.toLowerCase()}`}
    >
      <span className="alert-risk-dot" />

      {category}
    </span>
  );
}

function StatusBadge({ status }) {
  return (
    <span className="alert-status-badge">
      {status}
    </span>
  );
}

function AlertsLoading() {
  return (
    <section className="page-content">
      <div className="alerts-loading">
        <div className="loading-spinner" />

        <div>
          <div className="loading-title">
            Loading alerts
          </div>

          <div className="loading-description">
            Retrieving current investigation alerts...
          </div>
        </div>
      </div>
    </section>
  );
}

function AlertsError({ message }) {
  return (
    <section className="page-content">
      <div className="alerts-error">
        <div className="error-title">
          Alerts unavailable
        </div>

        <div className="error-description">
          AegisGraph could not retrieve the current
          alert queue.
        </div>

        <div className="error-details">
          {message}
        </div>
      </div>
    </section>
  );
}

export default function Alerts() {
  const navigate = useNavigate();

  const [alerts, setAlerts] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  const [riskFilter, setRiskFilter] =
    useState("ALL");

  const [search, setSearch] = useState("");

  const [currentPage, setCurrentPage] =
    useState(1);

  useEffect(() => {
    let mounted = true;

    async function loadAlerts() {
      try {
        setLoading(true);
        setError("");

        const data = await getAlerts();

        if (mounted) {
          setAlerts(data.alerts || []);
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

    loadAlerts();

    return () => {
      mounted = false;
    };
  }, []);

  const highCount = alerts.filter(
    (alert) =>
      alert.risk_category === "HIGH"
  ).length;

  const mediumCount = alerts.filter(
    (alert) =>
      alert.risk_category === "MEDIUM"
  ).length;

  const filteredAlerts = useMemo(() => {
    const normalizedSearch =
      search.trim().toLowerCase();

    return alerts.filter((alert) => {
      const matchesRisk =
        riskFilter === "ALL" ||
        alert.risk_category === riskFilter;

      if (!matchesRisk) {
        return false;
      }

      if (!normalizedSearch) {
        return true;
      }

      return [
        alert.alert_id,
        alert.transaction_id,
        alert.sender_account,
        alert.receiver_account,
        alert.typology,
        alert.triggered_rules,
      ]
        .filter(Boolean)
        .some((value) =>
          String(value)
            .toLowerCase()
            .includes(normalizedSearch)
        );
    });
  }, [alerts, riskFilter, search]);

  const totalPages = Math.max(
    1,
    Math.ceil(
      filteredAlerts.length /
        ITEMS_PER_PAGE
    )
  );

  useEffect(() => {
    if (currentPage > totalPages) {
      setCurrentPage(totalPages);
    }
  }, [currentPage, totalPages]);

  useEffect(() => {
    setCurrentPage(1);
  }, [riskFilter, search]);

  const paginatedAlerts = useMemo(() => {
    const startIndex =
      (currentPage - 1) *
      ITEMS_PER_PAGE;

    return filteredAlerts.slice(
      startIndex,
      startIndex + ITEMS_PER_PAGE
    );
  }, [filteredAlerts, currentPage]);

  const firstVisibleAlert =
    filteredAlerts.length === 0
      ? 0
      : (currentPage - 1) *
          ITEMS_PER_PAGE +
        1;

  const lastVisibleAlert = Math.min(
    currentPage * ITEMS_PER_PAGE,
    filteredAlerts.length
  );

  if (loading) {
    return <AlertsLoading />;
  }

  if (error) {
    return (
      <AlertsError message={error} />
    );
  }

  return (
    <section className="page-content alerts-page">
      <div className="page-header alerts-header">
        <div>
          <div className="page-eyebrow">
            DETECTION QUEUE
          </div>

          <h1>Alerts</h1>

          <p>
            Review transactions that require
            analyst attention.
          </p>
        </div>

        <div className="alerts-summary">
          <div className="alerts-summary-item">
            <span className="alerts-summary-value">
              {alerts.length}
            </span>

            <span className="alerts-summary-label">
              Total alerts
            </span>
          </div>

          <div className="alerts-summary-divider" />

          <div className="alerts-summary-item">
            <span className="alerts-summary-value alert-summary-high">
              {highCount}
            </span>

            <span className="alerts-summary-label">
              High risk
            </span>
          </div>

          <div className="alerts-summary-divider" />

          <div className="alerts-summary-item">
            <span className="alerts-summary-value alert-summary-medium">
              {mediumCount}
            </span>

            <span className="alerts-summary-label">
              Medium risk
            </span>
          </div>
        </div>
      </div>

      <div className="alerts-controls">
        <div className="alert-filter-group">
          <button
            type="button"
            className={
              riskFilter === "ALL"
                ? "alert-filter active"
                : "alert-filter"
            }
            onClick={() =>
              setRiskFilter("ALL")
            }
          >
            All

            <span>
              {alerts.length}
            </span>
          </button>

          <button
            type="button"
            className={
              riskFilter === "HIGH"
                ? "alert-filter active high"
                : "alert-filter high"
            }
            onClick={() =>
              setRiskFilter("HIGH")
            }
          >
            High

            <span>
              {highCount}
            </span>
          </button>

          <button
            type="button"
            className={
              riskFilter === "MEDIUM"
                ? "alert-filter active medium"
                : "alert-filter medium"
            }
            onClick={() =>
              setRiskFilter("MEDIUM")
            }
          >
            Medium

            <span>
              {mediumCount}
            </span>
          </button>
        </div>

        <div className="alert-search">
          <span className="alert-search-icon">
            ⌕
          </span>

          <input
            type="text"
            placeholder="Search alert, transaction or account..."
            value={search}
            onChange={(event) =>
              setSearch(event.target.value)
            }
          />
        </div>
      </div>

      <div className="alerts-table-container">
        <table className="alerts-table">
          <thead>
            <tr>
              <th>Alert</th>
              <th>Transaction</th>
              <th>Risk</th>
              <th>Amount</th>
              <th>Pattern</th>
              <th>Risk Score</th>
              <th>Status</th>
              <th>Time</th>
            </tr>
          </thead>

          <tbody>
            {paginatedAlerts.map((alert) => (
              <tr
                key={alert.alert_id}
                onClick={() =>
                  navigate(
                    `/alerts/${alert.alert_id}`
                  )
                }
              >
                <td>
                  <div className="alert-id">
                    {alert.alert_id}
                  </div>

                  <div className="alert-account">
                    {alert.sender_account}
                    {" → "}
                    {alert.receiver_account}
                  </div>
                </td>

                <td>
                  <div className="transaction-id">
                    {alert.transaction_id}
                  </div>
                </td>

                <td>
                  <RiskBadge
                    category={
                      alert.risk_category
                    }
                  />
                </td>

                <td>
                  <div className="alert-amount">
                    {formatAmount(
                      alert.amount
                    )}
                  </div>
                </td>

                <td>
                  <div className="alert-pattern">
                    {getPatternName(alert)}
                  </div>
                </td>

                <td>
                  <div className="alert-score">
                    {formatRiskScore(
                      alert.risk_score
                    )}
                  </div>
                </td>

                <td>
                  <StatusBadge
                    status={
                      alert.alert_status
                    }
                  />
                </td>

                <td>
                  <div className="alert-time">
                    {formatDate(
                      alert.timestamp
                    )}
                  </div>
                </td>
              </tr>
            ))}

            {paginatedAlerts.length === 0 && (
              <tr>
                <td
                  colSpan="8"
                  className="alerts-empty"
                >
                  No alerts match the current
                  filters.
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>

      <div className="alerts-pagination">
        <div className="pagination-information">
          Showing{" "}
          <strong>
            {firstVisibleAlert}
          </strong>
          {"–"}
          <strong>
            {lastVisibleAlert}
          </strong>
          {" of "}
          <strong>
            {filteredAlerts.length}
          </strong>
          {" alerts"}
        </div>

        <div className="pagination-controls">
          <button
            type="button"
            className="pagination-button"
            disabled={currentPage === 1}
            onClick={() =>
              setCurrentPage(
                (page) => page - 1
              )
            }
          >
            ← Previous
          </button>

          <div className="pagination-pages">
            {Array.from(
              {
                length: totalPages,
              },
              (_, index) => index + 1
            ).map((page) => (
              <button
                key={page}
                type="button"
                className={
                  currentPage === page
                    ? "pagination-page active"
                    : "pagination-page"
                }
                onClick={() =>
                  setCurrentPage(page)
                }
              >
                {page}
              </button>
            ))}
          </div>

          <button
            type="button"
            className="pagination-button"
            disabled={
              currentPage === totalPages
            }
            onClick={() =>
              setCurrentPage(
                (page) => page + 1
              )
            }
          >
            Next →
          </button>
        </div>
      </div>
    </section>
  );
}