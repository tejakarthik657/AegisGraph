import { useEffect, useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";
import RiskEvidenceCards from "../components/models/RiskEvidenceCards";
import api from "../services/api";

export default function Transactions() {
  const navigate = useNavigate();

  const [transactions, setTransactions] = useState([]);
  const [selectedTransaction, setSelectedTransaction] = useState(null);
  const [pagination, setPagination] = useState({
    page: 1,
    page_size: 25,
    total: 0,
    total_pages: 0,
  });

  const [search, setSearch] = useState("");
  const [typology, setTypology] = useState("");
  const [status, setStatus] = useState("");
  const [fraudOnly, setFraudOnly] = useState(false);

  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  const loadTransactions = async (page = 1) => {
    try {
      setLoading(true);
      setError("");
      setSelectedTransaction(null);

      const params = {
        page,
        page_size: 25,
        search,
        typology,
        status,
        fraud_only: fraudOnly,
      };

      const response = await api.get("/transactions", {
        params,
      });

      setTransactions(response.data.transactions || []);
      setPagination(
        response.data.pagination || {
          page: 1,
          page_size: 25,
          total: 0,
          total_pages: 0,
        }
      );
    } catch (err) {
      console.error("Failed to load transactions:", err);

      setError(
        err?.response?.data?.detail ||
          "Unable to load transaction data."
      );
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadTransactions(1);
  }, [search, typology, status, fraudOnly]);

  const formatAmount = (amount) => {
    const value = Number(amount);

    if (!Number.isFinite(value)) {
      return "—";
    }

    return new Intl.NumberFormat("en-IN", {
      style: "currency",
      currency: "INR",
      maximumFractionDigits: 2,
    }).format(value);
  };

  const formatTimestamp = (timestamp) => {
    if (!timestamp) {
      return "—";
    }

    const date = new Date(timestamp);

    if (Number.isNaN(date.getTime())) {
      return timestamp;
    }

    return date.toLocaleString("en-IN", {
      day: "2-digit",
      month: "short",
      year: "numeric",
      hour: "2-digit",
      minute: "2-digit",
      second: "2-digit",
    });
  };

  const getTypologyLabel = (value) => {
    if (!value) {
      return "Normal";
    }

    return String(value)
      .replaceAll("_", " ")
      .replace(/\b\w/g, (char) => char.toUpperCase());
  };

  const getRiskClass = (transaction) => {
    if (Number(transaction.is_fraud) === 1) {
      return "transaction-risk-fraud";
    }

    return "transaction-risk-normal";
  };

  const visibleRange = useMemo(() => {
    if (pagination.total === 0) {
      return "0";
    }

    const start =
      (pagination.page - 1) * pagination.page_size + 1;

    const end = Math.min(
      pagination.page * pagination.page_size,
      pagination.total
    );

    return `${start}–${end}`;
  }, [pagination]);

  const handlePageChange = (page) => {
    if (
      page < 1 ||
      page > pagination.total_pages ||
      page === pagination.page
    ) {
      return;
    }

    loadTransactions(page);
  };

  const clearFilters = () => {
    setSearch("");
    setTypology("");
    setStatus("");
    setFraudOnly(false);
  };

  return (
    <section className="transactions-page">

      {/* PAGE HEADER */}
      <div className="page-header">
        <div>
          <div className="page-eyebrow">
            DETECTION
          </div>

          <h1>Transactions</h1>

          <p>
            Review transaction activity across the
            monitored financial network.
          </p>
        </div>

        <div className="transactions-header-stat">
          <span className="transactions-header-stat-label">
            TOTAL TRANSACTIONS
          </span>

          <strong>
            {pagination.total.toLocaleString("en-IN")}
          </strong>
        </div>
      </div>

      {/* FILTER BAR */}
      <div className="transactions-filter-panel">

        <div className="transaction-search">
          <label htmlFor="transaction-search">
            Search
          </label>

          <input
            id="transaction-search"
            type="text"
            placeholder="Transaction ID or account ID..."
            value={search}
            onChange={(event) =>
              setSearch(event.target.value)
            }
          />
        </div>

        <div className="transaction-filter">
          <label htmlFor="transaction-typology">
            Typology
          </label>

          <select
            id="transaction-typology"
            value={typology}
            onChange={(event) =>
              setTypology(event.target.value)
            }
          >
            <option value="">
              All typologies
            </option>

            <option value="normal">
              Normal
            </option>

            <option value="structuring">
              Structuring
            </option>

            <option value="rapid_movement">
              Rapid Movement
            </option>

            <option value="fan_out">
              Fan Out
            </option>

            <option value="circular">
              Circular
            </option>
          </select>
        </div>

        <div className="transaction-filter">
          <label htmlFor="transaction-status">
            Status
          </label>

          <select
            id="transaction-status"
            value={status}
            onChange={(event) =>
              setStatus(event.target.value)
            }
          >
            <option value="">
              All statuses
            </option>

            <option value="COMPLETED">
              Completed
            </option>

            <option value="FAILED">
              Failed
            </option>
          </select>
        </div>

        <label className="transaction-fraud-filter">
          <input
            type="checkbox"
            checked={fraudOnly}
            onChange={(event) =>
              setFraudOnly(event.target.checked)
            }
          />

          <span>
            Fraud only
          </span>
        </label>

        <button
          type="button"
          className="transaction-clear-button"
          onClick={clearFilters}
        >
          Clear
        </button>

      </div>

      {/* CONTENT */}
      <div className="transactions-card">

        {loading && (
          <div className="transactions-state">
            <div className="transactions-spinner" />
            <p>Loading transactions...</p>
          </div>
        )}

        {!loading && error && (
          <div className="transactions-state transactions-error">
            <div className="transactions-state-icon">
              !
            </div>

            <h3>
              Unable to load transactions
            </h3>

            <p>{error}</p>

            <button
              type="button"
              onClick={() =>
                loadTransactions(pagination.page)
              }
            >
              Retry
            </button>
          </div>
        )}

        {!loading &&
          !error &&
          transactions.length === 0 && (
            <div className="transactions-state">
              <div className="transactions-state-icon">
                —
              </div>

              <h3>
                No transactions found
              </h3>

              <p>
                Try changing the current filters.
              </p>
            </div>
          )}

        {!loading &&
          !error &&
          transactions.length > 0 && (
            <>
              <div className="transactions-table-wrapper">
                <table className="transactions-table">

                  <thead>
                    <tr>
                      <th>Transaction</th>
                      <th>Timestamp</th>
                      <th>Sender</th>
                      <th>Receiver</th>
                      <th>Amount</th>
                      <th>Type</th>
                      <th>Channel</th>
                      <th>Status</th>
                      <th>Classification</th>
                    </tr>
                  </thead>

                  <tbody>
                    {transactions.map((transaction) => (
                      <tr
                        key={transaction.transaction_id}
                        className={
                          Number(transaction.is_fraud) === 1
                            ? "transaction-row-fraud"
                            : ""
                        }
                      >

                        <td>
                          <div className="transaction-id-cell">
                            <button
                              type="button"
                              className="transaction-id-button"
                              aria-label={`View risk details for ${transaction.transaction_id}`}
                              aria-pressed={
                                selectedTransaction?.transaction_id ===
                                transaction.transaction_id
                              }
                              onClick={() =>
                                setSelectedTransaction(transaction)
                              }
                            >
                              {transaction.transaction_id}
                            </button>

                            {Number(transaction.is_fraud) === 1 && (
                              <span className="transaction-fraud-dot" />
                            )}
                          </div>
                        </td>

                        <td>
                          <span className="transaction-time">
                            {formatTimestamp(
                              transaction.timestamp
                            )}
                          </span>
                        </td>

                        <td>
                          <span className="transaction-account">
                            {transaction.sender_account}
                          </span>
                        </td>

                        <td>
                          <span className="transaction-account">
                            {transaction.receiver_account}
                          </span>
                        </td>

                        <td>
                          <strong className="transaction-amount">
                            {formatAmount(
                              transaction.amount
                            )}
                          </strong>
                        </td>

                        <td>
                          <span className="transaction-type">
                            {transaction.transaction_type ||
                              "—"}
                          </span>
                        </td>

                        <td>
                          <span className="transaction-channel">
                            {transaction.channel || "—"}
                          </span>
                        </td>

                        <td>
                          <span
                            className={`transaction-status transaction-status-${String(
                              transaction.status || ""
                            ).toLowerCase()}`}
                          >
                            {transaction.status || "—"}
                          </span>
                        </td>

                        <td>
                          <span
                            className={`transaction-classification ${getRiskClass(
                              transaction
                            )}`}
                          >
                            {getTypologyLabel(
                              transaction.typology
                            )}
                          </span>
                        </td>

                      </tr>
                    ))}
                  </tbody>

                </table>
              </div>

              {selectedTransaction && (
                <section className="detail-section transaction-risk-details">
                  <div className="detail-section-heading">
                    <h2>
                      Risk Signals —{" "}
                      {selectedTransaction.transaction_id}
                    </h2>
                    <p>
                      Model probabilities, AML rule score, and final
                      classification returned by the backend.
                    </p>
                  </div>

                  <RiskEvidenceCards scores={selectedTransaction} />
                </section>
              )}

              {/* PAGINATION */}
              <div className="transactions-pagination">

                <div className="transactions-pagination-info">
                  Showing{" "}
                  <strong>{visibleRange}</strong>{" "}
                  of{" "}
                  <strong>
                    {pagination.total.toLocaleString(
                      "en-IN"
                    )}
                  </strong>
                </div>

                <div className="transactions-pagination-controls">

                  <button
                    type="button"
                    disabled={pagination.page <= 1}
                    onClick={() =>
                      handlePageChange(
                        pagination.page - 1
                      )
                    }
                  >
                    Previous
                  </button>

                  <span>
                    Page{" "}
                    <strong>
                      {pagination.page}
                    </strong>{" "}
                    of{" "}
                    <strong>
                      {pagination.total_pages}
                    </strong>
                  </span>

                  <button
                    type="button"
                    disabled={
                      pagination.page >=
                      pagination.total_pages
                    }
                    onClick={() =>
                      handlePageChange(
                        pagination.page + 1
                      )
                    }
                  >
                    Next
                  </button>

                </div>
              </div>
            </>
          )}

      </div>

    </section>
  );
}