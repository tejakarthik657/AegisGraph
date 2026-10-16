import { useEffect, useMemo, useState } from "react";
import api from "../services/api";

export default function Performance() {
  const [summary, setSummary] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  useEffect(() => {
    let active = true;

    async function loadSummary() {
      try {
        setLoading(true);
        setError("");

        const response = await api.get("/summary");

        if (active) {
          setSummary(response.data);
        }
      } catch (err) {
        console.error("Failed to load monitoring summary:", err);

        if (active) {
          setError(
            err?.response?.data?.detail ||
              "Unable to load monitoring metrics."
          );
        }
      } finally {
        if (active) {
          setLoading(false);
        }
      }
    }

    loadSummary();

    return () => {
      active = false;
    };
  }, []);

  const riskCategories = useMemo(() => {
    const categories = summary?.risk?.categories || {};
    const total = Number(summary?.risk?.total_transactions || 0);

    const definitions = [
      { key: "VERY_LOW", label: "Very Low", color: "very-low" },
      { key: "LOW", label: "Low", color: "low" },
      { key: "MEDIUM", label: "Medium", color: "medium" },
      { key: "HIGH", label: "High", color: "high" },
    ];

    return definitions.map((item) => {
      const count = Number(categories[item.key] || 0);

      return {
        ...item,
        count,
        percentage: total > 0 ? (count / total) * 100 : 0,
      };
    });
  }, [summary]);

  const casePriorities = useMemo(() => {
    const priorities = summary?.investigation?.priority || {};
    const total = Number(summary?.investigation?.total_cases || 0);

    return [
      {
        key: "HIGH",
        label: "High priority",
        count: Number(priorities.HIGH || 0),
        color: "high",
      },
      {
        key: "MEDIUM",
        label: "Medium priority",
        count: Number(priorities.MEDIUM || 0),
        color: "medium",
      },
      {
        key: "LOW",
        label: "Low priority",
        count: Number(priorities.LOW || 0),
        color: "low",
      },
    ].map((item) => ({
      ...item,
      percentage: total > 0 ? (item.count / total) * 100 : 0,
    }));
  }, [summary]);

  const handleRetry = async () => {
    try {
      setLoading(true);
      setError("");

      const response = await api.get("/summary");
      setSummary(response.data);
    } catch (err) {
      setError(
        err?.response?.data?.detail ||
          "Unable to load monitoring metrics."
      );
    } finally {
      setLoading(false);
    }
  };

  if (loading) {
    return (
      <section className="performance-page">
        <div className="performance-state">
          <div className="performance-spinner" />
          <p>Loading monitoring metrics...</p>
        </div>
      </section>
    );
  }

  if (error || !summary) {
    return (
      <section className="performance-page">
        <div className="performance-state performance-error">
          <h2>Monitoring data unavailable</h2>
          <p>{error || "No summary data was returned."}</p>
          <button type="button" onClick={handleRetry}>
            Retry
          </button>
        </div>
      </section>
    );
  }

  const totalTransactions = Number(
    summary.risk?.total_transactions || 0
  );

  const totalAlerts = Number(summary.alerts?.total || 0);

  const totalCases = Number(
    summary.investigation?.total_cases || 0
  );

  const totalDossiers = Number(
    summary.investigation?.total_dossiers || 0
  );

  const totalHighRisk = Number(
    summary.risk?.categories?.HIGH || 0
  );

  const totalMediumRisk = Number(
    summary.risk?.categories?.MEDIUM || 0
  );

  return (
    <section className="performance-page">
      <div className="performance-header">
        <div>
          <div className="page-eyebrow">MONITORING</div>
          <h1>System Performance</h1>
          <p>
            Monitor transaction risk distribution, alert generation,
            and investigation workload.
          </p>
        </div>

        <div className="performance-live-indicator">
          <span />
          Connected to backend
        </div>
      </div>

      <div className="performance-metric-grid">
        <article className="performance-metric-card">
          <div className="performance-metric-label">
            Transactions Analyzed
          </div>
          <strong>{totalTransactions.toLocaleString("en-IN")}</strong>
          <span className="performance-metric-description">
            Transactions represented in the risk summary
          </span>
        </article>

        <article className="performance-metric-card">
          <div className="performance-metric-label">
            Alerts Generated
          </div>
          <strong>{totalAlerts.toLocaleString("en-IN")}</strong>
          <span className="performance-metric-description">
            Alerts in the detection output
          </span>
        </article>

        <article className="performance-metric-card">
          <div className="performance-metric-label">
            Investigation Cases
          </div>
          <strong>{totalCases.toLocaleString("en-IN")}</strong>
          <span className="performance-metric-description">
            Cases created for investigation
          </span>
        </article>

        <article className="performance-metric-card">
          <div className="performance-metric-label">
            Evidence Dossiers
          </div>
          <strong>{totalDossiers.toLocaleString("en-IN")}</strong>
          <span className="performance-metric-description">
            Dossiers available for review
          </span>
        </article>
      </div>

      <div className="performance-content-grid">
        <section className="performance-panel">
          <div className="performance-panel-header">
            <div>
              <h2>Risk Distribution</h2>
              <p>Transactions grouped by assigned risk category.</p>
            </div>
            <span className="performance-panel-total">
              {totalTransactions.toLocaleString("en-IN")} total
            </span>
          </div>

          <div className="performance-risk-list">
            {riskCategories.map((item) => (
              <div className="performance-risk-row" key={item.key}>
                <div className="performance-risk-heading">
                  <span className="performance-risk-name">
                    <i className={`performance-dot ${item.color}`} />
                    {item.label}
                  </span>
                  <strong>{item.count.toLocaleString("en-IN")}</strong>
                </div>

                <div
                  className="performance-progress-track"
                  role="progressbar"
                  aria-label={`${item.label} risk transactions`}
                  aria-valuenow={Number(item.percentage.toFixed(2))}
                  aria-valuemin={0}
                  aria-valuemax={100}
                >
                  <div
                    className={`performance-progress-fill ${item.color}`}
                    style={{ width: `${item.percentage}%` }}
                  />
                </div>

                <div className="performance-risk-percentage">
                  {item.percentage.toFixed(2)}% of analyzed transactions
                </div>
              </div>
            ))}
          </div>
        </section>

        <section className="performance-panel">
          <div className="performance-panel-header">
            <div>
              <h2>Investigation Priority</h2>
              <p>Case workload by assigned priority.</p>
            </div>
            <span className="performance-panel-total">
              {totalCases.toLocaleString("en-IN")} total
            </span>
          </div>

          <div className="performance-priority-list">
            {casePriorities.map((item) => (
              <div className="performance-priority-row" key={item.key}>
                <div className="performance-priority-heading">
                  <span>{item.label}</span>
                  <strong>{item.count}</strong>
                </div>

                <div
                  className="performance-progress-track"
                  role="progressbar"
                  aria-label={item.label}
                  aria-valuenow={Number(item.percentage.toFixed(2))}
                  aria-valuemin={0}
                  aria-valuemax={100}
                >
                  <div
                    className={`performance-progress-fill ${item.color}`}
                    style={{ width: `${item.percentage}%` }}
                  />
                </div>
              </div>
            ))}
          </div>

          <div className="performance-priority-note">
            <strong>{totalHighRisk + totalMediumRisk}</strong>
            <span>
              High- and medium-risk transactions in the risk summary
            </span>
          </div>
        </section>
      </div>

      <section className="performance-panel performance-integrity-panel">
        <div className="performance-panel-header">
          <div>
            <h2>Pipeline Overview</h2>
            <p>Counts reported by the current backend summary.</p>
          </div>
        </div>

        <div className="performance-pipeline">
          <div className="performance-pipeline-step">
            <span className="performance-pipeline-number">01</span>
            <div>
              <strong>Risk Assessment</strong>
              <p>
                {totalTransactions.toLocaleString("en-IN")} transactions
                categorized
              </p>
            </div>
          </div>

          <div className="performance-pipeline-connector" />

          <div className="performance-pipeline-step">
            <span className="performance-pipeline-number">02</span>
            <div>
              <strong>Alert Generation</strong>
              <p>{totalAlerts} alerts reported</p>
            </div>
          </div>

          <div className="performance-pipeline-connector" />

          <div className="performance-pipeline-step">
            <span className="performance-pipeline-number">03</span>
            <div>
              <strong>Case Investigation</strong>
              <p>{totalCases} cases created</p>
            </div>
          </div>

          <div className="performance-pipeline-connector" />

          <div className="performance-pipeline-step">
            <span className="performance-pipeline-number">04</span>
            <div>
              <strong>Evidence Dossiers</strong>
              <p>{totalDossiers} dossiers reported</p>
            </div>
          </div>
        </div>

        <div className="performance-data-note">
          These figures reflect the current summary endpoint. They are
          aggregate counts, not live throughput, latency, or model-quality
          measurements.
        </div>
      </section>
    </section>
  );
}