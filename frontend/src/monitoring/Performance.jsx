import { useEffect, useMemo, useState } from "react";
import api from "../services/api";

const evaluationMetricColumns = [
  ["precision", "Precision"],
  ["recall", "Recall"],
  ["f1", "F1-score"],
  ["roc_auc", "ROC-AUC"],
  ["pr_auc", "PR-AUC / Average Precision"],
  ["accuracy", "Accuracy"],
];

function displayMetric(value) {
  return value === null || value === undefined
    ? "N/A"
    : String(value);
}

function evaluationDescription(model) {
  const details = [];

  if (model.threshold !== null && model.threshold !== undefined) {
    details.push(`Threshold: ${model.threshold}`);
  }

  for (const [field, label] of [
    ["train_transactions", "Train transactions"],
    ["test_transactions", "Test transactions"],
    ["test_accounts", "Test accounts"],
    ["test_frauds", "Test frauds"],
    ["test_fraud", "Test frauds"],
  ]) {
    const value = model.evaluation?.[field];
    if (value !== null && value !== undefined) {
      details.push(`${label}: ${value}`);
    }
  }

  return details.length > 0
    ? details.join(" · ")
    : "Evaluation dataset metadata not present in the source file.";
}

export default function Performance() {
  const [summary, setSummary] = useState(null);
  const [modelMetrics, setModelMetrics] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  useEffect(() => {
    let active = true;

    async function loadSummary() {
      try {
        setLoading(true);
        setError("");

        const [summaryResponse, metricsResponse] = await Promise.all([
          api.get("/summary"),
          api.get("/model-metrics"),
        ]);

        if (active) {
          setSummary(summaryResponse.data);
          setModelMetrics(metricsResponse.data.models || []);
        }
      } catch (err) {
        console.error("Failed to load performance data:", err);

        if (active) {
          setError(
            err?.response?.data?.detail ||
              "Unable to load performance data."
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

      const [summaryResponse, metricsResponse] = await Promise.all([
        api.get("/summary"),
        api.get("/model-metrics"),
      ]);
      setSummary(summaryResponse.data);
      setModelMetrics(metricsResponse.data.models || []);
    } catch (err) {
      setError(
        err?.response?.data?.detail ||
          "Unable to load performance data."
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

  const totalDatasetTransactions = Number(
    summary.dataset.total_transactions
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
            Total Dataset Transactions
          </div>
          <strong>
            {totalDatasetTransactions.toLocaleString("en-IN")}
          </strong>
          <span className="performance-metric-description">
            Transactions in the complete cleaned dataset
          </span>
        </article>

        <article className="performance-metric-card">
          <div className="performance-metric-label">
            Transactions Scored
          </div>
          <strong>{totalTransactions.toLocaleString("en-IN")}</strong>
          <span className="performance-metric-description">
            Transactions with final risk-fusion scores
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

      <section className="performance-panel performance-model-metrics">
        <div className="performance-panel-header">
          <div>
            <h2>Model Evaluation Metrics</h2>
            <p>
              Offline evaluation results read from the saved backend CSV
              files. Values are shown at the precision stored in each file
              and are not live transaction risk scores.
            </p>
          </div>
        </div>

        <div className="performance-model-metrics-table-wrapper">
          <table className="performance-model-metrics-table">
            <thead>
              <tr>
                <th>Model</th>
                {evaluationMetricColumns.map(([, label]) => (
                  <th key={label}>{label}</th>
                ))}
                <th>Evaluation setup</th>
              </tr>
            </thead>
            <tbody>
              {modelMetrics.map((model) => (
                <tr key={model.key}>
                  <th scope="row">
                    <span>{model.model_name}</span>
                    <small>
                      {model.available
                        ? model.source_file
                        : "Evaluation file unavailable"}
                    </small>
                  </th>
                  {evaluationMetricColumns.map(([field]) => (
                    <td key={field}>
                      {displayMetric(model.metrics?.[field])}
                    </td>
                  ))}
                  <td>{evaluationDescription(model)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        {modelMetrics.find((model) => model.key === "aml_rules")
          ?.rules?.length > 0 && (
          <div className="performance-rule-metrics">
            <h3>AML Rule Detection — Individual Rules</h3>
            <div className="performance-model-metrics-table-wrapper">
              <table className="performance-model-metrics-table">
                <thead>
                  <tr>
                    <th>Rule</th>
                    {evaluationMetricColumns.map(([, label]) => (
                      <th key={label}>{label}</th>
                    ))}
                    <th>Test transactions</th>
                  </tr>
                </thead>
                <tbody>
                  {modelMetrics
                    .find((model) => model.key === "aml_rules")
                    .rules.map((rule) => (
                      <tr key={rule.rule_name}>
                        <th scope="row">{rule.rule_name}</th>
                        {evaluationMetricColumns.map(([field]) => (
                          <td key={field}>
                            {displayMetric(rule.metrics?.[field])}
                          </td>
                        ))}
                        <td>
                          {displayMetric(
                            rule.evaluation?.test_transactions
                          )}
                        </td>
                      </tr>
                    ))}
                </tbody>
              </table>
            </div>
          </div>
        )}
      </section>

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
                    {item.percentage.toFixed(2)}% of scored transactions
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
                scored and categorized
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
          measurements. The model evaluation metrics above are offline
          evaluation snapshots from the source CSVs.
        </div>
      </section>
    </section>
  );
}