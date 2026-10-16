import { useEffect, useState } from "react";

import { getSummary } from "../services/api";

const activityPatterns = [
  {
    name: "Rapid Movement",
    count: 20,
    description: "Fast movement of funds through accounts.",
  },
  {
    name: "Fan-out",
    count: 13,
    description: "Funds distributed to multiple accounts.",
  },
  {
    name: "Structuring",
    count: 12,
    description:
      "Transactions showing possible threshold avoidance.",
  },
  {
    name: "Large Transaction",
    count: 10,
    description:
      "Transactions exceeding the configured amount threshold.",
  },
  {
    name: "Circular Movement",
    count: 3,
    description:
      "Funds moving through a circular account relationship.",
  },
];

function MetricCard({
  label,
  value,
  description,
  className = "",
}) {
  return (
    <div className={`dashboard-metric-card ${className}`}>
      <div className="metric-label">
        {label}
      </div>

      <div className="metric-value">
        {value.toLocaleString()}
      </div>

      <div className="metric-description">
        {description}
      </div>
    </div>
  );
}

function RiskOverview({ categories, total }) {
  const riskLevels = [
    {
      label: "Very Low Risk",
      key: "VERY_LOW",
      description:
        "Transactions with minimal detected risk.",
      className: "risk-very-low",
    },
    {
      label: "Low Risk",
      key: "LOW",
      description:
        "Transactions with limited risk signals.",
      className: "risk-low",
    },
    {
      label: "Medium Risk",
      key: "MEDIUM",
      description:
        "Transactions requiring analyst review.",
      className: "risk-medium",
    },
    {
      label: "High Risk",
      key: "HIGH",
      description:
        "Transactions with significant risk signals.",
      className: "risk-high",
    },
  ];

  return (
    <section className="dashboard-section">
      <div className="section-heading">
        <div>
          <h2>Risk Overview</h2>

          <p>
            How the analyzed transactions are distributed
            across the current risk levels.
          </p>
        </div>
      </div>

      <div className="risk-grid">
        {riskLevels.map((risk) => {
          const count = categories[risk.key] || 0;

          const percentage =
            total > 0
              ? ((count / total) * 100).toFixed(1)
              : "0.0";

          return (
            <div
              className={`risk-card ${risk.className}`}
              key={risk.key}
            >
              <div className="risk-card-top">
                <span className="risk-indicator" />

                <span className="risk-label">
                  {risk.label}
                </span>
              </div>

              <div className="risk-count">
                {count.toLocaleString()}
              </div>

              <div className="risk-percentage">
                {percentage}% of analyzed transactions
              </div>

              <div className="risk-description">
                {risk.description}
              </div>
            </div>
          );
        })}
      </div>
    </section>
  );
}

function ActivityPatterns() {
  return (
    <section className="dashboard-section">
      <div className="section-heading">
        <div>
          <h2>Detected Activity Patterns</h2>

          <p>
            AML detection signals that contributed to
            the current investigation workload.
          </p>
        </div>
      </div>

      <div className="activity-panel">
        {activityPatterns.map((pattern) => (
          <div
            className="activity-row"
            key={pattern.name}
          >
            <div className="activity-information">
              <div className="activity-name">
                {pattern.name}
              </div>

              <div className="activity-description">
                {pattern.description}
              </div>
            </div>

            <div className="activity-count">
              {pattern.count}
            </div>
          </div>
        ))}
      </div>
    </section>
  );
}

function InvestigationFocus({ priority }) {
  return (
    <section className="dashboard-section">
      <div className="section-heading">
        <div>
          <h2>Investigation Focus</h2>

          <p>
            What the analyst should pay attention to.
          </p>
        </div>
      </div>

      <div className="focus-panel">
        <div className="focus-item">
          <div className="focus-number">
            {priority.HIGH}
          </div>

          <div>
            <div className="focus-title">
              High-risk cases
            </div>

            <div className="focus-description">
              These cases represent the highest current
              priority for analyst review.
            </div>
          </div>
        </div>

        <div className="focus-divider" />

        <div className="focus-item">
          <div className="focus-number">
            {priority.MEDIUM}
          </div>

          <div>
            <div className="focus-title">
              Medium-risk cases
            </div>

            <div className="focus-description">
              These cases contain meaningful risk signals
              and require further investigation.
            </div>
          </div>
        </div>

        <div className="focus-divider" />

        <div className="focus-item">
          <div className="focus-number">
            {priority.HIGH + priority.MEDIUM}
          </div>

          <div>
            <div className="focus-title">
              Total open investigations
            </div>

            <div className="focus-description">
              Current medium- and high-risk alerts requiring
              analyst attention.
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}

function DashboardLoading() {
  return (
    <section className="page-content dashboard-page">
      <div className="dashboard-loading">
        <div className="loading-spinner" />

        <div>
          <div className="loading-title">
            Loading AegisGraph dashboard
          </div>

          <div className="loading-description">
            Connecting to the detection and investigation
            services...
          </div>
        </div>
      </div>
    </section>
  );
}

function DashboardError({ message }) {
  return (
    <section className="page-content dashboard-page">
      <div className="dashboard-error">
        <div className="error-title">
          Dashboard data unavailable
        </div>

        <div className="error-description">
          AegisGraph could not retrieve the current
          summary from the backend.
        </div>

        <div className="error-details">
          {message}
        </div>

        <div className="error-help">
          Make sure the FastAPI server is running at
          127.0.0.1:8000.
        </div>
      </div>
    </section>
  );
}

export default function Dashboard() {
  const [summary, setSummary] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  useEffect(() => {
    let mounted = true;

    async function loadSummary() {
      try {
        setLoading(true);
        setError("");

        const data = await getSummary();

        if (mounted) {
          setSummary(data);
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

    loadSummary();

    return () => {
      mounted = false;
    };
  }, []);

  if (loading) {
    return <DashboardLoading />;
  }

  if (error || !summary) {
    return (
      <DashboardError
        message={
          error || "No summary data was returned."
        }
      />
    );
  }

  const totalTransactions =
    summary.risk.total_transactions;

  const riskCategories =
    summary.risk.categories;

  const totalAlerts =
    summary.alerts.total;

  const totalCases =
    summary.investigation.total_cases;

  const priority =
    summary.investigation.priority;

  return (
    <section className="page-content dashboard-page">
      <div className="page-header dashboard-header">
        <div>
          <div className="page-eyebrow">
            AEGISGRAPH OVERVIEW
          </div>

          <h1>Dashboard</h1>

          <p>
            Overview of transaction activity, risk and
            investigations.
          </p>
        </div>

        <div className="dashboard-status">
          <span className="status-dot" />

          Detection pipeline operational
        </div>
      </div>

      <div className="dashboard-metrics">
        <MetricCard
          label="Transactions Analyzed"
          value={totalTransactions}
          description="Transactions evaluated by the current risk detection pipeline."
        />

        <MetricCard
          label="Alerts Requiring Investigation"
          value={totalAlerts}
          description="Medium- and high-risk transactions currently requiring review."
          className="metric-alert"
        />

        <MetricCard
          label="High-Risk Transactions"
          value={riskCategories.HIGH}
          description="Transactions classified as high risk after signal fusion."
          className="metric-high"
        />

        <MetricCard
          label="Open Investigation Cases"
          value={totalCases}
          description="Investigation cases currently open for analyst review."
          className="metric-case"
        />
      </div>

      <RiskOverview
        categories={riskCategories}
        total={totalTransactions}
      />

      <div className="dashboard-two-column">
        <ActivityPatterns />

        <InvestigationFocus
          priority={priority}
        />
      </div>
    </section>
  );
}