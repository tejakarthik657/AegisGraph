import { formatScorePercentage } from "../../utils/riskScores";

const evidence = [
  {
    field: "rf_probability",
    title: "Random Forest",
    description: "Probability assigned by the Random Forest model.",
    className: "evidence-blue",
  },
  {
    field: "tgn_probability",
    title: "Temporal Graph Network (TGN)",
    description: "Probability assigned by the temporal graph network.",
    className: "evidence-purple",
  },
  {
    field: "thgnn_probability",
    title: "THGNN",
    description:
      "Probability assigned by the Temporal Heterogeneous Graph Neural Network.",
    className: "evidence-teal",
  },
  {
    field: "rule_score",
    title: "AML Rule Score",
    description: "Combined score from triggered AML detection rules.",
    className: "evidence-orange",
  },
  {
    field: "risk_score",
    title: "Final Risk Score",
    description: "Final risk score returned by the backend.",
    className: "evidence-red",
  },
];

function EvidenceCard({ title, value, description, className }) {
  return (
    <div className={`evidence-card ${className}`}>
      <div className="evidence-card-title">{title}</div>
      <div className="evidence-card-value">{value}</div>
      <div className="evidence-card-description">{description}</div>
    </div>
  );
}

export default function RiskEvidenceCards({ scores }) {
  return (
    <div className="evidence-grid">
      {evidence.map((item) => (
        <EvidenceCard
          key={item.field}
          title={item.title}
          value={formatScorePercentage(scores[item.field])}
          description={item.description}
          className={item.className}
        />
      ))}

      <EvidenceCard
        title="Risk Category"
        value={scores.risk_category || "Not scored"}
        description="Risk classification returned by the backend."
        className="evidence-gray"
      />
    </div>
  );
}
