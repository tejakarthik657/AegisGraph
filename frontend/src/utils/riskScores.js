const RISK_SCORE_FIELDS = [
  "rf_probability",
  "tgn_probability",
  "thgnn_probability",
  "rule_score",
  "risk_score",
  "risk_category",
];

export function getScorePercentage(value) {
  if (value === null || value === undefined || value === "") {
    return null;
  }

  const score = Number(value);
  return Number.isFinite(score) ? score * 100 : null;
}

export function formatScorePercentage(value) {
  const percentage = getScorePercentage(value);
  return percentage === null
    ? "Not scored"
    : `${percentage.toFixed(2)}%`;
}

export function mergeTransactionRiskScores(record, transaction) {
  if (!transaction) {
    return record;
  }

  return {
    ...record,
    ...Object.fromEntries(
      RISK_SCORE_FIELDS.map((field) => [
        field,
        transaction[field],
      ])
    ),
  };
}
