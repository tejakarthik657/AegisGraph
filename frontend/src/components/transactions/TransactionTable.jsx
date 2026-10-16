import EmptyState from "../common/EmptyState";

export default function TransactionTable({ transactions = [], onSelect }) {
  if (!transactions.length) return <EmptyState title="No transactions" />;
  return <table><thead><tr><th>ID</th><th>Amount</th><th>Risk</th><th>Status</th></tr></thead><tbody>{transactions.map((row) => <tr key={row.transaction_id || row.id} onClick={() => onSelect?.(row)}><td>{row.transaction_id || row.id}</td><td>{row.amount ?? "—"}</td><td>{row.risk_score ?? row.fraud_probability ?? "—"}</td><td>{row.status ?? "—"}</td></tr>)}</tbody></table>;
}
