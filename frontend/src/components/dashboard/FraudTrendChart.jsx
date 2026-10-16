export default function FraudTrendChart({ data = [] }) { return <section className="chart-card"><h2>Fraud trend</h2><pre>{JSON.stringify(data, null, 2)}</pre></section>; }
