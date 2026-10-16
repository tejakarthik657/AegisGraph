export default function RiskDistributionChart({ data = [] }) { return <section className="chart-card"><h2>Risk distribution</h2><pre>{JSON.stringify(data, null, 2)}</pre></section>; }
