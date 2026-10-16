export default function TransactionVolumeChart({ data = [] }) { return <section className="chart-card"><h2>Transaction volume</h2><pre>{JSON.stringify(data, null, 2)}</pre></section>; }
