export default function SummaryCards({ summary = {} }) {
  return <div className="summary-cards">{Object.entries(summary).slice(0, 4).map(([key, value]) => <article key={key}><span>{key.replace(/_/g, " ")}</span><strong>{String(value)}</strong></article>)}</div>;
}
