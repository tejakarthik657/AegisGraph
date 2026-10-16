export default function EmptyState({ title = "No data", description = "" }) {
  return <div className="empty-state"><strong>{title}</strong>{description && <p>{description}</p>}</div>;
}
