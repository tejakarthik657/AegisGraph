export default function AlertStatusBadge({ status = "open" }) { return <span className={`status-badge status-${status}`}>{status}</span>; }
