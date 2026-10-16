export default function CaseStatusBadge({ status = "open" }) { return <span className={`status-badge status-${status}`}>{status}</span>; }
