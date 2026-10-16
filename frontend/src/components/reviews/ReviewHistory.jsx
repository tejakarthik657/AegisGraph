export default function ReviewHistory({ history = [] }) { return <ul>{history.map((item, index) => <li key={item.id || index}>{item.decision || item.action} — {item.timestamp || ""}</li>)}</ul>; }
