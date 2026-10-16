export default function EvidenceList({ evidence = [] }) { return <ul>{evidence.map((item, index) => <li key={item.id || index}>{item.title || item.description || JSON.stringify(item)}</li>)}</ul>; }
