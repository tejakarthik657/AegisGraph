import { Link } from "react-router-dom";

export default function Breadcrumbs({ items = [] }) {
  return <nav aria-label="Breadcrumbs">{items.map((item, index) => <span key={`${item.label}-${index}`}>{index > 0 && " / "}{item.to ? <Link to={item.to}>{item.label}</Link> : item.label}</span>)}</nav>;
}
