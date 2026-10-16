import { NavLink } from "react-router-dom";

const links = [["Dashboard", "/dashboard"], ["Transactions", "/transactions"], ["Alerts", "/alerts"], ["Cases", "/cases"], ["Reviews", "/reviews"], ["Models", "/models"]];

export default function Sidebar() {
  return <aside className="sidebar"><div className="brand"><div className="brand-mark">A</div><div><div className="brand-name">AegisGraph</div><div className="brand-subtitle">Financial Crime Intelligence</div></div></div><nav className="sidebar-navigation">{links.map(([label, path]) => <NavLink className="navigation-item" key={path} to={path}>{label}</NavLink>)}</nav><div className="sidebar-footer"><span className="status-dot" /> System Online</div></aside>;
}
