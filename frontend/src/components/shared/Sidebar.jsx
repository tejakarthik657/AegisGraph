import { NavLink } from "react-router-dom";

const navigationGroups = [
  {
    title: "OVERVIEW",
    items: [
      {
        label: "Dashboard",
        path: "/dashboard",
        icon: "▦",
      },
    ],
  },
  {
    title: "DETECTION",
    items: [
      {
        label: "Alerts",
        path: "/alerts",
        icon: "⚠",
      },
      {
        label: "Transactions",
        path: "/transactions",
        icon: "⇄",
      },
    ],
  },
  {
    title: "INVESTIGATION",
    items: [
      {
        label: "Cases",
        path: "/cases",
        icon: "□",
      },
      {
        label: "Workspace",
        path: "/workspace",
        icon: "⌕",
      },
    ],
  },
  {
    title: "MONITORING",
    items: [
      {
        label: "Performance",
        path: "/monitoring",
        icon: "◉",
      },
    ],
  },
];

export default function Sidebar() {
  return (
    <aside className="sidebar">
      <div className="brand">
        <div className="brand-mark">A</div>

        <div>
          <div className="brand-name">AegisGraph</div>
          <div className="brand-subtitle">
            Financial Crime Intelligence
          </div>
        </div>
      </div>

      <nav className="sidebar-navigation">
        {navigationGroups.map((group) => (
          <div className="navigation-group" key={group.title}>
            <div className="navigation-label">
              {group.title}
            </div>

            {group.items.map((item) => (
              <NavLink
                key={item.path}
                to={item.path}
                className={({ isActive }) =>
                  `navigation-item ${
                    isActive ? "active" : ""
                  }`
                }
              >
                <span className="navigation-icon">
                  {item.icon}
                </span>

                <span>{item.label}</span>
              </NavLink>
            ))}
          </div>
        ))}
      </nav>

      <div className="sidebar-footer">
        <div className="system-status">
          <span className="status-dot" />
          <span>System Online</span>
        </div>

        <div className="version">
          AegisGraph v1.0
        </div>
      </div>
    </aside>
  );
}