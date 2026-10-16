export default function Header() {
  return (
    <header className="top-header">
      <div className="header-context">
        <div className="header-title">
          AML Intelligence Platform
        </div>

        <div className="header-subtitle">
          Transaction monitoring, risk detection and investigation
        </div>
      </div>

      <div className="header-right">
        <div className="environment-badge">
          <span className="status-dot" />
          Synthetic Environment
        </div>

        <div className="user-area">
          <div className="user-avatar">
            A
          </div>

          <div>
            <div className="user-name">
              Analyst
            </div>

            <div className="user-role">
              Investigation Team
            </div>
          </div>
        </div>
      </div>
    </header>
  );
}