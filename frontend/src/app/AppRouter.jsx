import {
  Navigate,
  Route,
  Routes,
} from "react-router-dom";

import Dashboard from "../dashboard/Dashboard";
import Alerts from "../alerts/Alerts";
import AlertDetails from "../alerts/AlertDetails";
import InvestigationWorkspace from "../workspace/InvestigationWorkspace";
import Transactions from "../transactions/Transactions";
import Cases from "../cases/Cases";
import Performance from "../monitoring/Performance";  

export default function AppRouter() {
  return (
    <Routes>

      {/* =====================================================
          DEFAULT
          ===================================================== */}

      <Route
        path="/"
        element={
          <Navigate
            to="/dashboard"
            replace
          />
        }
      />


      {/* =====================================================
          DASHBOARD
          ===================================================== */}

      <Route
        path="/dashboard"
        element={<Dashboard />}
      />


      {/* =====================================================
          ALERTS
          ===================================================== */}

      <Route
        path="/alerts"
        element={<Alerts />}
      />

      <Route
        path="/alerts/:alertId"
        element={<AlertDetails />}
      />


      {/* =====================================================
          INVESTIGATION WORKSPACE
          ===================================================== */}

      <Route
        path="/workspace/:alertId"
        element={
          <InvestigationWorkspace />
        }
      />


      {/* =====================================================
          UNIMPLEMENTED PAGES
          ===================================================== */}

      <Route
        path="/transactions"
        element={<Transactions />}
      />

      <Route
        path="/cases"
        element={<Cases />}
        
      />

      <Route
        path="/monitoring"
        element={
          <Performance />
        }
      />


      {/* =====================================================
          WORKSPACE WITHOUT ALERT ID
          ===================================================== */}

      <Route
        path="/workspace"
        element={
          <Navigate
            to="/alerts"
            replace
          />
        }
      />


      {/* =====================================================
          FALLBACK
          ===================================================== */}

      <Route
        path="*"
        element={
          <Navigate
            to="/dashboard"
            replace
          />
        }
      />

    </Routes>
  );
} 