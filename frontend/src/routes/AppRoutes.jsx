import { Navigate, Route, Routes } from "react-router-dom";
import Dashboard from "../pages/Dashboard";
import Transactions from "../pages/Transactions";
import TransactionDetailsPage from "../pages/TransactionDetailsPage";
import Alerts from "../pages/Alerts";
import AlertDetailsPage from "../pages/AlertDetailsPage";
import GraphExplorer from "../pages/GraphExplorer";
import Cases from "../pages/Cases";
import CaseDetailsPage from "../pages/CaseDetailsPage";
import Reviews from "../pages/Reviews";
import InvestigationDossierPage from "../pages/InvestigationDossierPage";
import ModelPerformance from "../pages/ModelPerformance";
import HealthStatus from "../pages/HealthStatus";
import NotFound from "../pages/NotFound";

export default function AppRoutes() {
  return <Routes><Route path="/" element={<Navigate to="/dashboard" replace />} /><Route path="/dashboard" element={<Dashboard />} /><Route path="/transactions" element={<Transactions />} /><Route path="/transactions/:transactionId" element={<TransactionDetailsPage />} /><Route path="/alerts" element={<Alerts />} /><Route path="/alerts/:alertId" element={<AlertDetailsPage />} /><Route path="/graph" element={<GraphExplorer />} /><Route path="/cases" element={<Cases />} /><Route path="/cases/:caseId" element={<CaseDetailsPage />} /><Route path="/reviews" element={<Reviews />} /><Route path="/dossiers/:dossierId" element={<InvestigationDossierPage />} /><Route path="/models" element={<ModelPerformance />} /><Route path="/health" element={<HealthStatus />} /><Route path="*" element={<NotFound />} /></Routes>;
}
