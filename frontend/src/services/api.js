import axios from "axios";

const api = axios.create({
  baseURL: "http://127.0.0.1:8000/api",
  headers: {
    "Content-Type": "application/json",
  },
});

// ============================================================
// HEALTH
// ============================================================

export async function getHealth() {
  const response = await api.get("/health");
  return response.data;
}

// ============================================================
// DASHBOARD / SUMMARY
// ============================================================

export async function getSummary() {
  const response = await api.get("/summary");
  return response.data;
}

export async function getRiskSummary() {
  const response = await api.get("/risk-summary");
  return response.data;
}

// ============================================================
// TRANSACTIONS
// ============================================================

export async function getTransactions(params = {}) {
  const response = await api.get("/transactions", { params });
  return response.data;
}

// ============================================================
// ALERTS
// ============================================================

export async function getAlerts() {
  const response = await api.get("/alerts");
  return response.data;
}

export async function getAlert(alertId) {
  const response = await api.get(`/alerts/${alertId}`);
  return response.data;
}

// ============================================================
// GRAPH
// ============================================================

export async function getInvestigationGraph(alertId) {
  const response = await api.get(`/graph/${alertId}`);
  return response.data;
}

// ============================================================
// CASES
// ============================================================

export async function getCases() {
  const response = await api.get("/cases");
  return response.data;
}

export async function getCase(caseId) {
  const response = await api.get(`/cases/${caseId}`);
  return response.data;
}

// ============================================================
// REVIEWS
// ============================================================

export async function getReviews() {
  const response = await api.get("/reviews");
  return response.data;
}

// ============================================================
// DOSSIERS
// ============================================================

export async function getDossiers() {
  const response = await api.get("/dossiers");
  return response.data;
}

// ============================================================
// CASE MANAGEMENT
// ============================================================

export async function getManagedCase(caseId) {
  const response = await api.get(`/case-management/${caseId}`);
  return response.data;
}

export async function updateCaseStatus(caseId, targetStatus) {
  const response = await api.patch(
    `/case-management/${caseId}/status`,
    { target_status: targetStatus }
  );
  return response.data;
}

export async function submitCaseReview(caseId, reviewData) {
  const response = await api.post(
    `/case-management/${caseId}/reviews`,
    reviewData
  );
  return response.data;
}

export default api;
