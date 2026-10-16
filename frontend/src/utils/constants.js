/**
 * AegisGraph — Application Constants.
 */

export const API_BASE_URL = "http://127.0.0.1:8000/api";

export const RISK_CATEGORIES = {
  CRITICAL: { label: "Critical", color: "#ef4444", bg: "rgba(239,68,68,0.12)" },
  HIGH: { label: "High", color: "#f97316", bg: "rgba(249,115,22,0.12)" },
  MEDIUM: { label: "Medium", color: "#eab308", bg: "rgba(234,179,8,0.12)" },
  LOW: { label: "Low", color: "#22c55e", bg: "rgba(34,197,94,0.12)" },
  VERY_LOW: { label: "Very Low", color: "#64748b", bg: "rgba(100,116,139,0.12)" },
};

export const CASE_STATUSES = {
  OPEN: { label: "Open", color: "#3b82f6", bg: "rgba(59,130,246,0.12)" },
  UNDER_REVIEW: { label: "Under Review", color: "#f59e0b", bg: "rgba(245,158,11,0.12)" },
  ESCALATED: { label: "Escalated", color: "#ef4444", bg: "rgba(239,68,68,0.12)" },
  CLOSED: { label: "Closed", color: "#6b7280", bg: "rgba(107,114,128,0.12)" },
};

export const ALERT_STATUSES = {
  OPEN: { label: "Open", color: "#f59e0b" },
  UNDER_REVIEW: { label: "Under Review", color: "#3b82f6" },
  ESCALATED: { label: "Escalated", color: "#ef4444" },
  RESOLVED: { label: "Resolved", color: "#22c55e" },
  DISMISSED: { label: "Dismissed", color: "#6b7280" },
};

export const REVIEW_ACTIONS = [
  { value: "MARK_AS_FRAUD", label: "Mark as Fraud" },
  { value: "MARK_AS_LEGITIMATE", label: "Mark as Legitimate" },
  { value: "REQUEST_MORE_INFORMATION", label: "Request More Information" },
  { value: "ESCALATE", label: "Escalate" },
  { value: "CLOSE", label: "Close" },
];

export const FRAUD_TYPOLOGIES = [
  "smurfing",
  "layering",
  "funnel",
  "fan_out",
  "cycle",
  "none",
];

export const PAGE_SIZES = [10, 25, 50, 100];

export const ROUTES = {
  DASHBOARD: "/dashboard",
  ALERTS: "/alerts",
  TRANSACTIONS: "/transactions",
  CASES: "/cases",
  WORKSPACE: "/workspace",
  MONITORING: "/monitoring",
};
