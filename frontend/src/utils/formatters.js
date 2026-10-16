/**
 * AegisGraph — Formatting Utilities.
 */

/**
 * Format a number as USD currency.
 */
export function formatCurrency(value) {
  if (value == null || isNaN(value)) return "$0.00";
  return new Intl.NumberFormat("en-US", {
    style: "currency",
    currency: "USD",
    minimumFractionDigits: 2,
    maximumFractionDigits: 2,
  }).format(value);
}

/**
 * Format a number with commas.
 */
export function formatNumber(value) {
  if (value == null || isNaN(value)) return "0";
  return new Intl.NumberFormat("en-US").format(value);
}

/**
 * Format a decimal as percentage.
 */
export function formatPercent(value, decimals = 1) {
  if (value == null || isNaN(value)) return "0%";
  return `${(value * 100).toFixed(decimals)}%`;
}

/**
 * Format a risk score (0-1) with 4 decimal places.
 */
export function formatRiskScore(value) {
  if (value == null || isNaN(value)) return "0.0000";
  return parseFloat(value).toFixed(4);
}

/**
 * Format a timestamp string for display.
 */
export function formatTimestamp(timestamp) {
  if (!timestamp) return "—";
  try {
    const date = new Date(timestamp);
    if (isNaN(date.getTime())) return timestamp;
    return date.toLocaleString("en-US", {
      year: "numeric",
      month: "short",
      day: "numeric",
      hour: "2-digit",
      minute: "2-digit",
    });
  } catch {
    return timestamp;
  }
}

/**
 * Format a date string (date only).
 */
export function formatDate(timestamp) {
  if (!timestamp) return "—";
  try {
    const date = new Date(timestamp);
    if (isNaN(date.getTime())) return timestamp;
    return date.toLocaleDateString("en-US", {
      year: "numeric",
      month: "short",
      day: "numeric",
    });
  } catch {
    return timestamp;
  }
}

/**
 * Truncate a string to a maximum length.
 */
export function truncate(str, maxLength = 24) {
  if (!str) return "";
  if (str.length <= maxLength) return str;
  return str.slice(0, maxLength) + "…";
}

/**
 * Capitalize first letter.
 */
export function capitalize(str) {
  if (!str) return "";
  return str.charAt(0).toUpperCase() + str.slice(1).toLowerCase();
}

/**
 * Format risk category for display.
 */
export function formatRiskCategory(category) {
  if (!category) return "Unknown";
  return category.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase());
}
