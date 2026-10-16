/**
 * AegisGraph — Validation Utilities.
 */

/**
 * Validate that a value is a non-empty string.
 */
export function isNonEmptyString(value) {
  return typeof value === "string" && value.trim().length > 0;
}

/**
 * Validate a transaction ID format.
 */
export function isValidTransactionId(id) {
  if (!isNonEmptyString(id)) return false;
  return /^TX\d+$/i.test(id.trim());
}

/**
 * Validate an alert ID format.
 */
export function isValidAlertId(id) {
  if (!isNonEmptyString(id)) return false;
  return /^ALT\d+$/i.test(id.trim());
}

/**
 * Validate a case ID format.
 */
export function isValidCaseId(id) {
  if (!isNonEmptyString(id)) return false;
  return /^CASE\d+$/i.test(id.trim());
}

/**
 * Validate a positive number.
 */
export function isPositiveNumber(value) {
  const num = parseFloat(value);
  return !isNaN(num) && num > 0;
}

/**
 * Validate that a risk score is between 0 and 1.
 */
export function isValidRiskScore(value) {
  const num = parseFloat(value);
  return !isNaN(num) && num >= 0 && num <= 1;
}

/**
 * Validate review form data.
 */
export function validateReviewForm({ action, reviewer, comments }) {
  const errors = {};

  if (!isNonEmptyString(action)) {
    errors.action = "Action is required.";
  }

  if (!isNonEmptyString(reviewer)) {
    errors.reviewer = "Reviewer name is required.";
  }

  if (comments && comments.length > 2000) {
    errors.comments = "Comments must be under 2000 characters.";
  }

  return {
    isValid: Object.keys(errors).length === 0,
    errors,
  };
}
