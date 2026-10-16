# AegisGraph — API Reference

## Base URL

```
http://127.0.0.1:8000/api
```

## Health

### GET /
Root health check.

**Response:**
```json
{ "service": "AegisGraph API", "status": "running", "version": "1.1.0" }
```

### GET /api/health
Service health status.

**Response:**
```json
{ "status": "healthy", "service": "aegisgraph" }
```

---

## Summary

### GET /api/summary
Project-wide statistics.

**Response:**
```json
{
  "risk": { "total_transactions": 2040, "categories": { "HIGH": 310, "MEDIUM": 420 } },
  "alerts": { "total": 730 },
  "investigation": { "total_cases": 180, "total_dossiers": 180, "priority": { ... } }
}
```

### GET /api/risk-summary
Risk score distribution.

**Response:**
```json
{
  "total_transactions": 2040,
  "risk_categories": { "HIGH": 310, "MEDIUM": 420, "LOW": 810, "VERY_LOW": 500 },
  "average_risk_score": 0.3421,
  "maximum_risk_score": 0.9876
}
```

---

## Transactions

### GET /api/transactions
Paginated transaction list with filters.

**Query Parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| page | int | 1 | Page number |
| page_size | int | 25 | Items per page (max 100) |
| search | string | "" | Search transaction/account IDs |
| typology | string | "" | Filter by typology |
| status | string | "" | Filter by status |
| fraud_only | bool | false | Show only fraud transactions |

**Response:**
```json
{
  "transactions": [ ... ],
  "pagination": { "page": 1, "page_size": 25, "total": 10197, "total_pages": 408 },
  "filters": { "search": "", "typology": "", "status": "", "fraud_only": false }
}
```

---

## Alerts

### GET /api/alerts
All alerts.

### GET /api/alerts/{alert_id}
Single alert by ID.

---

## Graph Investigation

### GET /api/graph/{alert_id}
Transaction neighborhood graph for an alert.

**Response:**
```json
{
  "alert_id": "ALT000001",
  "investigated_transaction": { ... },
  "network": {
    "nodes": [ ... ],
    "links": [ ... ],
    "total_nodes": 15,
    "total_links": 28,
    "transaction_count": 12,
    "truncated": false,
    "source": "transaction_graph.graphml"
  },
  "alert": { ... }
}
```

---

## Cases

### GET /api/cases
All investigation cases.

### GET /api/cases/{case_id}
Case dossier by ID.

---

## Reviews

### GET /api/reviews
All case reviews.

---

## Dossiers

### GET /api/dossiers
All investigation dossiers.

---

## Case Management

### GET /api/case-management/{case_id}
Managed case record.

### PATCH /api/case-management/{case_id}/status
Update case lifecycle status.

**Request Body:**
```json
{ "target_status": "UNDER_REVIEW" }
```

**Valid Statuses:** OPEN, UNDER_REVIEW, ESCALATED, CLOSED

**Valid Transitions:**
- OPEN → UNDER_REVIEW, ESCALATED, CLOSED
- UNDER_REVIEW → OPEN, ESCALATED, CLOSED
- ESCALATED → UNDER_REVIEW, CLOSED
- CLOSED → (terminal, no transitions)

### POST /api/case-management/{case_id}/reviews
Submit a case review.

**Request Body:**
```json
{
  "action": "MARK_AS_FRAUD",
  "reviewer": "analyst_name",
  "comments": "Evidence supports fraud determination."
}
```

**Valid Actions:** MARK_AS_FRAUD, MARK_AS_LEGITIMATE, REQUEST_MORE_INFORMATION, ESCALATE, CLOSE

---

## Error Responses

All errors follow the standard format:
```json
{ "detail": "Error message describing what went wrong." }
```

| Status | Meaning |
|--------|---------|
| 400 | Bad request / invalid input |
| 404 | Resource not found |
| 500 | Server error |
