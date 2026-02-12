# API Specification – Chest X-ray Disease Detection System (cxr-detect)

## 1. Purpose

This document defines the **REST API contract** for the Chest X-ray Disease Detection System (cxr-detect). It specifies endpoints, request/response formats, validation rules, and error handling, following **OpenAPI-style best practices**.

The API is designed to be:

* Simple and predictable
* ML-friendly
* Easy to integrate with web or programmatic clients
* Production-ready in structure

---

## 2. API Overview

**Base URL (local):**

```
http://localhost:8000
```

**Protocol:** HTTP / HTTPS
**Format:** JSON
**Framework:** FastAPI

---

## 3. Authentication & Authorization

**Current version:**

* No authentication (educational / MVP)

**Future options:**

* API key authentication
* OAuth2 / JWT

---

## 4. Common Conventions

### 4.1 Request Headers

| Header       | Required | Description                      |
| ------------ | -------- | -------------------------------- |
| Content-Type | Yes      | `multipart/form-data` for images |
| Accept       | No       | `application/json`               |

---

### 4.2 Response Structure (Standard)

```json
{
  "status": "success",
  "data": {},
  "request_id": "uuid"
}
```

**On error:**

```json
{
  "status": "error",
  "error": {
    "code": "INVALID_INPUT",
    "message": "Unsupported image format"
  },
  "request_id": "uuid"
}
```

---

## 5. Endpoints

---

### 5.1 Health Check

**Endpoint:**

```
GET /health
```

**Purpose:**

* Verify service availability
* Used for monitoring and orchestration

**Response (200 OK):**

```json
{
  "status": "ok"
}
```

---

### 5.2 Model Information

**Endpoint:**

```
GET /model-info
```

**Purpose:**

* Retrieve metadata about the currently loaded model

**Response (200 OK):**

```json
{
  "model_name": "MobileNetV2",
  "version": "1.0.0",
  "input_shape": [224, 224, 3],
  "classes": ["Normal", "Pneumonia", "COVID-19"]
}
```

---

### 5.3 Predict Disease

**Endpoint:**

```
POST /predict
```

**Purpose:**

* Perform chest X-ray disease prediction

**Request:**

* Content-Type: `multipart/form-data`
* Single image file

**Request Body:**

| Field | Type  | Required | Description                 |
| ----- | ----- | -------- | --------------------------- |
| file  | Image | Yes      | Chest X-ray image (PNG/JPG) |

---

**Success Response (200 OK):**

```json
{
  "status": "success",
  "data": {
    "prediction": "Pneumonia",
    "confidence": 0.91,
    "all_scores": {
      "Normal": 0.05,
      "Pneumonia": 0.91,
      "COVID-19": 0.04
    }
  },
  "request_id": "uuid"
}
```

---

## 6. Validation Rules

### 6.1 File Validation

* Supported formats: PNG, JPG, JPEG
* Maximum file size: configurable (e.g., 5 MB)
* Image must be readable

---

### 6.2 Input Constraints

* One image per request
* No batch inference (v1)

---

## 7. Error Handling

### 7.1 Error Codes

| Code           | Description                 |
| -------------- | --------------------------- |
| INVALID_INPUT  | Invalid or unsupported file |
| FILE_TOO_LARGE | File exceeds size limit     |
| MODEL_ERROR    | Inference failure           |
| INTERNAL_ERROR | Unexpected server error     |

---

### 7.2 Example Error Response

```json
{
  "status": "error",
  "error": {
    "code": "MODEL_ERROR",
    "message": "Model inference failed"
  },
  "request_id": "uuid"
}
```

---

## 8. Status Codes

| Status Code | Meaning               |
| ----------- | --------------------- |
| 200         | Success               |
| 400         | Bad request           |
| 413         | Payload too large     |
| 422         | Validation error      |
| 500         | Internal server error |

---

## 9. Rate Limiting

**Current version:**

* Not enforced

**Future:**

* Per-IP request limits
* Throttling

---

## 10. Logging & Traceability

**Logged per request:**

* Request ID
* Timestamp
* Endpoint
* Processing time

---

## 11. API Versioning Strategy

* URL-based versioning (e.g., `/v1/predict`)
* Backward compatibility for minor changes

---

## 12. OpenAPI / Swagger Support

* Auto-generated docs via FastAPI
* Available at `/docs` and `/redoc`

---

## 13. Security Considerations

* Input validation
* File size limits
* No image persistence by default

---

## 14. Future Enhancements

* Batch prediction endpoint
* Authentication & authorization
* Async job-based inference

---

## 15. Related Documents

* `system_design.md`
* `ml_pipeline.md`
* `data_pipeline.md`

---

**Document version:** 1.0
**Last updated:** 2026-01-24
