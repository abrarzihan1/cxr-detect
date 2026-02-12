# System Design – Chest X-ray Disease Detection System (cxr-detect)

## 1. Purpose

This document describes the **system architecture and design decisions** for the Chest X-ray Disease Detection System (cxr-detect). The goal is to demonstrate **professional-grade ML system design**, covering data flow, model lifecycle, APIs, deployment, and non-functional requirements.

The system is designed to be:

*   Modular and scalable
*   Reproducible and testable
*   Suitable for academic, portfolio, and production-style deployment

***

## 2. High-Level Architecture Overview

**Core idea:**
A user uploads a chest X-ray → backend validates & preprocesses → ML model performs inference → results are returned via API → optional storage & monitoring.

### High-level components

1.  Client (Web / API Consumer)
2.  Backend API (FastAPI)
3.  ML Inference Service
4.  Model & Artifact Store
5.  Data Storage (optional)
6.  Monitoring & Logging

***

## 3. Dataset Specifications

The core machine learning component is trained on the **NIH ChestX-ray14** dataset.

**Dataset Overview:**
*   **Source:** National Institutes of Health (NIH) Clinical Center. 
*   **Scale:** 112,120 frontal-view X-ray images from 30,805 unique patients. 
*   **Resolution:** Native images are 1024x1024 PNGs; typically resized to 224x224 or 256x256 for standard CNN backbones like ResNet or DenseNet.

**Class Labels:**
The dataset supports multi-label classification for 14 common thoracic pathologies plus a "No Finding" class: 
1.  Atelectasis
2.  Cardiomegaly
3.  Effusion
4.  Infiltration
5.  Mass
6.  Nodule
7.  Pneumonia
8.  Pneumothorax
9.  Consolidation
10. Edema
11. Emphysema
12. Fibrosis
13. Pleural_Thickening
14. Hernia

**Data Characteristics & Constraints:**
*   **Label Noise:** Labels were text-mined from radiology reports using NLP, resulting in "weak" labels with estimated accuracy >90%, but not 100% verified by biopsy. 
*   **Class Imbalance:** Significant imbalance exists; "No Finding" appears in >50% of images, while "Hernia" appears in <0.5%.
*   **Split Strategy:** Patient-level splitting is mandatory to prevent data leakage (images from the same patient must not span training and test sets). 

***

## 4. System Architecture Diagram (Logical)

```
+------------------+        HTTP/HTTPS        +---------------------+
|  Web / API User  |  -------------------->  |   FastAPI Backend   |
+------------------+                          |  (REST Interface)  |
                                              +----------+----------+
                                                         |
                                                         |
                                                Preprocessing Layer
                                               (Resize to 224x224)
                                                         |
                                                         v
                                              +----------+----------+
                                              |   ML Inference      |
                                              | (CNN: DenseNet121)  |
                                              +----------+----------+
                                                         |
                                                         v
                                              +----------+----------+
                                              | Post-processing     |
                                              | (Sigmoid -> Labels) |
                                              +----------+----------+
                                                         |
                                                         v
                                              +----------+----------+
                                              | Response Formatter  |
                                              +---------------------+

Optional:
+------------------+     +------------------+     +------------------+
| Model Registry   |     | Image Storage    |     | Monitoring       |
| (Local / Cloud)  |     | (S3 / Local FS)  |     | (Logs, Metrics)  |
+------------------+     +------------------+     +------------------+
```

***

## 5. Component-Level Design

### 5.1 Client Layer

**Responsibilities:**
*   Upload chest X-ray images
*   Display predictions & confidence scores
*   Handle validation feedback

**Possible clients:**
*   React web app
*   Postman / cURL
*   Python client

**Input format:**
*   PNG / JPG image
*   Single image per request (v1)

***

### 5.2 Backend API (FastAPI)

**Responsibilities:**
*   API routing
*   Input validation
*   Request orchestration
*   Security controls

**Key endpoints (example):**
*   `POST /predict`
*   `GET /health`
*   `GET /model-info`

**Design principles:**
*   Thin controllers
*   Business logic separated from routes
*   Async-ready

***

### 5.3 Preprocessing Module

**Responsibilities:**
*   **Resolution Standardization:** Resize inputs to 224x224 or 256x256 (matching NIH ChestX-ray14 training dimensions). 
*   **Normalization:** Rescale pixel intensity (0-1 or -1 to 1) using ImageNet mean/std if transfer learning was used.
*   **Format Handling:** Convert 1-channel (grayscale) or 4-channel (RGBA) inputs to 3-channel RGB as required by standard backbones.

**Design notes:**
*   Deterministic transformations
*   Version-controlled preprocessing
*   Same logic used during training & inference

***

### 5.4 ML Inference Service

**Model:**
*   **Architecture:** DenseNet121 or ResNet50 (Standard benchmarks for NIH dataset).
*   **Output:** 14-element vector (Multi-label classification).

**Responsibilities:**
*   Load trained model into memory
*   Perform forward pass
*   Return raw logits

**Design considerations:**
*   Model loaded once at startup
*   GPU support (optional)
*   Batch inference extensibility

***

### 5.5 Post-processing Layer

**Responsibilities:**
*   Map logits → probabilities (Sigmoid activation for multi-label).
*   Thresholding (e.g., `p > 0.5`) or top-k selection.
*   Class label mapping (Indices 0-13 to Disease Names).

**Example output:**

```json
{
  "prediction": ["Pneumonia", "Infiltration"],
  "confidence": {
    "Pneumonia": 0.88,
    "Infiltration": 0.65
  },
  "all_scores": {
    "Atelectasis": 0.12,
    "Cardiomegaly": 0.01,
    "...": "..."
  }
}
```

***

### 5.6 Model & Artifact Management

**Artifacts:**
*   Trained model weights (`.pt` / `.h5`)
*   Label encoders (JSON)
*   Preprocessing config

**Storage options:**
*   Local filesystem (MVP)
*   Cloud object storage (future)

**Versioning strategy:**
*   Semantic versioning (v1.0.0)
*   Hash-based integrity checks

***

### 5.7 Data Storage (Optional)

**Stored data (if enabled):**
*   Uploaded images (anonymized)
*   Prediction results
*   Request metadata

**Purpose:**
*   Auditing
*   Model evaluation
*   Dataset expansion

***

### 5.8 Logging & Monitoring

**Logging:**
*   Request IDs
*   Errors & exceptions
*   Model inference time

**Metrics:**
*   Latency
*   Throughput
*   Error rates

**Tools:**
*   Python logging
*   Prometheus / Grafana (future)

***

## 6. ML Lifecycle Integration

### Training Pipeline (Offline)

1.  **Ingestion:** Load NIH ChestX-ray14 images and CSV metadata.
2.  **Splitting:** Stratified split by *Patient ID* (Train/Val/Test).
3.  **Augmentation:** Random rotation, flip, crop (to handle class imbalance). 
4.  **Training:** Multi-label Binary Cross Entropy Loss (often weighted).
5.  **Evaluation:** AUC-ROC per class (due to imbalance).

### Inference Pipeline (Online)

1.  Image upload
2.  Preprocessing (Resize/Normalize)
3.  Inference
4.  Post-processing (Sigmoid + Threshold)
5.  Response

***

## 7. Non-Functional Requirements

### Performance
*   Target latency: < 500 ms per image (CPU)
*   Scalable via horizontal API replicas

### Reliability
*   Health checks
*   Graceful failure handling

### Security
*   Input validation
*   File size limits
*   No PHI persistence by default

### Maintainability
*   Modular codebase
*   Clear interfaces
*   High test coverage

***

## 8. Deployment Architecture (Initial)

```
Client → FastAPI (Uvicorn) → ML Model (Local)
```

**Containerization:**
*   Docker
*   Separate build stages

**Future extensions:**
*   Kubernetes
*   Model-serving frameworks
*   CI/CD pipelines

***

## 9. Assumptions & Constraints

*   **Dataset Limitation:** The model is trained on **weakly labeled data** (NLP-mined). Predictions should be treated as "screening" suggestions, not biopsy-confirmed diagnoses. 
*   **Class Bias:** Rare classes (e.g., Hernia) may have lower sensitivity due to extreme dataset imbalance. 
*   **Scope:** Educational / portfolio-focused; not a certified medical device.

***

## 10. Future Improvements

*   **Weighted Loss Functions:** Implement Focal Loss to better handle the NIH dataset's class imbalance. 
*   **Model Explainability:** Integrate Grad-CAM to visualize which image regions triggered the disease classification.
*   **Dataset Expansion:** Augment NIH data with COVID-19 specific datasets (e.g., COVID-19 Radiography Database) to add viral pneumonia capabilities.
*   **Authentication & rate limiting.**

***

## 11. Related Documents

*   `requirements.md`
*   `data_pipeline.md`
*   `ml_pipeline.md`
*   `api_spec.md`

***

**Document version:** 1.1
**Last updated:** 2026-01-25