# Data Pipeline – Chest X-ray Disease Detection System (cxr-detect)

## 1. Purpose

This document defines the **data pipeline design** for the Chest X-ray Disease Detection System (cxr-detect). It explains how data is collected, validated, processed, stored, and prepared for machine learning, following **good data engineering and ML best practices**.

The data pipeline is designed to ensure:

* Data quality and consistency
* Reproducibility
* Prevention of data leakage
* Traceability from raw data to model inputs

---

## 2. Data Pipeline Overview

The data pipeline supports both **offline training** and **future online data collection**.

```
Raw Data → Validation → Cleaning → Preprocessing → Split → Storage → Training / Evaluation
```

---

## 3. Data Sources

### 3.1 Training Data Sources

**Primary sources:**

* Public chest X-ray datasets

  * Pneumonia detection datasets
  * COVID-19 chest X-ray datasets
  * Normal chest X-ray images

**Formats:**

* Image files: PNG, JPG, JPEG
* Optional metadata: CSV files

---

### 3.2 Inference-Time Data

**Source:**

* User-uploaded chest X-ray images via API

**Constraints:**

* Single image per request (v1)
* No long-term storage by default

---

## 4. Data Ingestion

### 4.1 Raw Data Collection

**Responsibilities:**

* Download datasets
* Verify dataset licenses
* Preserve original directory structure

**Best practice:**

> Raw data must remain **immutable** and never modified.

---

### 4.2 Data Cataloging

**Stored information:**

* Dataset name and version
* Number of samples per class
* Source URL
* Date of ingestion

---

## 5. Data Validation

### 5.1 File-Level Validation

**Checks:**

* File readability
* Valid image encoding
* Supported formats
* Non-zero file size

---

### 5.2 Label Validation

**Checks:**

* One label per image
* Valid class names
* No missing labels

---

### 5.3 Dataset-Level Validation

**Checks:**

* Class imbalance detection
* Duplicate images
* Corrupted samples

---

## 6. Data Cleaning

**Operations:**

* Remove corrupted images
* Remove duplicates
* Resolve inconsistent labels

**Output:**

* Cleaned dataset manifest

---

## 7. Data Preprocessing

### 7.1 Image Transformations

**Operations:**

* Resize to fixed dimensions (e.g., 224×224)
* Convert color channels (RGB / grayscale)
* Normalize pixel values

---

### 7.2 Standardization Rules

* Same preprocessing for training and inference
* Config-driven transformations
* Versioned preprocessing pipeline

---

## 8. Data Augmentation (Training Only)

**Techniques:**

* Random rotation
* Horizontal flipping
* Zoom and shift
* Brightness / contrast variation

**Purpose:**

* Improve robustness
* Reduce overfitting

---

## 9. Dataset Splitting

### 9.1 Split Strategy

* Train / Validation / Test split (e.g., 70/15/15)
* Stratified by class

### 9.2 Leakage Prevention

* Patient-level separation (if metadata available)
* No overlap between splits

---

## 10. Data Storage Structure

### 10.1 Local Storage Layout

```
data/
├── raw/
│   ├── dataset_name/
│   └── ...
├── processed/
│   ├── train/
│   ├── val/
│   └── test/
└── metadata/
    ├── dataset_info.json
    └── splits.csv
```

---

### 10.2 Metadata Storage

**Stored metadata:**

* Image paths
* Labels
* Split assignment
* Dataset version

---

## 11. Data Access Layer

**Purpose:**

* Abstract filesystem access
* Provide unified dataset loaders

**Consumers:**

* Training pipeline
* Evaluation pipeline

---

## 12. Data Quality Monitoring

**Metrics:**

* Class distribution
* Missing or corrupted files
* Dataset size changes

**Frequency:**

* At ingestion
* Before training

---

## 13. Privacy & Compliance

**Practices:**

* Use only publicly available datasets
* No personally identifiable information (PII)
* No patient identifiers stored

**Disclaimer:**

* Educational and research use only

---

## 14. Failure Handling

**Scenarios:**

* Corrupted images
* Missing labels
* Invalid formats

**Approach:**

* Log and skip invalid samples
* Maintain audit trail

---

## 15. Future Extensions

* Cloud object storage (S3, GCS)
* Automated data validation (Great Expectations)
* Data versioning tools (DVC)
* Continuous data ingestion

---

## 16. Related Documents

* `system_design.md`
* `ml_pipeline.md`
* `api_spec.md`

---

**Document version:** 1.0
**Last updated:** 2026-01-24
