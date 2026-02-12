# ML Pipeline – Chest X-ray Disease Detection System (cxr-detect)

## 1. Purpose

This document describes the **end-to-end machine learning pipeline** for the Chest X-ray Disease Detection System (cxr-detect). It covers all stages from raw data ingestion to model deployment and monitoring, following **professional ML engineering and MLOps practices**.

The pipeline is designed to be:

* Reproducible
* Modular
* Experiment-friendly
* Deployment-ready

---

## 2. ML Pipeline Overview

The ML pipeline is divided into two major workflows:

1. **Offline Training Pipeline** – data preparation, training, evaluation, and model packaging
2. **Online Inference Pipeline** – real-time prediction using trained models

```
Raw Data → Preprocessing → Training → Evaluation → Model Registry → Deployment → Inference
```

---

## 3. Offline Training Pipeline

### 3.1 Data Ingestion

**Input sources:**

* Public chest X-ray datasets (e.g., Pneumonia, COVID-19, Normal)
* Images in PNG / JPG format

**Responsibilities:**

* Load images and labels
* Validate file integrity
* Remove corrupted or invalid samples

**Output:**

* Structured dataset (image paths + labels)

---

### 3.2 Data Splitting

**Strategy:**

* Train / Validation / Test split (e.g., 70/15/15)
* Patient-level split (if metadata available)

**Goal:**

* Prevent data leakage
* Ensure unbiased evaluation

---

### 3.3 Data Preprocessing

**Operations:**

* Resize images to model input size (e.g., 224×224)
* Convert to RGB / grayscale
* Normalize pixel values
* Optional histogram equalization

**Key principle:**

> Preprocessing logic must be **identical** for training and inference.

---

### 3.4 Data Augmentation

**Techniques:**

* Random rotation
* Horizontal flip
* Zoom & shift
* Contrast adjustment

**Purpose:**

* Improve generalization
* Reduce overfitting

---

### 3.5 Model Architecture Selection

**Baseline models:**

* MobileNetV2
* ResNet-18 / ResNet-50

**Design choices:**

* Transfer learning with ImageNet weights
* Custom classification head

---

### 3.6 Model Training

**Training loop:**

* Forward pass
* Loss computation (e.g., Cross-Entropy)
* Backpropagation
* Weight updates

**Hyperparameters:**

* Learning rate
* Batch size
* Optimizer (Adam / SGD)
* Number of epochs

**Frameworks:**

* PyTorch or TensorFlow

---

### 3.7 Model Evaluation

**Metrics:**

* Accuracy
* Precision / Recall
* F1-score
* ROC-AUC
* Confusion matrix

**Evaluation datasets:**

* Validation set (during training)
* Test set (final evaluation)

---

### 3.8 Experiment Tracking

**Tracked artifacts:**

* Model checkpoints
* Hyperparameters
* Metrics
* Training logs

**Tools (optional):**

* MLflow
* Weights & Biases
* TensorBoard

---

### 3.9 Model Selection

**Selection criteria:**

* Best validation performance
* Balanced precision/recall
* Stability across epochs

**Output:**

* Selected model for deployment

---

### 3.10 Model Packaging

**Packaged artifacts:**

* Model weights
* Label mappings
* Preprocessing configuration

**Format:**

* `.pt` / `.pth` (PyTorch)
* `.h5` / SavedModel (TensorFlow)

---

## 4. Model Registry

**Purpose:**

* Central storage of trained models
* Version control
* Rollback support

**Versioning strategy:**

* Semantic versioning (v1.0.0)
* Metadata: training date, dataset version

---

## 5. Online Inference Pipeline

### 5.1 Request Handling

* Image uploaded via API
* File validation
* Size and format checks

---

### 5.2 Inference Preprocessing

* Resize
* Normalize
* Tensor conversion

---

### 5.3 Model Inference

* Load model into memory (startup)
* Forward pass
* Raw prediction output

---

### 5.4 Post-processing

* Softmax / sigmoid
* Thresholding
* Class label mapping

---

### 5.5 Response Generation

**Output:**

```json
{
  "prediction": "Pneumonia",
  "confidence": 0.92
}
```

---

## 6. Model Validation in Production

**Checks:**

* Input distribution drift
* Output confidence monitoring
* Error rates

**Purpose:**

* Detect model degradation

---

## 7. Retraining Strategy

**Triggers:**

* New labeled data
* Performance drop
* Periodic retraining

**Approach:**

* Offline retraining
* Model comparison
* Safe redeployment

---

## 8. Reproducibility & Governance

**Practices:**

* Fixed random seeds
* Dataset versioning
* Environment locking (requirements.txt)

---

## 9. Assumptions & Constraints

* Research / educational use
* Limited dataset size
* No real-time learning

---

## 10. Future Enhancements

* Multi-label classification
* Explainability (Grad-CAM)
* Active learning loop
* Full MLOps automation

---

## 11. Related Documents

* `system_design.md`
* `data_pipeline.md`
* `api_spec.md`

---

**Document version:** 1.0
**Last updated:** 2026-01-24
