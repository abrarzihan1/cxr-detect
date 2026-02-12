# CXR-Detect

Chest X-ray Disease Detection System using Deep Learning.

## Problem Statement
Early detection of diseases from chest X-ray images.

## Objectives
- Build a reliable CNN-based disease detection model
- Provide an API for inference
- Follow professional ML & software engineering practices

## Tech Stack
- Python, PyTorch/TensorFlow
- FastAPI
- Docker
- GitHub Actions (planned)

## System Architecture

```mermaid
flowchart TB
    U["End User"] -->|HTTPS| F["Frontend Web App\nReact"]
    F -->|REST API| B["Backend API\nFastAPI"]
    
    B --> M["ML Inference Layer"]
    
    M --> P["Preprocessing"]
    P --> I["Model Inference"]
    I --> O["Postprocessing and Explainability"]
    
    I --> MS["Model Storage\nVersioned Models"]
    B --> DB["Metadata DB\nPredictions and Logs"]
    
    O --> B
    B --> F
```

## Project Status
🚧 In progress
