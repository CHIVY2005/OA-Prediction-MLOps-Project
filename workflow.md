# End-to-End Workflow

This document describes the complete workflow from data preparation to model deployment and monitoring in the OA-Prediction-MLOps project.

## Overview

The workflow follows a standard MLOps pipeline with four main phases:
1. Data Engineering & Preparation
2. Model Development & Training
3. Deployment & Serving
4. Monitoring & Maintenance

Each phase has specific steps, tools, and outputs as detailed below.

## Phase 1: Data Engineering & Preparation

### Objective
Prepare clean, structured data for model training with proper handling of class imbalance.

### Steps
1. **Data Collection**: 
   - Source: KneeKL224 dataset from Mendeley (https://data.mendeley.com/datasets/56rmx5bjcr/1)
   - Format: X-ray images organized by KL grade (0-4)

2. **Data Structure**:
   ```
   data/
   └── kneeKL224/
       ├── train/
       │   ├── 0/
       │   ├── 1/
       │   ├── 2/
       │   ├── 3/
       │   └── 4/
       ├── val/
       │   ├── 0/
       │   ├── 1/
       │   ├── 2/
       │   ├── 3/
       │   └── 4/
       └── test/
           ├── 0/
           ├── 1/
           ├── 2/
           ├── 3/
           └── 4/
   ```

3. **Data Processing** (`src/data_loader.py`):
   - Custom `KneeDataset` class loads images and labels
   - Image transforms: resize, normalization, augmentation (training only)
   - Stratified sampling to maintain class distribution
   - WeightedRandomSampler to handle class imbalance
   - Train/validation split with configurable fraction (default 0.02 for fast testing)

### Output
- PyTorch DataLoaders for training and validation
- Class weights for loss function to address imbalance

## Phase 2: Model Development & Training

### Objective
Develop and train an osteoarthritis classification model using transfer learning.

### Steps
1. **Model Selection**:
   - Architecture: EfficientNet-B0 (pretrained on ImageNet)
   - Feature extraction layers frozen for faster training on CPU
   - Custom classifier head for 5-class classification (KL grades 0-4)

2. **Configuration** (`configs/config.yml`):
   ```yaml
   data:
     dataset_name: "kneeKL224"
     num_classes: 5
     class_names: ['0', '1', '2', '3', '4']
     img_size: 224
     fraction: 0.02  # 2% for fast testing, 1.0 for full training
   
   model:
     name: "efficientnet_b0"
     pretrained: true
     freeze_feature_layers: true
   
   train:
     batch_size: 8
     epochs: 1
     learning_rate: 0.0001
     device: "auto"  # auto-detects CUDA
     save_name: "best_knee_model.pth"
   ```

3. **Model Building** (`src/model.py`):
   - Loads pretrained EfficientNet-B0
   - Freezes feature layers if configured
   - Replaces classifier head with dropout and linear layer

4. **Training Process** (`src/train.py`):
   - Loads data via `get_data_loaders()` from data_loader.py
   - Initializes model, loss function (weighted CrossEntropyLoss), optimizer
   - Training loop with validation after each epoch
   - Saves best model based on validation accuracy
   - Logs metrics to MLflow (automatically tracked)
   - Generates training/validation curves

5. **MLflow Tracking**:
   - Automatically logs parameters, metrics, and model artifacts
   - Experiment name: "Knee_Osteoarthritis_Dev"
   - Accessible via `mlflow ui` command pointing to `experiments/mlruns/`

### Output
- Trained model weights saved to `models/best_knee_model.pth`
- MLflow experiment with run metrics and parameters
- Training logs showing loss and accuracy curves

## Phase 3: Deployment & Serving

### Objective
Deploy the trained model as a REST API with a user-friendly web interface.

### Steps
1. **API Development** (`deployment/api/main.py`):
   - FastAPI application with CORS middleware
   - Model loading on startup (singleton pattern)
   - `/predict` endpoint:
     * Accepts image file upload
     * Preprocesses image (resize, normalize)
     * Runs inference to get prediction and confidence
     * Generates Grad-CAM heatmap for explainability
     * Returns JSON with prediction, confidence, and base64-encoded heatmap
   - Health check endpoint (`/`)

2. **Docker Containerization** (`deployment/docker/`):
   - `Dockerfile`:
     * Base image: python:3.9-slim
     * Installs dependencies from requirements.txt
     * Copies source code
     * Exposes port 8000
     * Runs uvicorn server
   - `docker-compose.yml`:
     * Defines api service
     * Maps port 8000:8000
     * Optional volume mounts for development

3. **Web Interface** (`deployment/web_ui/`):
   - Simple HTML interface with drag-and-drop upload
   - JavaScript to:
     * Send image to `/predict` endpoint
     * Display prediction result and confidence
     * Show Grad-CAM heatmap overlay
   - Served via Python's http.server (for simplicity) or can be deployed with Docker/nginx

### Output
- Running API service accessible at http://localhost:8000
- Interactive web UI at http://localhost:3000/web_ui/ (when served separately)
- API documentation at http://localhost:8000/docs (Swagger UI)

## Phase 4: Monitoring & Maintenance

### Objective
Ensure model performance remains stable over time and detect issues early.

### Planned Implementation (Future Work)

1. **Data Drift Detection**:
   - Monitor input data distribution vs. training data
   - Use statistical tests (KS-test, PSI) or ML-based detectors
   - Alert when significant drift detected

2. **Performance Monitoring**:
   - Track prediction confidence distribution
   - Monitor for sudden drops in confidence (possible model degradation)
   - Log prediction latency and throughput

3. **Logging & Alerting**:
   - Structured logging (JSON format) for all API requests
   - Centralized logging solution (ELK stack or similar)
   - Alerts for error rates, latency spikes, or system issues

4. **Explainability Monitoring**:
   - Validate that heatmaps focus on clinically relevant regions
   - Detect when model attends to artifacts or irrelevant features

5. **Automated Retraining Pipeline**:
   - Trigger retraining when data drift exceeds threshold
   - Versioned model registry (MLflow Model Registry)
   - A/B testing framework for comparing model versions
   - Automated deployment pipeline (CI/CD)

6. **Guardrails & Input Validation**:
   - Validate input file type, size, and dimensions
   - Reject inappropriate requests (non-medical images)
   - Rate limiting to prevent abuse
   - Confidence thresholding for low-confidence predictions

### Current Monitoring Capabilities
- Basic error handling in API (returns prediction even if heatmap fails)
- Startup logging shows model loading status
- MLflow tracks training experiments
- Docker provides isolation and reproducibility

### Output (When Implemented)
- Monitoring dashboard showing data/model health metrics
- Automated alerts via email/Slack for anomalies
- Retraining pipeline logs and version history
- Audit trail of all predictions for compliance

## Complete End-to-End Example

1. **Data Preparation**:
   - Dataset organized in `data/kneeKL224/` with train/val/test splits
   - DataLoader applies transforms and handles class imbalance

2. **Model Training**:
   - Run `python -m src.train` 
   - Model trains for 1 epoch on 2% data (fast mode)
   - Best model saved to `models/best_knee_model.pth`
   - Metrics logged to MLflow

3. **Deployment**:
   - Build Docker image: `docker compose -f docker/docker-compose.yml build`
   - Start services: `docker compose -f docker/docker-compose.yml up`
   - API available at http://localhost:8000
   - Web UI served separately or via reverse proxy

4. **Usage**:
   - User uploads knee X-ray via web UI
   - API processes image and returns:
     * Prediction: KL grade (0-4)
     * Confidence: percentage score
     * Heatmap: visual explanation highlighting relevant regions
   - Results displayed in web interface

5. **Maintenance** (Future):
   - Monitoring detects data drift from new hospital data
   - Retraining pipeline triggered automatically
   - New model version registered in MLflow
   - Canary deployment compares new vs old model
   - Promotion to production if performance improved

## Configuration Points

- **Fast vs Full Training**: Adjust `fraction` in `configs/config.yml`
  - 0.02 = 2% of data (fast testing)
  - 1.0 = 100% of data (full training)
  
- **Device Selection**: Set `train.device` to "auto", "cuda", or "cpu"
  
- **Model Architecture**: Change `model.name` to other torchvision models
  
- **Training Parameters**: Modify batch_size, epochs, learning_rate as needed

This workflow provides a complete MLOps pipeline that can be extended with advanced monitoring, automated retraining, and production-grade deployment practices.