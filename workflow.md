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
   
   *Note: Due to size constraints, only `data/kneeKL224/test` is tracked via DVC. The full dataset should be downloaded separately for training.*
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
   - Root `Dockerfile`:
     * Base image: python:3.9-slim
     * Installs system dependencies and python dependencies
     * Copies source code
     * Exposes port 7860 for Hugging Face Spaces
     * Runs `uvicorn deployment.api.main:app --host 0.0.0.0 --port 7860`
   - `docker-compose.yml`:
     * Defines api service
     * Maps port 7860:7860
     * Optional volume mounts for development

3. **CI/CD Pipeline** (`.github/workflows/`):
   - `ci.yml`: Automated testing and linting on pull requests.
   - `cd.yml`: Automated Docker image building and pushing to GitHub Container Registry (GHCR).
   - Pipelines report status directly to Discord via rich, styled cards (Passed/Failed) on the `#ci-cd_alerts` channel.
   - **Deployment to Hugging Face** is handled manually via a **Dual-Remote Git push** configuration (pushing to both GitHub and Hugging Face simultaneously from the local machine), taking advantage of the root `Dockerfile` and native `7860` port mapping.

3. **Web Interface** (`deployment/web_ui/`):
   - Simple HTML interface with drag-and-drop upload
   - JavaScript to:
     * Send image to `/predict` endpoint
     * Display prediction result and confidence
     * Show Grad-CAM heatmap overlay
   - Served by the same FastAPI app at `/` on Hugging Face Spaces, so the browser calls the same-origin `/predict` endpoint

### Output
- Production Space page at https://huggingface.co/spaces/bindeptrai/OA-PREDICTON-MLOps
- Interactive Web UI served by FastAPI at the Space app root
- API documentation available at `/docs` on the Space app URL

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
- **Discord Webhook System**:
  * **🚨 Red Alert (#api-alerts)**: Triggers on API 500 errors, rate limit abuse (429), file sizes exceeding 5MB, or model startup failures. Acts as the "emergency room" for the project.
  * **📊 Gold Tier (#ai-predictions)**: Sends successful predictions (Status 200) including the uploaded image, predicted class, confidence percentage, and processing time. Enables real-time visual monitoring for anomalies and data drift.
- **CI/CD Notifications**: Automated alerts for GitHub Actions pipeline statuses.
- Basic error handling in API (returns prediction even if heatmap fails)
- Startup logging shows model loading status
- MLflow tracks training experiments
- Docker provides isolation and reproducibility

### Output (When Implemented)
- Advanced monitoring dashboard showing data/model health metrics
- Automated retraining pipeline triggered when confidence scores drop
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
   - Initialize environment: `pip install -r requirements.txt`
   - Fetch DVC data: `dvc pull`
   - Build Docker image: `docker compose -f deployment/docker/docker-compose.yml build`
   - Start services: `docker compose -f deployment/docker/docker-compose.yml up`
   - Local API/Web UI available at http://localhost:7860
   - Production API/Web UI available through Hugging Face Spaces

4. **Usage**:
   - User opens the Hugging Face Space and uploads a knee X-ray via Web UI
   - API processes image and returns:
     * Prediction: KL grade (0-4)
     * Confidence: percentage score
     * Heatmap: visual explanation highlighting relevant regions
   - Results displayed in web interface
   - Successful predictions are sent to Discord through `ai_prediction_webhook` with the image/heatmap, class, confidence, and latency

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
