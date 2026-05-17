# MLOps Journey: From Data to Deployment

This document outlines the complete MLOps lifecycle of the Knee Osteoarthritis Prediction project. It explains how we built, trained, and deployed the model, following best practices in machine learning operations.

## 1. Project Inception
**Goal**: Automate the grading of Knee Osteoarthritis (OA) from X-ray images using Deep Learning.
**Problem**: Manual diagnosis is time-consuming and subjective.
**Solution**: An AI model (EfficientNet-B0) that classifies X-rays into 5 grades (0-4), served via a web interface.

## 2. Data Pipeline
The data handling is managed by `src/data_loader.py`.
- **Source**: Medical X-ray datasets (e.g., Mendeley Data).
- **Preprocessing**:
    - **Resize**: All images are resized to 224x224 (Input size for EfficientNet).
    - **Augmentation**: To prevent overfitting, we apply transformations like rotation and flipping during training.
    - **Loading**: We use PyTorch `DataLoader` for efficient batched loading.
- **Handling Imbalance**: The code supports weighting classes in the Loss function (`CrossEntropyLoss`) to handle class imbalance.

## 3. Model Development & Training
The core training logic resides in `src/train.py` and `src/model.py`.
- **Architecture**: We use **EfficientNet-B0** (pretrained on ImageNet) for its balance of speed and accuracy.
- **Configuration**: All hyperparameters (learning rate, batch size, epochs) are decoupled from code and stored in `configs/config.yml`. This allows us to change experiments without touching the code.
- **Experiment Tracking (MLflow)**:
    - We use MLflow to track every training run.
    - Metrics monitored: `train_loss`, `val_loss`, `val_accuracy`.
    - Artifacts: The best model is automatically saved (`models/best_knee_model.pth`) and logged.

## 4. Evaluation & Verification
Before deployment, the model undergoes verification:
- **Fast Testing**: We use a `fraction` parameter in `config.yml` to run quick sanity checks (train on 2% data) to ensure the pipeline works.
- **Explainability (Grad-CAM)**: To trust the AI, we implemented Grad-CAM in `src/utils.py`. This highlights *where* the model is looking (e.g., bone edges, joint space) to make its decision.

## 5. Deployment Information
We have two interfaces for the user:
1.  **Modern Web UI (`web_ui/`)**: A fast, premium interface built with HTML/CSS/JS.
2.  **Legacy App (`ui/app.py`)**: A Streamlit prototype for quick internal demoing.

The backend (`api/main.py`) using **FastAPI** serves the model to both interfaces, ensuring consistent logic.

**Containerization (Docker)**:
We have containerized the application to ensure consistency across environments.
- **API Container**: Runs the FastAPI backend.
- **UI Container**: Runs the Streamlit app.
- **Orchestration**: `docker-compose` manages both services, handling networking and startup dependency.

## 6. Continuous Improvement & Monitoring (CI/CD)
- **Versioning**: Code is versioned with Git.
- **Reproducibility**: `requirements.txt` ensures the environment is consistent.
- **Containerization**: Fully containerized API using Docker and Docker Compose.
- **CI/CD Pipeline**: GitHub Actions are set up for Continuous Integration (testing with `pytest`, linting with `flake8`) and Continuous Deployment (building and pushing Docker images to GHCR). Pipeline statuses are automatically reported to Discord.
- **Real-time Monitoring (Discord Webhooks)**:
    - **🚨 Red Alert (#api-alerts)**: Instantly notifies the team of critical failures (500 errors, rate limit abuses, oversized files, or model startup failures).
    - **📊 Gold Tier (#ai-predictions)**: A live feed of all successful predictions with images, results, confidence scores, and processing times. This allows visual monitoring of data drift and model accuracy in production.
- **Future Steps**:
    - **Data Versioning**: Implement **DVC (Data Version Control)** to manage large datasets and track changes in data over time.
    - **Cloud Deployment**: Deploy the Docker containers to a cloud server (AWS EC2 / GCP) so the app can be accessed globally.
    - **Advanced Monitoring**: Integrate Prometheus/Grafana to monitor system metrics.
