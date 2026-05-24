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
We have two interfaces:
1.  **Production Web UI (`deployment/web_ui/`)**: A fast HTML/CSS/JS interface served directly by FastAPI on Hugging Face Spaces.
2.  **Legacy Streamlit App (`deployment/web_ui/app.py`)**: A prototype kept for internal demoing.

The backend (`deployment/api/main.py`) using **FastAPI** serves the Web UI at `/`, exposes `/predict`, returns prediction results to users, and sends prediction images/heatmaps to Discord through `ai_prediction_webhook`.

**Containerization (Docker)**:
We have containerized the application to ensure consistency across environments.
- **Hugging Face Space Container**: Runs the FastAPI backend and serves the Web UI from the same origin.
- **Port**: The root Dockerfile exposes `7860`, which is the native Hugging Face Spaces app port.
- **Local orchestration**: `deployment/docker/docker-compose.yml` can run the same app locally for verification.


## 6. Continuous Improvement & Monitoring (CI/CD)
- **Versioning**: Code is versioned with Git.
- **Reproducibility**: `requirements.txt` ensures the environment is consistent.
- **Containerization**: Fully containerized API using Docker and Docker Compose.
- **CI/CD Pipeline**: GitHub Actions are set up for Continuous Integration (testing with `pytest`, linting with `flake8`) and Continuous Deployment (building and pushing Docker images to GHCR). Pipeline statuses (Passed/Failed) are automatically reported to Discord via rich, styled cards showing the actor and branch name. Deployment to Hugging Face Spaces is handled via a local **Dual-Remote Git push** configuration which pushes to both GitHub and Hugging Face simultaneously.
- **Real-time Monitoring (Discord Webhooks)**:
    - **🚨 Red Alert (#api-alerts)**: Instantly notifies the team of critical failures (500 errors, rate limit abuses, oversized files, or model startup failures).
    - **📊 Gold Tier (#ai-predictions)**: A live feed of successful predictions from the Hugging Face Space with the image/heatmap, result, confidence score, and processing time. This allows visual monitoring of data drift and model accuracy in production.
- **API Security & Reliability**:
    - **Offline IP Geolocation**: The API uses a localized cache of Vietnam CIDR blocks with binary search (`bisect`) to strictly limit access to Vietnamese IP addresses in `< 0.1ms` without network latency.
    - **Endpoint Rate Limiting**: Uses `slowapi` to enforce strict in-memory rate limiting per IP (`10 req/min` for predictions, `60 req/min` for health checks) preventing abuse and out-of-memory errors.
- **Data Versioning (DVC)**: Implemented **DVC** to track large files (`data/kneeKL224/test/` and `models/best_knee_model.pth`), keeping the Git repository lightweight and enabling reproducible models.
- **Future Steps**:
    - **Data Drift Detection**: Implement data drift monitoring using tools like Evidently AI to detect when production data distribution changes compared to training data.
    - **Cloud Deployment**: Deploy the Docker containers to a cloud server (AWS EC2 / GCP) so the app can be accessed globally.
    - **Advanced Monitoring**: Integrate Prometheus/Grafana to monitor system metrics.
    - **Persistent Feedback Storage**: Migrate the local ephemeral `data/feedback/` directory to a Persistent Storage solution (e.g., Hugging Face Persistent Storage, AWS S3, or Google Drive API). This guarantees that valuable user/clinician feedback is securely preserved across container restarts and can be automatically aggregated for future model retraining loops.
