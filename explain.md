# Explaination of Files and Folders

This document provides an overview of the purpose and functionality of each file and folder in the restructured OA-Prediction-MLOps project.

## Root Level

- **.github/**: Contains GitHub-specific configurations (e.g., issue templates, workflows for CI/CD).
  - **workflows/**: CI/CD pipelines.
    - `ci.yml`: Automated testing and linting.
    - `cd.yml`: Automated Docker building and push, as well as **automated sync and deployment to Hugging Face Spaces**. Pipelines send rich, colored Discord alerts (Passed/Failed) to `#ci-cd_alerts`.
- **configs/**: Configuration files for the project.
  - `config.yml`: Main configuration file defining data, model, and training parameters.
- **data/**: Stores the dataset used for training and evaluation.
  - `kneeKL224/`: The knee osteoarthritis dataset split into train, validation, and test sets.
- **deployment/**: Contains all files related to deploying the model as a service.
  - **api/**: FastAPI application for serving predictions.
    - `main.py`: Entry point for the API; handles model loading, prediction endpoints, and heatmap generation.
  - **docker/**: Docker-related files for containerization.
    - `docker-compose.yml`: Defines services for running the application with Docker Compose.
    - `Dockerfile`: Instructions for building the Docker image.
  - **web_ui/**: Web interface for interacting with the model.
    - `index.html`: Main HTML file for the web UI.
    - `assets/`: Static assets (CSS, JavaScript, images) for the web UI.
- **experiments/**: For research and experimentation.
  - **notebooks/**: Jupyter notebooks for data exploration, model prototyping, etc.
    - `01_data_exlporation.ipynb`: Initial data exploration and analysis.
  - **mlruns/**: MLflow tracking data (automatically generated during training).
- **models/**: Stores trained model weights.
  - `best_knee_model.pth`: The best model checkpoint saved during training.
- **pipelines/**: Scripts for orchestrating workflows (e.g., training, evaluation, deployment).
- **src/**: Core source code of the project (a reusable library).
  - **config_loader.py**: Loads and processes the configuration from `configs/config.yml`.
  - **data_loader.py**: Defines the dataset class and functions for loading and preparing data.
  - **model.py**: Contains functions to build and configure the model architecture.
  - **train.py**: Script to train the model.
  - **utils.py**: Utility functions, including Grad-CAM for heatmap generation.
- **tests/**: Unit and integration tests.
- **monitoring/**: For setting up monitoring, logging, and alerting (to be implemented).
- **reports/**: For storing evaluation results, metrics, and visualizations (to be implemented).
- **requirements/**: Separated requirement files for different environments.
  - `base.txt`: Core dependencies.
  - `dev.txt`: Development dependencies (e.g., testing, linting).
  - `prod.txt`: Production dependencies.

## Key Files

- `README.md`: Overview of the project and instructions for getting started.
- `MLOPS_JOURNEY.md`: Document detailing the MLOps journey and lessons learned.
- `change logs.md`: History of changes made to the project.
- `LICENSE`: License information.
- `.gitignore`: Specifies files and directories to be ignored by Git.
- `.env`: (Not committed to Git) Contains sensitive environment variables like `api_alerts_webhook` and `ai_prediction_webhook` for Discord monitoring.

## Notes

- The `src/` directory is designed to be importable as a module. The `deployment/api/main.py` adjusts the system path to import from `src`.
- The `experiments/` directory is isolated from the production code to keep the main repository clean.
- The `monitoring/` and `reports/` directories are placeholders for future work on model monitoring and reporting.
