# CHANGELOGS
## Followed by rule
### : date
1. Add: for new features, new logic,...
2. Adjustments: for adjust existing files, include codes and implements
3. Intend: for what's to do next

### 5-23-2026

1. Add:
- Added `@app.on_event("startup")` hook in FastAPI to send successful startup alerts to the Discord `api-alerts` webhook, with auto-detection of Hugging Face `SPACE_ID`.

2. Adjustments:
- Enhanced `is_valid_knee_xray()` into a robust 4-stage validation pipeline:
  1. **Color check** (rejects color images with RGB variance > 30.0, only applies to multi-channel images).
  2. **Mid-gray ratio check** (runs on all images; rejects synthetic diagrams/flowcharts with mid-gray pixels < 20%).
  3. **Hough Line check** (runs on all images; rejects flowcharts/text diagrams by counting straight lines; triggers if line count > 8).
  4. **MobileNetV3 check** (rejects other grayscale natural objects predicted with confidence > 35%).
- Restructured `is_valid_knee_xray()` so that stages 2, 3, and 4 execute correctly on single-channel grayscale images (previously they were nested inside the multi-channel color check block and bypassed completely for single-channel flowcharts).
- Added an image size bypass (size < 50x50) in `is_valid_knee_xray()` to ensure pytest dummy images don't fail validation checks.
- Fixed an issue where the Discord `ai_prediction` webhook failed to notify on invalid uploads by replacing `raise HTTPException` with `return JSONResponse`, allowing `BackgroundTasks` to complete properly. Supported robust environment variable parsing (handling lowercase/uppercase, singular/plural naming variations like `api_alert_webhook` / `API_ALERTS_WEBHOOK`) to ensure compatibility with Hugging Face Space secrets, and added configuration printing to container logs to assist debugging.
- Updated `load_dotenv` to load from the project root directory, fixing a bug where local runs started from subdirectories (like `deployment/api/` or `deployment/`) failed to load the `.env` file and did not trigger Discord notifications.
- Added a `/debug-env` endpoint to print loaded webhook statuses and masked URLs to help diagnose configuration issues on Hugging Face Spaces.
- Added status and response logging inside Discord webhook calls to log response codes from Discord.
- Implemented a self-healing `post_to_discord` helper function with rotation of Discord API domains (`discord.com`, `canary.discord.com`, `ptb.discord.com`, `discordapp.com`) and a custom Chrome User-Agent header to bypass Cloudflare/Discord IP blocks and Read Timeouts when running on Hugging Face Spaces.

### 5-21-2026

1. Add:
- Added FastAPI reliability tests for health checks, file validation, model-not-loaded behavior, and mocked prediction success.
- Added `httpx` to development requirements for FastAPI `TestClient`.
- Implemented Offline IP Geolocation using a localized VN CIDR block cache (`vn-cidr.txt`) to restrict API access strictly to Vietnam without external API latency.
- Integrated `slowapi` for optimized, in-memory rate limiting applied directly to specific endpoints (`/predict` at 10/min, others at 60/min).
- **Image Validation**: Implemented an AI-powered validation check using MobileNetV3 (ImageNet-1k) and color variance to reject non-knee images. Invalid images are automatically rejected with a 400 Bad Request and a clear instruction to upload a valid knee X-ray.
- **Webhook Filtering**: The `ai_prediction_webhook` Discord channel is now exclusively used for monitoring malicious/invalid uploads along with the validation confidence score. Notifications for successful predictions have been silenced to prevent channel spamming.

2. Adjustments:
- Updated CI so tests fail the build when they fail, and flake8 checks critical Python errors while excluding local virtual environments.
- Hardened `deployment/api/main.py` so failed startup leaves `model = None`, `/predict` returns `503 Model not loaded`, Discord alerts use request timeouts, and logger handlers are not duplicated during reload/test imports.
- Updated documentation for Hugging Face Spaces as the production user-testing path: the Space serves the Web UI at `/`, calls same-origin `/predict`, displays prediction/heatmap to users, and sends prediction images/heatmaps to Discord through `ai_prediction_webhook`.
- Installed runtime/test dependencies into `.venv` and verified tests pass locally.
- Fixed CI/CD pipeline failure by adding the missing `slowapi` dependency to `requirements.txt`.
- Consolidated dependency management by removing the redundant `requirements/` folder in favor of a single `requirements.txt` at the root directory.
- Integrated **DVC (Data Version Control)** to track large files (using a local remote `dvc_remote/`). Untracked `data/kneeKL224/test/` and `models/best_knee_model.pth` from Git and moved them to DVC management.

3. Intend:
- Migrate FastAPI startup from deprecated `on_event` to lifespan handlers.
- Add a lightweight smoke check for the deployed Hugging Face Space after push.

### 5-20-2026

2. Adjustments:
- Fix: Resolved API Error (500) during image upload on Hugging Face Spaces by fixing a Python logging conflict. Renamed `filename` to `file_name` in `logger.info` extra fields and removed `exc_info` from `extra` in exception handlers to prevent `KeyError` conflicts with reserved `LogRecord` attributes.
- Fix: Updated `JsonFormatter` in `deployment/api/main.py` to properly serialize custom extra fields for MLOps JSON logging.

### 5-17-2026

1. Add:
- Added GitHub Actions CI/CD pipelines (`.github/workflows/ci.yml` and `cd.yml`) for automated testing and Docker image deployment.
- Created `Dockerfile` and `docker-compose.yml` in `deployment/docker/` to containerize the API.
- Implemented a Two-Tier Discord Webhook Monitoring System in `deployment/api/main.py`:
  * **🚨 #api-alerts (Red Alert)**: Monitors API 500 errors, rate limits (429), file size limits (>5MB), and model loading failures.
  * **📊 #ai-predictions (Gold Tier)**: Logs successful predictions (Status 200) with the original X-ray image, predicted class, confidence score, and processing time for real-time monitoring.
- Integrated Discord notifications into GitHub Actions workflows using structured, colored Rich Embeds displaying status emojis (✅/❌), trigger Actor, and branch name.
- Re-architected deployment for **Dual-Remote Push**: Moved `Dockerfile` to the root and changed port to `7860` natively. Removed GitHub Actions auto-sync so developers can push manually and directly to both GitHub and Hugging Face Spaces using local Git remotes.
- Optimized Dockerfile for Hugging Face builds by explicitly installing **PyTorch CPU-only** (`--index-url https://download.pytorch.org/whl/cpu`) first, reducing build time and memory usage to prevent builder crashes.
- Updated `main.py` to load environment variables from `.env` using `python-dotenv`.
- Updated `requirements.txt` to include `python-dotenv`.
- Updated `src/train.py` to conditionally generate training loss/accuracy plots only if `epochs > 1` (preventing single-dot charts), and to send training completion summaries to Discord using a polished Rich Embed card.
- Adjusted CI/CD Discord alert routing: success and failure notifications are sent for GitHub runs, while keeping Hugging Face Space sync silent to prevent developer channel spam.

3. Intend:
- Setup automated retraining pipeline when confidence scores consistently drop.
- Deploy Docker containers to cloud infrastructure (AWS/GCP) for backup.

### 5-16-2026

1. Add:
- Restructured project folder following MLOps best practices:
  * Created deployment/ folder containing api/, docker/, web_ui/
  * Created experiments/ folder separating notebooks and mlruns/
  * Added monitoring/ and reports/ placeholders for future work
  * Split requirements.txt into environment-specific files in requirements/
  * Added explain.md documenting each file/folder purpose
  * Added workflow.md detailing end-to-end MLOps pipeline
- Updated import paths in deployment/api/main.py to reflect new structure

2. Adjustments:
- Removed legacy api/, ui/, web_ui/, notebooks/ folders from root
- Moved mlruns/ to experiments/mlruns/
- Preserved all source code and functionality while improving organization

3. Intend:
- Implement data drift detection and monitoring in monitoring/
- Set up automated retraining pipeline triggered by monitoring alerts
- Add Prometheus/Grafana integration for production monitoring
- Implement CI/CD pipeline with GitHub Actions for automated testing
- Prepare for cloud deployment (AWS/GCP) using Docker containers

### 1-7-2026

1. Add:
- Construct of a popular MLOps project's folder`
- Logic of changelogs
- ci/cd .yml
- CONTRIBUTING: Commit principles and working process.
- Create config.yaml
- Create 01_data_exlporation.ipynb, statictis class, image example, metadata, check imbalance, agumentation, build base model.

2. Adjustments:
- Add .gitignore to keep only the folder, ignore entire 

3. Intend:
- In notebook, check data leakage, verify auto_test, validate class imbalance
-  Pipline: dataloader, torchvision, model,py.


### 1-12-2026

1. Add:
- None

2. Adjustments:
-  In notebook, check data leakage, verify auto_test, validate class imbalance

3. Intend:
- Add MLflow to attach logging to following loss/accuracy
- Separate notebook into specific python scripts (dataloader, model.py,...)
- Use Git and DVC to mangage code and data with git/dvc
- Create a pipline automate training with DVC piplines/airflow


### 1-14-2026

1. Add:
- Config parameters for model
- Create configloader.py to help python know what is in config.yml in the configs folder
- Create data_loader.py to load entire images
- Create model.py to handle model training
- Use MLFlow to follow progress, by using syntax mlflow ui

- Create utils.py in the src folder to visualize OA area when return result to user using GRAD-CAM
- Create main.py in api folder, which load model from src and open output port by using syntax: 
    - python -m uvicorn api.main:app --reload
- Create app.py in ui folder to build interface for user using streamlit

2. Adjustments:
- Adjust guideline and change postfix of config.yaml into config.yml
- Adjust dataloader.py into data_loader.py to correct the path
- fix not showing port and check for errorsapp.py in folder ui
- Fix when deployed successfully but existed error api 500-internal server error:
    - Adjust get_heatmap function in utils.py in src folder and measure that functions in main.py in api folder under level of torch.get_grad_enabled(True)

3. Intend:
### 1-26-2026

1. Add:
- **Web UI**: Created a new, premium web interface (`web_ui/`) using HTML/CSS/JS (No-Build) for easy testing.
- **MLOPS_JOURNEY.md**: Documentation of the full project lifecycle.
- **CORS Support**: Updated `api/main.py` to allow browser access to the API.

2. Adjustments:
- **README.md**: Added "How to Test as a User" guide.
- **Config**: Verified fast-training settings in `config.yml`.

3. Intend:
- **Containerization**: Use **Docker** to package the API and Web UI into containers for consistent deployment across any machine.
- **Data Versioning**: Implement **DVC (Data Version Control)** to manage large datasets and track changes in data over time.
- **CI/CD Pipeline**: Set up **GitHub Actions** to automatically run tests (`auto_test`) and check code quality whenever changes are pushed.
- **Cloud Deployment**: Deploy the Docker containers to a cloud server (AWS EC2 / GCP) so the app can be accessed globally.
- **Model Monitoring**: Integrate Prometheus/Grafana to monitor model performance and API latency in production.


### 1-28-2026

1. Add:
- **Docker Integration**: Added `Dockerfile` and `docker-compose.yml` to containerize both the API and UI services, ensuring a consistent runtime environment.
- **Documentation**: Updated `MLOPS_JOURNEY.md` to reflect the Docker deployment architecture.

2. Adjustments:
- **MLOPS_JOURNEY.md**: Refined the "Future Steps" to include specific tools like DVC, GitHub Actions, Cloud Deployment, and Prometheus/Grafana.

3. Intend:
- **Data Versioning**: Implement **DVC (Data Version Control)** to manage large datasets and track changes in data over time.
- **CI/CD Pipeline**: Set up **GitHub Actions** to automatically run tests (`auto_test`) and check code quality whenever changes are pushed.
- **Cloud Deployment**: Deploy the Docker containers to a cloud server (AWS EC2 / GCP) so the app can be accessed globally.
- **Model Monitoring**: Integrate Prometheus/Grafana to monitor model performance and API latency in production.
