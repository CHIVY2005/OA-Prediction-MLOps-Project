---
title: Knee Osteoarthritis Prediction
emoji: 🦴
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
---

# Knee Osteoarthritis Prediction

FastAPI + EfficientNet-B0 app for classifying knee X-ray osteoarthritis grade and showing a Grad-CAM heatmap. The production demo is served as a Docker Hugging Face Space.

## Test On Hugging Face Space

Open the live Space:

- https://huggingface.co/spaces/bindeptrai/OA-PREDICTON-MLOps
- App URL: https://bindeptrai-oa-predicton-mlops.hf.space

User flow:
1. Open the Space app.
2. Drag and drop a JPEG/PNG knee X-ray image.
3. Wait for the API to return diagnosis grade, confidence, and Grad-CAM heatmap.
4. The app also sends the prediction image/heatmap, result, confidence, and latency to Discord through `ai_prediction_webhook`.

The Web UI is served by the same FastAPI app at `/`, and browser uploads call the same-origin `/predict` endpoint. Do not serve `deployment/web_ui` separately for production because the JavaScript expects the backend on the same origin.

## Hugging Face Configuration

Set these Space secrets in **Settings > Variables and secrets**:

- `api_alerts_webhook`: Discord webhook for API alerts such as startup failure, rate limits, oversized files, and 500 errors.
- `ai_prediction_webhook`: Discord webhook for successful prediction monitoring with image/heatmap attachments.

The Space uses the root `Dockerfile`, exposes port `7860`, and starts:

```bash
uvicorn deployment.api.main:app --host 0.0.0.0 --port 7860
```

## Local Development

Create/use a virtual environment and install dependencies:

```bash
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\dvc.exe pull
```

Run the API and Web UI locally:

```bash
.\.venv\Scripts\python.exe -m uvicorn deployment.api.main:app --reload
```

Open:

```text
http://localhost:8000/
```

Run tests:

```bash
.\.venv\Scripts\python.exe -m pytest -q
```

## Training

To train a quick verification model:

1. Check `configs/config.yml` and use `fraction: 0.02` for fast mode.
2. Run:

```bash
.\.venv\Scripts\python.exe -m src.train
```

The best checkpoint is saved to `models/best_knee_model.pth`.

## Deployment And Push

This repo uses dual push remotes:

- GitHub: source control and CI/CD checks.
- Hugging Face Space: live Docker app hosting.

Current push setup:

```bash
git remote set-url --add --push origin https://github.com/CHIVY2005/OA-Prediction-MLOps-Project.git
git remote set-url --add --push origin https://huggingface.co/spaces/bindeptrai/OA-PREDICTON-MLOps
```

Then:

```bash
git push origin main
```

This pushes the same commit to both GitHub and Hugging Face Space.

## Docker

For local Docker testing:

```bash
docker compose -f deployment/docker/docker-compose.yml up --build
```

## Data

The full dataset is large. DVC (Data Version Control) is used to track the `data/kneeKL224/test/` folder and model weights (`models/best_knee_model.pth`) to keep the Git repository lightweight.
For training on the full dataset, download KneeKL224 from:

- https://data.mendeley.com/datasets/56rmx5bjcr/1
