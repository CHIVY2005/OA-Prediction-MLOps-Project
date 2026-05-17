---
title: Knee Osteoarthritis Prediction
emoji: 🦴
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
---

# How to Test as a User (Start-to-End)

## 1. Setup Environment
Ensure you have Python installed and dependencies loaded:
```bash
pip install -r requirements.txt
```

## 2. Train the Model (Optional/Verify Pipeline)
To train a new model quickly (Fast Mode):
1.  Check `configs/config.yml`: ensure `fraction: 0.02` (trains on 2% data for speed).
2.  Run:
    ```bash
    python -m src.train
    ```
    *Result: This will output training metrics and save `best_knee_model.pth` to `models/`.*

## 3. Run the Backend API
Start the AI Server which serves predictions:
```bash
python -m uvicorn api.main:app --reload
```
*Wait until you see "Application startup complete".*

## 4. Run the User Interface (New Web App)
Since this is a lightweight web app, you can serve it with Python:
1.  Open a **new** terminal (keep the API running).
2.  Run:
    ```bash
    python -m http.server 3000
    ```
3.  Open your browser to: [http://localhost:3000/web_ui/](http://localhost:3000/web_ui/)

## 5. Test the Diagnosis
1.  Drag & Drop a knee X-ray image into the box.
2.  Wait for the AI to analyze (Grad-CAM heatmap will appear).
3.  Check the "Diagnosis Grade" and "Confidence".

---
*(Legacy) To run the old Streamlit app:*
```bash
streamlit run ui/app.py
```

# Dual-Remote Deployment (GitHub + Hugging Face)

This project uses a dual-remote setup where you can push to both GitHub (for CI/CD testing and Docker registry) and Hugging Face Spaces (for live demo hosting).

## Secrets Configuration

### 1. GitHub Secrets
In your GitHub Repository, navigate to **Settings > Secrets and variables > Actions** and add:
- `CI_CD_WEBHOOK`: Discord Webhook URL for the `#ci-cd_alerts` channel. This ensures pipeline statuses are reported.

### 2. Hugging Face Space Secrets
In your Hugging Face Space, navigate to **Settings > Variables and secrets** and add:
- `api_alerts_webhook`: Discord Webhook URL for the `#api-alerts` channel.
- `ai_prediction_webhook`: Discord Webhook URL for the `#ai-prediction` channel.

## How to Push
Add the Hugging Face Space as a secondary remote on your local machine:
```bash
git remote set-url --add --push origin https://github.com/YOUR_USERNAME/OA-Prediction-MLOps-Project.git
git remote set-url --add --push origin https://huggingface.co/spaces/YOUR_USERNAME/YOUR_SPACE_NAME
```

Once configured, any `git push origin main` will push to **both** GitHub and Hugging Face simultaneously!

# Docker
docker compose -f docker/docker-compose.yml up --build
# ATTENTION

1. Data folder ~ 202mb , so i'll give you the link:
    - https://data.mendeley.com/datasets/56rmx5bjcr/1

2. In this project, i just use kneeKL224.
