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

# Docker
docker compose -f docker/docker-compose.yml up --build
# ATTENTION

1. Data folder ~ 202mb , so i'll give you the link:
    - https://data.mendeley.com/datasets/56rmx5bjcr/1

2. In this project, i just use kneeKL224.
