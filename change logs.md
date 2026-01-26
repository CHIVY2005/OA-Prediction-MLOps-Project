# CHANGELOGS
## Followed by rule
### : date
1. Add: for new features, new logic,...
2. Adjustments: for adjust existing files, include codes and implements
3. Intend: for what's to do next

### 1-7-2026

1. Add: 

- Construct of a popular MLOps project's folder`
- Logic of changelogs
- ci/cd .yml
- CONTRIBUTING: Commit principles and working process.
- Create config.yaml
- Create 01_data_exlporation.ipynb, statictis class, image example, metadata, check imbalance, agumentation, build base model.

2. Adjustment:

- Add .gitignore to keep only the folder, ignore entire 

3. Intend:

- In notebook, check data leakage, verify auto_test, validate class imbalance
-  Pipline: dataloader, torchvision, model,py.


### 1-12-2026

1. Add:

- None

2. Adjustment:

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

2. Adjustment:

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

2. Adjustment:
- **README.md**: Added "How to Test as a User" guide.
- **Config**: Verified fast-training settings in `config.yml`.

3. Intend:
- **Containerization**: Use **Docker** to package the API and Web UI into containers for consistent deployment across any machine.
- **Data Versioning**: Implement **DVC (Data Version Control)** to manage large datasets and track changes in data over time.
- **CI/CD Pipeline**: Set up **GitHub Actions** to automatically run tests (`auto_test`) and check code quality whenever changes are pushed.
- **Cloud Deployment**: Deploy the Docker containers to a cloud server (AWS EC2 / GCP) so the app can be accessed globally.
- **Model Monitoring**: Integrate Prometheus/Grafana to monitor model performance and API latency in production.

