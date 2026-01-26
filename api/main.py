# api/main.py
import sys
import os
import io
import base64
import cv2
import torch
import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile

# --- SETUP ĐƯỜNG DẪN ---
# Thêm thư mục gốc dự án vào sys.path để import được src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config_loader import CFG
from src.model import build_model
from src.data_loader import get_transforms
from src.utils import get_heatmap

# --- KHỞI TẠO APP (QUAN TRỌNG: Uvicorn tìm biến này) ---
app = FastAPI(title="Knee Osteoarthritis Detection API")

# --- CORS SETUP (Cho phép Web UI gọi API) ---
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Trong production nên thay bằng domain cụ thể
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Biến global
model = None
device = CFG['train']['device']

@app.on_event("startup")
def load_predictor():
    global model
    try:
        print("[...] Dang load model...")
        model = build_model()
        
        # Load weights
        model_name = CFG['train']['save_name']
        model_path = os.path.join(CFG['paths']['models'], model_name)
        
        # map_location de chay duoc ca tren may khong co GPU
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint)
        
        model.to(device)
        model.eval()
        print(f"[OK] Model loaded from {model_path}")
    except Exception as e:
        print(f"[ERROR] Loi load model: {e}")

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # 1. Doc anh
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    
    # 2. Preprocess
    transform = get_transforms()['val']
    input_tensor = transform(image).unsqueeze(0).to(device)

    # 3. Inference
    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        conf, pred_idx = torch.max(probs, 1)
    
    label_idx = pred_idx.item()
    confidence = conf.item()
    class_name = CFG['data']['class_names'][label_idx]

    # --- SUA DOAN NAY (BAT DAU) ---
    heatmap_base64 = "" # Mac dinh la rong de khong loi neu heatmap fail
    
    try:
        # 4. Heatmap
        # Bat buoc bat grad cho luong nay de GradCAM tinh toan nguoc duoc
        with torch.set_grad_enabled(True):
            # Quan trong: Tensor dau vao phai cho phep tinh gradient
            input_tensor.requires_grad = True 
            
            # Goi ham ve heatmap
            heatmap_img = get_heatmap(model, input_tensor, image)
        
        # Ma hoa anh sang Base64
        _, buffer = cv2.imencode('.jpg', cv2.cvtColor(heatmap_img, cv2.COLOR_RGB2BGR))
        heatmap_base64 = base64.b64encode(buffer).decode('utf-8')
        
    except Exception as e:
        print(f"[WARN] Loi tao heatmap: {e}")
        # Nếu lỗi, heatmap_base64 vẫn là chuỗi rỗng "", API vẫn trả về kết quả dự đoán chứ không sập (500)

    return {
        "filename": file.filename,
        "prediction": class_name,
        "confidence": f"{confidence:.2%}",
        "heatmap_base64": heatmap_base64
    }

@app.get("/")
def home():
    return {"message": "API is running!"}