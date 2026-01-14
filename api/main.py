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

# Biến global
model = None
device = CFG['train']['device']

@app.on_event("startup")
def load_predictor():
    global model
    try:
        print("⏳ Đang load model...")
        model = build_model()
        
        # Load weights
        model_name = CFG['train']['save_name']
        model_path = os.path.join(CFG['paths']['models'], model_name)
        
        # map_location để chạy được cả trên máy không có GPU
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint)
        
        model.to(device)
        model.eval()
        print(f"✅ Model loaded from {model_path}")
    except Exception as e:
        print(f"❌ Lỗi load model: {e}")

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # 1. Đọc ảnh
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

    # --- SỬA ĐOẠN NÀY (BẮT ĐẦU) ---
    heatmap_base64 = "" # Mặc định là rỗng để không lỗi nếu heatmap fail
    
    try:
        # 4. Heatmap
        # Bắt buộc bật grad cho luồng này để GradCAM tính toán ngược được
        with torch.set_grad_enabled(True):
            # Quan trọng: Tensor đầu vào phải cho phép tính gradient
            input_tensor.requires_grad = True 
            
            # Gọi hàm vẽ heatmap
            heatmap_img = get_heatmap(model, input_tensor, image)
        
        # Mã hóa ảnh sang Base64
        _, buffer = cv2.imencode('.jpg', cv2.cvtColor(heatmap_img, cv2.COLOR_RGB2BGR))
        heatmap_base64 = base64.b64encode(buffer).decode('utf-8')
        
    except Exception as e:
        print(f"⚠️ Lỗi tạo heatmap: {e}")
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