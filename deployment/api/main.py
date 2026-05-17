# api/main.py
import sys
import os
import io
import base64
import cv2
import torch
import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile, Request, HTTPException
from fastapi.responses import JSONResponse
import time
import json
import logging
from datetime import datetime
from collections import defaultdict
import asyncio
import requests

# --- SETUP ĐƯỜNG DẪN ---
# Thêm thư mục gốc dự án vào sys.path để import được src
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.config_loader import CFG
from src.model import build_model
from src.data_loader import get_transforms
from src.utils import get_heatmap

# --- CẤU HÌNH LOGGING ---
# Cấu hình logger để xuất ra JSON
logger = logging.getLogger("oa_prediction")
logger.setLevel(logging.INFO)

# Handler để xuất ra console dưới dạng JSON
class JsonFormatter(logging.Formatter):
    def format(self, record):
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno
        }
        # Thêm extra fields nếu có
        if hasattr(record, "extra"):
            log_entry.update(record.extra)
        return json.dumps(log_entry, ensure_ascii=False)

handler = logging.StreamHandler()
handler.setFormatter(JsonFormatter())
logger.addHandler(handler)

# --- CẤU HÌNH ---
MAX_FILE_SIZE = 5 * 1024 * 1024  # 5 MB
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}
DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL", "https://discord.com/api/webhooks/your-webhook-url-here")

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

# --- RATE LIMITING (GIẢN ĐẢN) ---
# Lưu trữ count request theo IP và thời gian
request_counts = defaultdict(list)
RATE_LIMIT_REQUESTS = 10  # 10 requests
RATE_LIMIT_WINDOW = 60    # per 60 seconds

@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    client_ip = request.client.host
    now = time.time()
    
    # Xóa các request cũ ngoài cửa sổ thời gian
    request_counts[client_ip] = [t for t in request_counts[client_ip] if now - t < RATE_LIMIT_WINDOW]
    
    # Kiểm tra nếu vượt quá giới hạn
    if len(request_counts[client_ip]) >= RATE_LIMIT_REQUESTS:
        logger.warning("Rate limit exceeded", extra={
            "client_ip": client_ip,
            "path": request.url.path,
            "method": request.method
        })
        raise HTTPException(status_code=429, detail="Too Many Requests")
    
    # Thêm request hiện tại
    request_counts[client_ip].append(now)
    
    response = await call_next(request)
    return response

@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    
    # Xử lý request
    try:
        response = await call_next(request)
        status_code = response.status_code
    except Exception as e:
        status_code = 500
        raise e
    finally:
        process_time = time.time() - start_time
        # Ghi log sau khi response được trả về (hoặc exception)
        logger.info("Request processed", extra={
            "method": request.method,
            "url": str(request.url),
            "status_code": status_code,
            "process_time_ms": round(process_time * 1000, 2),
            "client_ip": request.client.host if request.client else None
        })
    
    return response

# --- KHỞI TẠO ỨNG DỤNG ---
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
    # Input Validation
    # Check file extension
    file_extension = os.path.splitext(file.filename)[1].lower()
    if file_extension not in ALLOWED_EXTENSIONS:
        logger.warning("Invalid file type", extra={
            "filename": file.filename,
            "extension": file_extension,
            "allowed": list(ALLOWED_EXTENSIONS)
        })
        raise HTTPException(status_code=400, detail=f"Invalid file type. Allowed: {list(ALLOWED_EXTENSIONS)}")
    
    # Check file size
    contents = await file.read()
    if len(contents) > MAX_FILE_SIZE:
        logger.warning("File too large", extra={
            "filename": file.filename,
            "size_bytes": len(contents),
            "max_size_bytes": MAX_FILE_SIZE
        })
        # Send Discord alert for oversized file
        if DISCORD_WEBHOOK_URL and "your-webhook-url-here" not in DISCORD_WEBHOOK_URL:
            try:
                requests.post(DISCORD_WEBHOOK_URL, json={
                    "content": f"⚠️ **OVERSIZED FILE BLOCKED**\n"
                              f"Filename: `{file.filename}`\n"
                              f"Size: {len(contents) / 1024 / 1024:.2f} MB\n"
                              f"Limit: {MAX_FILE_SIZE / 1024 / 1024:.2f} MB"
                })
            except:
                pass  # Don't let alerting break the main flow
        raise HTTPException(status_code=413, detail="File too large")
    
    # 1. Doc anh
    try:
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        logger.warning("Invalid image file", extra={
            "filename": file.filename,
            "error": str(e)
        })
        raise HTTPException(status_code=400, detail="Invalid image file")
    
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
        logger.warning("Heatmap generation failed", extra={
            "filename": file.filename,
            "error": str(e)
        })
        # Nếu lỗi, heatmap_base64 vẫn là chuỗi rỗng "", API vẫn trả về kết quả dự đoán chứ không sập (500)
    
    logger.info("Prediction made", extra={
        "filename": file.filename,
        "prediction": class_name,
        "confidence": confidence,
        "has_heatmap": bool(heatmap_base64)
    })
    
    return {
        "filename": file.filename,
        "prediction": class_name,
        "confidence": f"{confidence:.2%}",
        "heatmap_base64": heatmap_base64
    }

@app.get("/")
def home():
    return {"message": "API is running!"}

@app.get("/health")
def health_check():
    global model
    if model is None:
        return JSONResponse(
            status_code=503,
            content={"status": "unhealthy", "detail": "Model not loaded"}
        )
    return {
        "status": "healthy", 
        "model_loaded": model is not None,
        "device": str(device),
        "timestamp": datetime.utcnow().isoformat()
    }

# Exception handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail}
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    logger.error("Unhandled exception", extra={
        "path": request.url.path,
        "method": request.method,
        "error": str(exc),
        "exc_info": True
    })
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"}
    )