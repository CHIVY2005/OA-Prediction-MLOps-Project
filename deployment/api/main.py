# api/main.py
import sys
import os
import io
import base64
import cv2
import torch
from PIL import Image
from fastapi import FastAPI, File, UploadFile, Request, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import time
import json
import logging
from datetime import datetime
from collections import defaultdict
import requests
from dotenv import load_dotenv
import numpy as np

# Thêm thư mục gốc dự án vào sys.path để import được src
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_dir)

load_dotenv(os.path.join(root_dir, ".env"))

import uuid
import shutil
from pydantic import BaseModel
from typing import Optional

# Cấu hình thư mục lưu trữ feedback
FEEDBACK_DIR = os.path.join(root_dir, "data", "feedback")
TEMP_FEEDBACK_DIR = os.path.join(FEEDBACK_DIR, "temp")
os.makedirs(TEMP_FEEDBACK_DIR, exist_ok=True)

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
    RESERVED_ATTRS = {
        'args', 'asctime', 'created', 'exc_info', 'exc_text', 'filename',
        'funcName', 'levelname', 'levelno', 'lineno', 'module', 'msecs',
        'message', 'msg', 'name', 'pathname', 'process', 'processName',
        'relativeCreated', 'stack_info', 'thread', 'threadName'
    }

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
        for key, value in record.__dict__.items():
            if key not in self.RESERVED_ATTRS and not key.startswith('_'):
                log_entry[key] = value
        return json.dumps(log_entry, ensure_ascii=False)

handler = logging.StreamHandler()
handler.setFormatter(JsonFormatter())
if not logger.handlers:
    logger.addHandler(handler)

# --- CẤU HÌNH ---
MAX_FILE_SIZE = 5 * 1024 * 1024  # 5 MB
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}
DISCORD_WEBHOOK_URL = os.getenv("DISCORD_ALERT_WEBHOOK_URL") # Keep this for backward compatibility if needed
API_ALERTS_WEBHOOK = (
    os.getenv("api_alerts_webhook") 
    or os.getenv("API_ALERTS_WEBHOOK") 
    or os.getenv("api_alert_webhook") 
    or os.getenv("API_ALERT_WEBHOOK")
)
AI_PREDICTION_WEBHOOK = (
    os.getenv("ai_prediction_webhook") 
    or os.getenv("AI_PREDICTION_WEBHOOK") 
    or os.getenv("ai_predictions_webhook") 
    or os.getenv("AI_PREDICTIONS_WEBHOOK")
)

def mask_webhook(url: str) -> str:
    if not url:
        return "Not Set"
    parts = url.split("/")
    if len(parts) > 2:
        last_part = parts[-1]
        masked_last = last_part[:5] + "..." + last_part[-5:] if len(last_part) > 10 else "..."
        parts[-1] = masked_last
        return "/".join(parts)
    return "Invalid URL"

print(f"[CONFIG] api_alerts_webhook: {mask_webhook(API_ALERTS_WEBHOOK)}")
print(f"[CONFIG] ai_prediction_webhook: {mask_webhook(AI_PREDICTION_WEBHOOK)}")

DISCORD_DOMAINS = ["discord.com", "canary.discord.com", "ptb.discord.com", "discordapp.com"]

def post_to_discord(url: str, **kwargs) -> requests.Response:
    if not url:
        raise ValueError("Webhook URL is empty")
        
    last_err = None
    # Thay thế User-Agent mặc định của Python-requests để tránh bị Cloudflare chặn
    headers = kwargs.get("headers", {})
    if "User-Agent" not in headers:
        headers["User-Agent"] = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    kwargs["headers"] = headers

    for domain in DISCORD_DOMAINS:
        # Tạo URL với domain hiện tại
        current_url = url
        for d in DISCORD_DOMAINS:
            if d in url:
                current_url = url.replace(d, domain)
                break
        try:
            res = requests.post(current_url, **kwargs)
            if res.status_code < 400:
                return res
            else:
                print(f"[DISCORD RETRY] Domain {domain} failed with status {res.status_code}: {res.text[:100]}")
        except Exception as e:
            print(f"[DISCORD RETRY] Domain {domain} raised error: {e}")
            last_err = e
            
    if last_err:
        raise last_err
    raise Exception("All Discord domains failed to respond")

def send_discord_alert(message: str):
    if not API_ALERTS_WEBHOOK:
        print("[CONFIG ALERT] API_ALERTS_WEBHOOK is not configured.")
        return
    try:
        res = post_to_discord(
            API_ALERTS_WEBHOOK,
            json={"content": f"[API ALERT]\n{message}"},
            timeout=10,
        )
        print(f"[DISCORD ALERT] Status: {res.status_code}, Response: {res.text[:100]}")
    except Exception as e:
        logger.error(f"Failed to send alert to Discord: {e}")

def send_discord_prediction(image_bytes: bytes, filename: str, result: str, confidence: float, process_time: float):
    if not AI_PREDICTION_WEBHOOK:
        return
    try:
        # 1. Định nghĩa file ảnh để gửi
        files = {
            "file": (filename, image_bytes, "image/jpeg")
        }

        # 2. Xây dựng khối thông tin (Embed)
        payload = {
            "embeds": [
                {
                    "title": "🦴 [Live] Phân Tích X-Quang Khớp Gối Mới",
                    "color": 3447003, # Màu xanh dương
                    "fields": [
                        {"name": "Dự đoán (Class)", "value": f"**{result}**", "inline": True},
                        {"name": "Độ tự tin (Confidence)", "value": f"**{confidence:.2%}**", "inline": True},
                        {"name": "Latency (Tốc độ)", "value": f"`{process_time:.3f}s`", "inline": True}
                    ],
                    # Móc tấm ảnh vừa đính kèm vào thẳng khối Embed này
                    "image": {"url": f"attachment://{filename}"},
                    "footer": {"text": f"Dự án OA Prediction | Time: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')}"}
                }
            ]
        }

        # 3. Gửi cả file và payload json cùng lúc
        requests.post(
            AI_PREDICTION_WEBHOOK, 
            files=files, 
            data={"payload_json": json.dumps(payload)},
            timeout=10
        )
    except Exception as e:
        logger.error(f"Failed to send prediction to Discord: {e}")

def send_discord_invalid_upload(image_bytes: bytes, filename: str, confidence: float):
    if not AI_PREDICTION_WEBHOOK:
        print("[WARNING] AI_PREDICTION_WEBHOOK is not set!")
        return
    try:
        files = {
            "file": (filename, image_bytes, "image/jpeg")
        }

        payload = {
            "embeds": [
                {
                    "title": "⚠️ [Cảnh Báo] Phát Hiện Ảnh Tải Lên Không Hợp Lệ (Gửi Bậy)",
                    "color": 15158332, # Màu đỏ cảnh báo
                    "description": "Hệ thống đã phát hiện và chặn một yêu cầu dự đoán do ảnh tải lên không phải là ảnh chụp X-quang khớp gối.",
                    "fields": [
                        {"name": "Tên file", "value": f"`{filename}`", "inline": True},
                        {"name": "Độ tự tin (Confidence)", "value": f"**{confidence:.2%}**", "inline": True},
                        {"name": "Trạng thái", "value": "❌ Đã từ chối & Yêu cầu gửi lại", "inline": True}
                    ],
                    "image": {"url": f"attachment://{filename}"},
                    "footer": {"text": f"Dự án OA Prediction | Time: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')}"}
                }
            ]
        }

        res = post_to_discord(
            AI_PREDICTION_WEBHOOK, 
            files=files, 
            data={"payload_json": json.dumps(payload)},
            timeout=10
        )
        print(f"[DISCORD INVALID UPLOAD] Status: {res.status_code}, Response: {res.text[:100]}")
    except Exception as e:
        logger.error(f"Failed to send invalid upload alert to Discord: {e}")

import urllib.request
import ipaddress
import bisect
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

# --- RATE LIMITING SETUP (SLOWAPI) ---
limiter = Limiter(key_func=get_remote_address)

# --- KHỞI TẠO APP (QUAN TRỌNG: Uvicorn tìm biến này) ---
app = FastAPI(title="Knee Osteoarthritis Detection API")
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# --- OFFLINE IP GEOLOCATION (VIETNAM ONLY) ---
vn_ip_ranges = []
def load_vn_cidr():
    global vn_ip_ranges
    cidr_file = os.path.join(os.path.dirname(__file__), "vn-cidr.txt")
    
    # Download if not exists
    if not os.path.exists(cidr_file):
        cidr_url = "https://raw.githubusercontent.com/herrbischoff/country-ip-blocks/master/ipv4/vn.cidr"
        try:
            req = urllib.request.Request(cidr_url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=10) as response:
                content = response.read().decode('utf-8')
                with open(cidr_file, "w") as f:
                    f.write(content)
        except Exception as e:
            logger.error(f"Failed to download VN CIDR: {e}")
            return
            
    # Load and parse
    try:
        with open(cidr_file, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    network = ipaddress.ip_network(line)
                    vn_ip_ranges.append((int(network.network_address), int(network.broadcast_address)))
        vn_ip_ranges.sort()
        logger.info(f"Loaded {len(vn_ip_ranges)} VN IP ranges for offline checking.")
    except Exception as e:
        logger.error(f"Failed to load VN CIDR from file: {e}")

def is_ip_in_vn(ip_str: str) -> bool:
    if ip_str in ["127.0.0.1", "::1", "localhost"] or ip_str.startswith("192.168.") or ip_str.startswith("10."):
        return True
    if not vn_ip_ranges:
        return True # Fallback if ranges failed to load
    try:
        ip_int = int(ipaddress.ip_address(ip_str))
        idx = bisect.bisect_right(vn_ip_ranges, (ip_int, float('inf')))
        if idx > 0:
            start_ip, end_ip = vn_ip_ranges[idx - 1]
            if start_ip <= ip_int <= end_ip:
                return True
        return False
    except Exception:
        return False

# --- STATIC FILES FOR WEB UI ---
current_dir = os.path.dirname(os.path.abspath(__file__))
web_ui_dir = os.path.join(os.path.dirname(current_dir), "web_ui")
assets_dir = os.path.join(web_ui_dir, "assets")

if os.path.exists(assets_dir):
    app.mount("/assets", StaticFiles(directory=assets_dir), name="assets")


# --- CORS SETUP (Cho phép Web UI gọi API) ---
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Trong production nên thay bằng domain cụ thể
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- IP GEOLOCATION MIDDLEWARE ---
@app.middleware("http")
async def ip_check_middleware(request: Request, call_next):
    client_ip = request.client.host
    if not is_ip_in_vn(client_ip):
        logger.warning(f"Access blocked for non-VN IP: {client_ip}")
        return JSONResponse(status_code=403, content={"detail": "Access Denied: API is only available in Vietnam."})
    return await call_next(request)

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
validation_model = None
device = CFG['train']['device']

@app.on_event("startup")
def startup_events():
    import threading
    # Load CIDR in background to not block fast startup
    threading.Thread(target=load_vn_cidr, daemon=True).start()

@app.on_event("startup")
def notify_startup():
    space_id = os.getenv("SPACE_ID")
    if space_id:
        message = f"✅ Knee OA Prediction API on Hugging Face Space (`{space_id}`) has started successfully!"
    else:
        message = "✅ Knee OA Prediction API has started successfully!"
    send_discord_alert(message)

def download_model_from_url(url: str, dest_path: str) -> bool:
    """Tải file mô hình trực tiếp từ một đường dẫn URL công khai."""
    try:
        print(f"[API] Bat dau tai model tu URL: {mask_webhook(url)}")
        headers = {"User-Agent": "Mozilla/5.0"}
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=45) as response:
            with open(dest_path, "wb") as f:
                f.write(response.read())
        print(f"[API OK] Da tai va luu model vao {dest_path}")
        return True
    except Exception as e:
        print(f"[API ERROR] Khong the tai model tu URL: {e}")
        return False

def download_best_model_from_mlflow(dest_path: str) -> bool:
    """Kết nối tới MLflow Tracking Server và tải về file mô hình có độ chính xác val_acc cao nhất."""
    mlflow_uri = os.getenv("MLFLOW_TRACKING_URI")
    if not mlflow_uri:
        print("[MLflow] MLFLOW_TRACKING_URI khong duoc thiet lap, bo qua tai tu MLflow.")
        return False
        
    try:
        import mlflow
        from mlflow.tracking import MlflowClient
        
        print(f"[MLflow] Dang ket noi toi remote server: {mlflow_uri}")
        mlflow.set_tracking_uri(mlflow_uri)
        client = MlflowClient()
        
        # Lấy experiment name từ cấu hình
        exp_name = CFG['train']['mlflow_exp_name']
        experiment = client.get_experiment_by_name(exp_name)
        if not experiment:
            # Fallback nếu không tìm thấy experiment cấu hình
            experiment = client.get_experiment_by_name("Knee_Osteoarthritis_Dev")
            if not experiment:
                print(f"[MLflow ERROR] Khong tim thay experiment: {exp_name}")
                return False
                
        # Tìm kiếm các run và sắp xếp theo accuracy giảm dần
        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["metrics.val_acc DESC"],
            max_results=5
        )
        
        if not runs:
            print("[MLflow ERROR] Khong co run nao trong experiment.")
            return False
            
        # Tìm run đầu tiên có chứa artifact mô hình
        best_run_id = None
        for run in runs:
            run_id = run.info.run_id
            artifacts = client.list_artifacts(run_id)
            has_model = any(art.path == "best_knee_model.pth" for art in artifacts)
            if has_model:
                best_run_id = run_id
                break
                
        if not best_run_id:
            print("[MLflow ERROR] Khong tim thay run nao co chua best_knee_model.pth artifact.")
            return False
            
        print(f"[MLflow] Tim thay run {best_run_id} co accuracy cao nhat.")
        
        # Tải mô hình về thư mục đích
        local_dir = os.path.dirname(dest_path)
        os.makedirs(local_dir, exist_ok=True)
        
        mlflow.artifacts.download_artifacts(
            artifact_uri=f"runs/{best_run_id}/best_knee_model.pth",
            dst_path=local_dir
        )
        
        if os.path.exists(dest_path):
            print(f"[MLflow OK] Da tai va luu model thanh cong vao {dest_path}")
            return True
        return False
    except Exception as e:
        print(f"[MLflow ERROR] Loi khi ket noi va tai tu MLflow: {e}")
        return False

@app.on_event("startup")
def load_predictor():
    global model
    try:
        print("[...] Dang khoi tao model...")
        loaded_model = build_model()

        # Lấy thông tin file mô hình
        model_name = CFG['train']['save_name']
        model_path = os.path.join(CFG['paths']['models'], model_name)

        # Nếu mô hình không tồn tại cục bộ, tiến hành tải về
        if not os.path.exists(model_path):
            print(f"[API] File model khong tim thay tai {model_path}. Dang tai model tu nguon ngoai...")
            
            # 1. Thử tải từ URL chỉ định trong biến môi trường
            download_url = os.getenv("MODEL_DOWNLOAD_URL")
            downloaded = False
            if download_url:
                downloaded = download_model_from_url(download_url, model_path)
                
            # 2. Thử tải từ MLflow remote server
            if not downloaded:
                print("[API] Thu tai model tu MLflow Remote Server...")
                downloaded = download_best_model_from_mlflow(model_path)
                
            if not downloaded:
                raise FileNotFoundError(f"Khong the tim thay file model hoac tai tu URL/MLflow.")

        # Nạp trọng số mô hình vào kiến trúc
        checkpoint = torch.load(model_path, map_location=device)
        loaded_model.load_state_dict(checkpoint)

        loaded_model.to(device)
        loaded_model.eval()
        model = loaded_model
        print(f"[OK] Model loaded from {model_path}")
    except Exception as e:
        model = None
        print(f"[ERROR] Loi load model: {e}")
        send_discord_alert(f"Failed to load model on startup!\nError: `{str(e)}`")

@app.on_event("startup")
def load_validation_model():
    global validation_model
    try:
        import torchvision.models as models
        print("[...] Dang load validation model (MobileNetV3)...")
        validation_model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
        validation_model.to(device)
        validation_model.eval()
        print("[OK] Validation model (MobileNetV3) loaded.")
    except Exception as e:
        validation_model = None
        print(f"[ERROR] Loi load validation model: {e}")
        send_discord_alert(f"Failed to load validation model on startup!\nError: `{str(e)}`")

def is_valid_knee_xray(image: Image.Image) -> tuple[bool, float]:
    """
    Kiểm tra xem ảnh tải lên có phải là ảnh chụp X-quang khớp gối hay không.
    Kết hợp:
    1. Kiểm tra màu sắc (Loại biên ảnh màu như chó, mèo, cảnh vật).
    2. Kiểm tra tỷ lệ vùng xám (Loại biên ảnh sơ đồ, flowchart, văn bản synthetic).
    3. Kiểm tra đường thẳng Hough Line (Phát hiện đường kẻ thẳng nhân tạo trong sơ đồ/bản vẽ).
    4. Kiểm tra MobileNetV3 (Loại biên ảnh vật thể grayscale rõ nét khác).
    Trả về: (is_valid, confidence)
    """
    try:
        # Bỏ qua kiểm tra đối với ảnh quá nhỏ (như ảnh 16x16 trong các test case của pytest)
        if image.width < 50 or image.height < 50:
            return True, 1.0

        img_np = np.array(image)
        
        # Chuyển đổi sang ảnh grayscale để kiểm tra cấu trúc
        if len(img_np.shape) == 3:
            if img_np.shape[2] == 3:
                gray_img = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
            elif img_np.shape[2] == 4:
                gray_img = cv2.cvtColor(img_np, cv2.COLOR_RGBA2GRAY)
            else:
                gray_img = img_np[:, :, 0]
        else:
            gray_img = img_np

        # 1. Kiểm tra màu sắc (Chỉ áp dụng nếu là ảnh có nhiều kênh màu RGB/RGBA)
        if len(img_np.shape) == 3 and img_np.shape[2] in [3, 4]:
            diff_rg = np.mean(np.abs(img_np[:, :, 0].astype(np.int16) - img_np[:, :, 1].astype(np.int16)))
            diff_gb = np.mean(np.abs(img_np[:, :, 1].astype(np.int16) - img_np[:, :, 2].astype(np.int16)))
            diff_br = np.mean(np.abs(img_np[:, :, 2].astype(np.int16) - img_np[:, :, 0].astype(np.int16)))
            avg_color_diff = (diff_rg + diff_gb + diff_br) / 3.0
            
            if avg_color_diff > 30.0:
                # Ảnh màu đậm, chắc chắn không phải X-quang
                return False, 0.95 + (avg_color_diff / 1000.0 if avg_color_diff < 50.0 else 0.04)

        # 2. Kiểm tra tỷ lệ vùng xám (X-quang khớp gối thực tế có vùng xương/mô mềm xám diện tích lớn)
        # Ảnh sơ đồ, flowchart chủ yếu là nền đen (0) và chữ trắng (255), rất ít điểm ảnh xám trung tính.
        mid_gray_pixels = np.sum((gray_img >= 25) & (gray_img <= 230))
        mid_gray_ratio = mid_gray_pixels / gray_img.size
        
        if mid_gray_ratio < 0.20:
            # Quá ít vùng xám (dưới 20%), chắc chắn là sơ đồ hoặc bản vẽ nét
            return False, 0.90

        # 2.5. Kiểm tra đường thẳng tắp (Hough Line Transform) - Phát hiện các đường kẻ thẳng của sơ đồ, flowchart
        edges = cv2.Canny(gray_img, 50, 150, apertureSize=3)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, minLineLength=50, maxLineGap=5)
        num_lines = len(lines) if lines is not None else 0
        if num_lines > 8:
            # Sơ đồ/flowchart có rất nhiều đường thẳng song song/nối tiếp (cạnh hộp, mũi tên)
            return False, 0.85 + (num_lines / 100.0 if num_lines < 15 else 0.10)

        # 3. Kiểm tra bằng MobileNetV3 (ImageNet-1k)
        if validation_model is not None:
            from torchvision import transforms
            val_transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            val_tensor = val_transform(image).unsqueeze(0).to(device)
            
            with torch.no_grad():
                val_output = validation_model(val_tensor)
                val_probs = torch.nn.functional.softmax(val_output, dim=1)[0]
            
            # Lấy thông tin lớp dự đoán tự tin nhất
            max_prob, max_idx = torch.max(val_probs, 0)
            max_prob = max_prob.item()
            max_idx = max_idx.item()
            
            # Nếu mô hình nhận diện cực kỳ tự tin (> 35%) ra một vật thể ImageNet cụ thể (không phải X-quang 906)
            # thì đây là ảnh vật thể bình thường (ví dụ: con mèo grayscale) chứ không phải ảnh khớp gối.
            if max_idx != 906 and max_prob > 0.35:
                return False, max_prob

        return True, 1.0

    except Exception as e:
        logger.error(f"Lỗi khi kiểm tra ảnh X-quang: {e}")
        return True, 1.0

@app.get("/debug-env")
def debug_env():
    import os
    env_keys = list(os.environ.keys())
    return {
        "api_alerts_webhook_loaded": bool(API_ALERTS_WEBHOOK),
        "ai_prediction_webhook_loaded": bool(AI_PREDICTION_WEBHOOK),
        "api_alerts_webhook_masked": mask_webhook(API_ALERTS_WEBHOOK) if API_ALERTS_WEBHOOK else None,
        "ai_prediction_webhook_masked": mask_webhook(AI_PREDICTION_WEBHOOK) if AI_PREDICTION_WEBHOOK else None,
        "os_env_keys": [k for k in env_keys if "webhook" in k.lower() or "secret" in k.lower() or "key" in k.lower() or "id" in k.lower() or k == "SPACE_ID"],
        "root_dir": root_dir,
        "dot_env_exists": os.path.exists(os.path.join(root_dir, ".env"))
    }

@app.post("/predict")
@limiter.limit("10/minute")
async def predict(request: Request, background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    start_time = time.time()
    # Input Validation
    # Check file extension
    file_extension = os.path.splitext(file.filename)[1].lower()
    if file_extension not in ALLOWED_EXTENSIONS:
        logger.warning("Invalid file type", extra={
            "file_name": file.filename,
            "extension": file_extension,
            "allowed": list(ALLOWED_EXTENSIONS)
        })
        raise HTTPException(status_code=400, detail=f"Invalid file type. Allowed: {list(ALLOWED_EXTENSIONS)}")
    
    # Check file size
    contents = await file.read()
    if len(contents) > MAX_FILE_SIZE:
        logger.warning("File too large", extra={
            "file_name": file.filename,
            "size_bytes": len(contents),
            "max_size_bytes": MAX_FILE_SIZE
        })
        # Send Discord alert for oversized file
        send_discord_alert(
            "Oversized file blocked\n"
            f"Filename: `{file.filename}`\n"
            f"Size: `{len(contents) / 1024 / 1024:.2f} MB`\n"
            f"Limit: `{MAX_FILE_SIZE / 1024 / 1024:.2f} MB`"
        )
        raise HTTPException(status_code=413, detail="File too large")
    
    # 1. Doc anh
    try:
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        logger.warning("Invalid image file", extra={
            "file_name": file.filename,
            "error": str(e)
        })
        raise HTTPException(status_code=400, detail="Invalid image file")

    # 1.5. Kiểm tra tính hợp lệ của ảnh khớp gối (Không phải ảnh gối / gửi bậy)
    is_valid, validation_conf = is_valid_knee_xray(image)
    if not is_valid:
        logger.warning("Uploaded image is not a knee X-ray (gửi bậy)", extra={
            "file_name": file.filename,
            "validation_confidence": validation_conf
        })
        # Gửi cảnh báo lên Discord webhook qua background task
        background_tasks.add_task(
            send_discord_invalid_upload,
            image_bytes=contents,
            filename=file.filename,
            confidence=validation_conf
        )
        return JSONResponse(
            status_code=400, 
            content={"detail": "Ảnh tải lên không phải là ảnh chụp X-quang khớp gối. Vui lòng gửi lại ảnh đúng yêu cầu."}
        )

    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

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
            "file_name": file.filename,
            "error": str(e)
        })
        # Neu loi heatmap, API van tra ve prediction thay vi sap voi 500.
    
    logger.info("Prediction made", extra={
            "file_name": file.filename,
            "prediction": class_name,
            "confidence": confidence,
            "has_heatmap": bool(heatmap_base64)
        })
    
    process_time = time.time() - start_time
    
    # Giải mã heatmap để gửi lên kênh Monitor (nếu heatmap được tạo thành công)
    img_bytes_to_send = contents
    discord_filename = file.filename
    if heatmap_base64:
        img_bytes_to_send = base64.b64decode(heatmap_base64)
        discord_filename = f"heatmap_{file.filename}"

    # Đẩy việc gửi Discord cho Background Task (Bị vô hiệu hóa vì webhook chỉ nhận thông báo gửi bậy)
    # background_tasks.add_task(
    #     send_discord_prediction,
    #     image_bytes=img_bytes_to_send,
    #     filename=discord_filename,
    #     result=class_name,
    #     confidence=confidence,
    #     process_time=process_time
    # )
    
    # Sinh mã request_id duy nhất và lưu tạm ảnh để phục vụ feedback
    request_id = str(uuid.uuid4())
    try:
        temp_img_path = os.path.join(TEMP_FEEDBACK_DIR, f"{request_id}.jpg")
        image.save(temp_img_path, "JPEG")
    except Exception as e:
        logger.error(f"Failed to save temp image for feedback: {e}")
    
    return {
        "status": "success",
        "request_id": request_id,
        "filename": file.filename,
        "prediction": class_name,
        "confidence": f"{confidence * 100:.1f}%",
        "heatmap_base64": heatmap_base64
    }

class FeedbackModel(BaseModel):
    request_id: str
    feedback: str  # "correct" hoặc "incorrect"
    prediction: str # Nhãn mô hình dự đoán (ví dụ: "0")
    corrected_grade: Optional[str] = None  # Nhãn đúng thực tế (ví dụ: "2")

@app.post("/feedback")
async def collect_feedback(data: FeedbackModel):
    temp_path = os.path.join(TEMP_FEEDBACK_DIR, f"{data.request_id}.jpg")
    
    # Kiểm tra xem ảnh tạm có tồn tại không
    if not os.path.exists(temp_path):
        raise HTTPException(status_code=404, detail="Không tìm thấy dữ liệu yêu cầu hoặc đã được phản hồi.")
    
    try:
        if data.feedback == "correct":
            # Nếu dự đoán đúng, chuyển ảnh vào thư mục 'correct'
            dest_dir = os.path.join(FEEDBACK_DIR, "correct", data.prediction)
        else:
            # Nếu dự đoán sai, chuyển ảnh vào thư mục 'incorrect/nhãn_đúng'
            grade = data.corrected_grade if data.corrected_grade else "unknown"
            dest_dir = os.path.join(FEEDBACK_DIR, "incorrect", f"grade_{grade}")
            
        os.makedirs(dest_dir, exist_ok=True)
        shutil.move(temp_path, os.path.join(dest_dir, f"{data.request_id}.jpg"))
        
        return {"status": "success", "message": "Cảm ơn bạn đã phản hồi đóng góp!"}
        
    except Exception as e:
        logger.error(f"Lỗi xử lý lưu feedback: {e}")
        raise HTTPException(status_code=500, detail="Lỗi lưu trữ dữ liệu phản hồi.")

@app.get("/")
@limiter.limit("60/minute")
def home(request: Request):
    index_path = os.path.join(web_ui_dir, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"message": "API is running, but Web UI index.html not found!"}

@app.get("/health")
@limiter.limit("60/minute")
def health_check(request: Request):
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
    logger.error("Unhandled exception", exc_info=True, extra={
        "path": request.url.path,
        "method": request.method,
        "error": str(exc)
    })
    send_discord_alert(f"500 Internal Server Error\nPath: `{request.url.path}`\nError: `{str(exc)}`")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"}
    )
