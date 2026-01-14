import yaml
import os
import torch

# Đường dẫn gốc của Project (dựa trên vị trí file này nằm trong src/)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def load_config(config_path="configs/config.yml"):
    full_path = os.path.join(BASE_DIR, config_path)
    
    if not os.path.exists(full_path):
        raise FileNotFoundError(f"❌ Không tìm thấy file config tại: {full_path}")

    with open(full_path, "r") as f:
        cfg = yaml.safe_load(f)
    
    # --- TỰ ĐỘNG CẤU HÌNH ĐƯỜNG DẪN ---
    cfg['paths'] = {}
    cfg['paths']['root'] = BASE_DIR
    cfg['paths']['data'] = os.path.join(BASE_DIR, "data", cfg['data']['dataset_name'])
    cfg['paths']['models'] = os.path.join(BASE_DIR, "models")
    
    # Tạo folder models nếu chưa có
    os.makedirs(cfg['paths']['models'], exist_ok=True)

    # --- TỰ ĐỘNG CẤU HÌNH DEVICE ---
    if cfg['train']['device'] == "auto":
        cfg['train']['device'] = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        cfg['train']['device'] = torch.device(cfg['train']['device'])

    return cfg

# Biến global để các file khác import
CFG = load_config()
print(f"✅ Đã load cấu hình. Thiết bị: {CFG['train']['device']}")