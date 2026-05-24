# monitoring/drift_detection.py
import sys
import os
import argparse
import glob
import random
import json
import time
import requests
from datetime import datetime
import numpy as np
import pandas as pd
import cv2
from dotenv import load_dotenv

# Thêm thư mục gốc của dự án vào sys.path để nạp cấu hình
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)

load_dotenv(os.path.join(root_dir, ".env"))

from src.config_loader import CFG

# Nạp các thành phần từ thư viện Evidently
try:
    from evidently import Report
    from evidently.presets import DataDriftPreset
except ImportError:
    print("[ERROR] 'evidently' library is not installed. Please run: pip install evidently")
    sys.exit(1)

# --- CẤU HÌNH & HẰNG SỐ ---
REPORTS_DIR = os.path.join(root_dir, "monitoring", "reports")
os.makedirs(REPORTS_DIR, exist_ok=True)

API_ALERTS_WEBHOOK = (
    os.getenv("api_alerts_webhook") 
    or os.getenv("API_ALERTS_WEBHOOK") 
    or os.getenv("api_alert_webhook") 
    or os.getenv("API_ALERT_WEBHOOK")
)

DISCORD_DOMAINS = ["discord.com", "canary.discord.com", "ptb.discord.com", "discordapp.com"]

def post_to_discord(url: str, payload: dict) -> bool:
    """Gửi dữ liệu cảnh báo lên Discord, xử lý các trường hợp chặn hoặc giới hạn tên miền."""
    if not url:
        return False
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    
    for domain in DISCORD_DOMAINS:
        current_url = url
        for d in DISCORD_DOMAINS:
            if d in url:
                current_url = url.replace(d, domain)
                break
        try:
            res = requests.post(current_url, json=payload, headers=headers, timeout=10)
            if res.status_code < 400:
                return True
        except Exception as e:
            print(f"[Discord Alert Warning] Domain {domain} failed: {e}")
            
    return False

def extract_image_features(image_path: str) -> dict:
    """Trích xuất các siêu dữ liệu ảnh chuẩn và đặc trưng thống kê từ ảnh chụp X-ray."""
    try:
        # Tải ảnh ở dạng ảnh xám
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return None
        
        # Kích thước ảnh
        height, width = img.shape
        aspect_ratio = float(width) / float(height) if height > 0 else 1.0
        
        # Thống kê mức độ sáng (brightness) và độ tương phản (contrast)
        mean_val = float(np.mean(img))
        std_val = float(np.std(img))
        
        # Độ sắc nét: Sử dụng phương sai bộ lọc Laplacian trên kích thước chuẩn hóa (224x224)
        # nhằm đảm bảo sự khác biệt về kích thước/độ phân giải của ảnh không làm sai lệch thống kê
        img_resized = cv2.resize(img, (224, 224))
        sharpness = float(cv2.Laplacian(img_resized, cv2.CV_64F).var())
        
        return {
            "brightness": mean_val,
            "contrast": std_val,
            "sharpness": sharpness,
            "width": width,
            "height": height,
            "aspect_ratio": aspect_ratio
        }
    except Exception as e:
        print(f"[Warning] Failed to extract features for {os.path.basename(image_path)}: {e}")
        return None

def collect_images_from_dir(directory: str) -> list:
    """Tìm kiếm đệ quy các file ảnh trong thư mục."""
    extensions = ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG")
    image_paths = []
    for ext in extensions:
        image_paths.extend(glob.glob(os.path.join(directory, "**", ext), recursive=True))
    return image_paths

def copy_local_images_to_feedback(sample_size: int = 30, apply_drift: bool = True) -> bool:
    """Sao chép các ảnh mẫu từ thư mục local val/test sang thư mục feedback để mô phỏng dữ liệu thực tế."""
    import shutil
    val_dir = os.path.join(root_dir, "data", CFG['data']['dataset_name'], "val")
    if not os.path.exists(val_dir):
        val_dir = os.path.join(root_dir, "data", CFG['data']['dataset_name'], "test")
    
    if not os.path.exists(val_dir):
        print(f"[Generate] Khong tim thay thu muc data local tai {val_dir}")
        return False
        
    image_paths = collect_images_from_dir(val_dir)
    if not image_paths:
        print("[Generate] Khong co anh nao de copy.")
        return False
        
    # Lấy mẫu ngẫu nhiên ảnh để sao chép
    to_copy = random.sample(image_paths, min(len(image_paths), sample_size))
    
    feedback_dir = os.path.join(root_dir, "data", "feedback")
    correct_dir = os.path.join(feedback_dir, "correct")
    incorrect_dir = os.path.join(feedback_dir, "incorrect")
    
    # Reset các thư mục đích để tránh tích tụ dữ liệu ảo cũ
    shutil.rmtree(correct_dir, ignore_errors=True)
    shutil.rmtree(incorrect_dir, ignore_errors=True)
    
    os.makedirs(correct_dir, exist_ok=True)
    os.makedirs(incorrect_dir, exist_ok=True)
    
    print(f"[Generate] Dang copy {len(to_copy)} anh tu local vao thu muc feedback de lam du lieu vi du...")
    
    for i, path in enumerate(to_copy):
        # 70% dự đoán đúng (correct), 30% dự đoán sai (incorrect)
        is_correct = random.random() < 0.7
        filename = f"simulated_{i}_{os.path.basename(path)}"
        
        # Xác định thư mục đích dựa trên loại phản hồi
        if is_correct:
            pred_class = random.choice(CFG['data']['class_names'])
            dest_dir = os.path.join(correct_dir, pred_class)
        else:
            correct_grade = random.choice(CFG['data']['class_names'])
            dest_dir = os.path.join(incorrect_dir, f"grade_{correct_grade}")
            
        os.makedirs(dest_dir, exist_ok=True)
        dest_path = os.path.join(dest_dir, filename)
        
        if apply_drift:
            # Chỉnh sửa độ sáng trong không gian màu HSV để mô phỏng lệch dữ liệu (drift)
            img = cv2.imread(path)
            if img is not None:
                hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
                h, s, v = cv2.split(hsv)
                # Tăng kênh Value (độ sáng) lên thêm 40 đơn vị
                v = np.clip(v.astype(np.int16) + 40, 0, 255).astype(np.uint8)
                final_hsv = cv2.merge((h, s, v))
                img_drifted = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
                cv2.imwrite(dest_path, img_drifted)
            else:
                shutil.copy(path, dest_path)
        else:
            shutil.copy(path, dest_path)
            
    print(f"[OK] Da tao xong du lieu feedback mau tai: {feedback_dir}")
    return True

def get_reference_features(sample_size: int = 100) -> pd.DataFrame:
    """Thu thập đặc trưng tập tham chiếu (Reference) từ tập validation/test gốc."""
    val_data_dir = os.path.join(root_dir, "data", CFG['data']['dataset_name'], "val")
    if not os.path.exists(val_data_dir):
        # Fallback về tập train nếu không tìm thấy thư mục val
        val_data_dir = os.path.join(root_dir, "data", CFG['data']['dataset_name'], "train")
    
    if not os.path.exists(val_data_dir):
        raise FileNotFoundError(f"Reference data directory not found at {val_data_dir}")
        
    image_paths = collect_images_from_dir(val_data_dir)
    if not image_paths:
        raise ValueError(f"No reference images found in {val_data_dir}")
        
    # Lấy mẫu ảnh nếu số lượng ảnh vượt quá kích thước yêu cầu
    if len(image_paths) > sample_size:
        image_paths = random.sample(image_paths, sample_size)
        
    print(f"[Reference] Extracting features from {len(image_paths)} reference images...")
    features_list = []
    for p in image_paths:
        feats = extract_image_features(p)
        if feats:
            features_list.append(feats)
            
    return pd.DataFrame(features_list)

def get_production_features(sample_size: int = 100, force_simulate: bool = False) -> tuple[pd.DataFrame, bool]:
    """
    Thu thập đặc trưng tập thực tế (Current/Production) từ các ảnh feedback.
    Nếu không có đủ dữ liệu phản hồi thực tế, hoặc force_simulate được bật,
    tiến hành mô phỏng đặc trưng bị lệch dữ liệu (drift) phục vụ demo/thử nghiệm.
    """
    feedback_dir = os.path.join(root_dir, "data", "feedback")
    image_paths = []
    
    # Không tìm kiếm trong thư mục tạm 'temp' của feedback, chỉ tìm ở các thư mục cuối
    if os.path.exists(feedback_dir) and not force_simulate:
        correct_dir = os.path.join(feedback_dir, "correct")
        incorrect_dir = os.path.join(feedback_dir, "incorrect")
        image_paths.extend(collect_images_from_dir(correct_dir))
        image_paths.extend(collect_images_from_dir(incorrect_dir))

    # Kiểm tra xem có cần chạy mô phỏng không
    is_simulated = False
    if len(image_paths) < 5 or force_simulate:
        is_simulated = True
        print(f"[Monitoring] Thu muc feedback trong hoac qua it anh ({len(image_paths)} anh). Chay o che do MO PHONG DRIFT.")
        # Nạp dữ liệu tham chiếu và áp dụng dịch chuyển toán học để mô phỏng drift
        ref_df = get_reference_features(sample_size=sample_size)
        
        simulated_list = []
        for _, row in ref_df.iterrows():
            sim_row = row.to_dict()
            # Mô phỏng tăng 25% độ sáng (ví dụ: máy quét mới thay đổi thông số)
            sim_row["brightness"] = min(255.0, row["brightness"] * 1.25)
            # Mô phỏng giảm 15% độ tương phản (ví dụ: ảnh bị mờ hoặc sương mù)
            sim_row["contrast"] = row["contrast"] * 0.85
            # Mô phỏng giảm độ sắc nét
            sim_row["sharpness"] = row["sharpness"] * 0.90
            simulated_list.append(sim_row)
            
        return pd.DataFrame(simulated_list), is_simulated

    # Nếu đã có các ảnh feedback thực tế
    if len(image_paths) > sample_size:
        image_paths = random.sample(image_paths, sample_size)
        
    print(f"[Production] Extracting features from {len(image_paths)} feedback images...")
    features_list = []
    for p in image_paths:
        feats = extract_image_features(p)
        if feats:
            features_list.append(feats)
            
    return pd.DataFrame(features_list), is_simulated

def run_drift_check(reference_df: pd.DataFrame, current_df: pd.DataFrame, drift_threshold: float = 0.33) -> dict:
    """Chạy báo cáo Evidently Data Drift và kiểm tra xem có phát hiện lệch dữ liệu không."""
    print("[Evidently] Running data drift analysis...")
    drift_report = Report(metrics=[
        DataDriftPreset()
    ])
    
    report_result = drift_report.run(reference_data=reference_df, current_data=current_df)
    
    # Lưu các file báo cáo
    html_path = os.path.join(REPORTS_DIR, "drift_report.html")
    json_path = os.path.join(REPORTS_DIR, "drift_report.json")
    
    report_result.save_html(html_path)
    report_result.save_json(json_path)
    print(f"[OK] Report HTML saved to: {html_path}")
    print(f"[OK] Report JSON saved to: {json_path}")
    
    # Đọc file JSON kết quả để kiểm tra trạng thái
    with open(json_path, "r", encoding="utf-8") as f:
        report_data = json.load(f)
        
    # Trích xuất thông tin tóm tắt metrics
    # Cấu trúc JSON của Evidently 0.7+:
    # metrics[0] là DriftedColumnsCount, metrics[1:] là ValueDrift cho từng cột
    columns_count_metric = report_data["metrics"][0]
    drift_share = columns_count_metric["value"]["share"]
    drifted_features_count = int(columns_count_metric["value"]["count"])
    number_of_features = len(report_data["metrics"]) - 1
    
    # Dataset drift là True nếu tỷ lệ cột bị lệch (drift_share) vượt quá hoặc bằng ngưỡng
    dataset_drift = bool(drift_share >= drift_threshold)
    
    # Xác định các cột (đặc trưng) cụ thể bị lệch
    drifted_features = []
    for metric in report_data["metrics"][1:]:
        col_name = metric["config"]["column"]
        p_value = metric["value"]
        threshold = metric["config"]["threshold"]
        
        if p_value is not None and p_value < threshold:
            drifted_features.append(f"`{col_name}` (p-value: {p_value:.4f})")
            
    result = {
        "dataset_drift": dataset_drift,
        "drift_share": drift_share,
        "drift_threshold": drift_threshold,
        "total_features": number_of_features,
        "drifted_count": drifted_features_count,
        "drifted_features": drifted_features
    }
    
    return result

def send_drift_alert(summary: dict, is_simulated: bool):
    """Gửi cảnh báo có cấu trúc lên Discord khi phát hiện lệch dữ liệu."""
    if not API_ALERTS_WEBHOOK:
        print("[Warning] API_ALERTS_WEBHOOK is not set, skipping Discord notification.")
        return
        
    status_emoji = "🚨 [CẢNH BÁO] PHÁT HIỆN DATA DRIFT!" if summary["dataset_drift"] else "✅ [Giám sát] Kiểm tra Data Drift Bình Thường"
    color = 15158332 if summary["dataset_drift"] else 3066993
    
    fields = [
        {"name": "Số lượng đặc trưng lệch", "value": f"`{summary['drifted_count']}/{summary['total_features']}`", "inline": True},
        {"name": "Tỉ lệ lệch (Drift Share)", "value": f"`{summary['drift_share']:.2%}` (Ngưỡng: `{summary['drift_threshold']:.0%}`)", "inline": True},
        {"name": "Trạng thái Drift", "value": "❌ **Đã Lệch Dữ Liệu**" if summary["dataset_drift"] else "✔️ **Bình Thường**", "inline": True}
    ]
    
    if summary["drifted_features"]:
        fields.append({
            "name": "Các đặc trưng bị lệch",
            "value": ", ".join(summary["drifted_features"]),
            "inline": False
        })
        
    if is_simulated:
        fields.append({
            "name": "⚠️ Chế độ kiểm tra",
            "value": "Chạy dưới chế độ **MÔ PHỎNG DRIFT** do chưa đủ dữ liệu thực tế từ clinician feedback.",
            "inline": False
        })
        
    payload = {
        "embeds": [
            {
                "title": status_emoji,
                "color": color,
                "description": "Evidently AI đã thực hiện phân tích thống kê trên thuộc tính ảnh khớp gối giữa tập Reference (Validation) và tập Current (Clinician Feedback).",
                "fields": fields,
                "footer": {"text": f"Knee Osteoarthritis Monitoring | Time: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')}"}
            }
        ]
    }
    
    success = post_to_discord(API_ALERTS_WEBHOOK, payload)
    if success:
        print("[OK] Da gui alert data drift len Discord!")
    else:
        print("[Error] Khong the gui alert data drift len Discord.")

def main():
    parser = argparse.ArgumentParser(description="Evidently AI Data Drift Monitoring for Image Properties")
    parser.add_argument("--ref-size", type=int, default=100, help="Reference dataset sample size")
    parser.add_argument("--curr-size", type=int, default=100, help="Current production dataset sample size")
    parser.add_argument("--threshold", type=float, default=0.33, help="Threshold ratio of drifted features to trigger alarm")
    parser.add_argument("--force-simulate", action="store_true", help="Force run in drift simulation mode")
    parser.add_argument("--generate-feedback", action="store_true", help="Generate simulated feedback data by copying and drifting local dataset images")
    parser.add_argument("--no-retrain", action="store_true", help="Disable automatic retraining when drift is detected")
    args = parser.parse_args()
    
    start_time = time.time()
    print("=" * 60)
    print(f"BAT DAU KIEM TRA DATA DRIFT - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    if args.generate_feedback:
        print("[Generate] Yeu cau tao du lieu feedback mau tu local dataset...")
        copy_local_images_to_feedback(sample_size=args.curr_size, apply_drift=True)
    
    try:
        # 1. Thu thập đặc trưng tập tham chiếu (Reference)
        ref_df = get_reference_features(sample_size=args.ref_size)
        print(f"[Reference] Loaded {len(ref_df)} samples.")
        
        # 2. Thu thập đặc trưng tập thực tế (Current)
        curr_df, is_simulated = get_production_features(sample_size=args.curr_size, force_simulate=args.force_simulate)
        print(f"[Current] Loaded {len(curr_df)} samples.")
        
        # 3. Chạy báo cáo Drift
        drift_results = run_drift_check(ref_df, curr_df, drift_threshold=args.threshold)
        
        # In tóm tắt kết quả kiểm tra
        print("-" * 60)
        print(f"KET QUA GIAM SAT DRIFT:")
        print(f"  - Dataset Drift: {drift_results['dataset_drift']}")
        print(f"  - Drifted Features: {drift_results['drifted_count']}/{drift_results['total_features']} ({drift_results['drift_share']:.2%})")
        if drift_results["drifted_features"]:
            print(f"  - Chi tiet dac trung bi lech: {', '.join(drift_results['drifted_features'])}")
        print("-" * 60)
        
        # 4. Gửi cảnh báo lên Discord nếu cần
        send_drift_alert(drift_results, is_simulated)
        
        # 5. Tự động kích hoạt huấn luyện lại (Retrain) khi phát hiện drift
        if drift_results['dataset_drift']:
            if args.no_retrain:
                print("[Retrain] Phat hien drift nhung bo qua retrain do flag --no-retrain duoc bat.")
            else:
                print("[Retrain] Phat hien dataset drift! Tu dong kich hoat qua trinh huan luyen lai...")
                try:
                    from src.retrain import run_retraining
                    run_retraining(min_feedback_samples=1)
                except Exception as retrain_err:
                    print(f"[Retrain Error] Khong the chay tu dong retraining: {retrain_err}")
        
        elapsed = time.time() - start_time
        print(f"[Hoan tat] Kiem tra drift trong {elapsed:.2f} giay.")
        print("=" * 60)
        
    except Exception as e:
        print(f"[CRITICAL ERROR] Failed to perform drift detection: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
