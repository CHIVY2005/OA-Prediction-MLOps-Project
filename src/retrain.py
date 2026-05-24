import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, ConcatDataset, DataLoader, Subset, WeightedRandomSampler
from PIL import Image
import numpy as np
from collections import Counter
from sklearn.model_selection import train_test_split
import mlflow
import requests
from dotenv import load_dotenv

# Đảm bảo thư mục gốc nằm trong sys.path
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root_dir not in sys.path:
    sys.path.append(root_dir)

load_dotenv(os.path.join(root_dir, ".env"))

from src.config_loader import CFG
from src.model import build_model
from src.data_loader import KneeDataset, get_transforms, get_data_loaders

class FeedbackDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        try:
            image = Image.open(self.image_paths[idx]).convert("RGB")
            label = self.labels[idx]
            if self.transform:
                image = self.transform(image)
            return image, label
        except Exception as e:
            # Trả về một ảnh trống giả (dummy) nếu lỗi đọc file xảy ra trong quá trình huấn luyện
            print(f"[WARN] Loi doc anh feedback: {self.image_paths[idx]} - {e}")
            size = CFG['data']['img_size']
            dummy_img = Image.new("RGB", (size, size), color=0)
            if self.transform:
                dummy_img = self.transform(dummy_img)
            return dummy_img, self.labels[idx]

def collect_feedback_data() -> tuple[list[str], list[int]]:
    """Quét các thư mục phản hồi để thu thập đường dẫn ảnh và nhãn chuẩn."""
    feedback_dir = os.path.join(root_dir, "data", "feedback")
    image_paths = []
    labels = []
    
    if not os.path.exists(feedback_dir):
        return image_paths, labels

    # 1. Thu thập từ các dự đoán đúng (correct)
    correct_dir = os.path.join(feedback_dir, "correct")
    if os.path.exists(correct_dir):
        for label_name in os.listdir(correct_dir):
            lbl_path = os.path.join(correct_dir, label_name)
            if os.path.isdir(lbl_path):
                try:
                    lbl_val = int(label_name)
                    for f in os.listdir(lbl_path):
                        if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                            image_paths.append(os.path.join(lbl_path, f))
                            labels.append(lbl_val)
                except ValueError:
                    pass

    # 2. Thu thập từ các dự đoán sai được bác sĩ sửa nhãn (incorrect)
    incorrect_dir = os.path.join(feedback_dir, "incorrect")
    if os.path.exists(incorrect_dir):
        for label_name in os.listdir(incorrect_dir):
            lbl_path = os.path.join(incorrect_dir, label_name)
            if os.path.isdir(lbl_path) and label_name.startswith("grade_"):
                try:
                    lbl_val = int(label_name.split("_")[1])
                    for f in os.listdir(lbl_path):
                        if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                            image_paths.append(os.path.join(lbl_path, f))
                            labels.append(lbl_val)
                except (ValueError, IndexError):
                    pass
                    
    return image_paths, labels

def run_retraining(min_feedback_samples: int = 1):
    """Chạy huấn luyện lại tinh chỉnh (fine-tune) tăng dần bằng cách trộn dữ liệu phản hồi với dữ liệu huấn luyện gốc."""
    print("=" * 60)
    print("[Retrain] BAT DAU TIEN TRINH TU DONG HUAN LUYEN LAI (RETRAINING)")
    print("=" * 60)

    # 1. Thu thập dữ liệu phản hồi
    fb_paths, fb_labels = collect_feedback_data()
    print(f"[Retrain] Thu thap duoc {len(fb_paths)} mau tu clinician feedback.")
    
    if len(fb_paths) < min_feedback_samples:
        print(f"[Retrain] So luong mau feedback ({len(fb_paths)}) it hon yeu cau toi thieu ({min_feedback_samples}). Huy retrain.")
        return False

    device = CFG['train']['device']
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Retrain] Su dung thiet bi: {device}")

    # 2. Xây dựng các Dataset
    transforms_dict = get_transforms()
    data_path = CFG['paths']['data']
    fraction = CFG['data']['fraction']
    batch_size = CFG['train']['batch_size']
    epochs = CFG['train']['epochs']
    lr = CFG['train']['learning_rate']

    # Tải tập dữ liệu huấn luyện gốc
    full_train_ds = KneeDataset(data_path, 'train', transforms_dict['train'])
    
    # Áp dụng tỷ lệ phân tách subset nếu cấu hình nhỏ hơn 1.0
    if fraction < 1.0:
        indices = np.arange(len(full_train_ds))
        try:
            _, subset_indices = train_test_split(
                indices, test_size=fraction, 
                stratify=full_train_ds.labels, random_state=42
            )
            train_ds = Subset(full_train_ds, subset_indices)
        except ValueError:
            train_ds = Subset(full_train_ds, indices[:int(len(full_train_ds)*fraction)])
    else:
        train_ds = full_train_ds

    # Tập dữ liệu phản hồi
    feedback_ds = FeedbackDataset(fb_paths, fb_labels, transform=transforms_dict['train'])

    # Trộn hai tập dữ liệu (Gốc + Phản hồi)
    combined_dataset = ConcatDataset([train_ds, feedback_ds])
    print(f"[Retrain] Hop nhat du lieu: Train goc ({len(train_ds)}) + Feedback ({len(feedback_ds)}) = Tong ({len(combined_dataset)})")

    # 3. Tạo bộ phân bổ mẫu cân bằng (Sampler) & Loader
    if isinstance(train_ds, Subset):
        train_labels = [train_ds.dataset.labels[i] for i in train_ds.indices]
    else:
        train_labels = train_ds.labels
    
    combined_labels = list(train_labels) + list(fb_labels)
    counts = Counter(combined_labels)
    num_classes = CFG['data']['num_classes']
    class_weights = [len(combined_labels)/counts[i] if counts[i] > 0 else 1.0 for i in range(num_classes)]
    sample_weights = [class_weights[l] for l in combined_labels]
    
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
    loss_weights = torch.tensor(class_weights, dtype=torch.float).to(device)

    train_loader = DataLoader(combined_dataset, batch_size=batch_size, sampler=sampler, num_workers=0)
    _, val_loader, _ = get_data_loaders()

    # 4. Xây dựng mô hình & Nạp trọng số tốt nhất hiện có
    model = build_model()
    save_name = CFG['train']['save_name']
    save_path = os.path.join(CFG['paths']['models'], save_name)
    if os.path.exists(save_path):
        print(f"[Retrain] Doc trong so model hien tai tai {save_path} de tiep tuc fine-tune...")
        try:
            model.load_state_dict(torch.load(save_path, map_location=device))
        except Exception as e:
            print(f"[WARN] Loi khi doc weight cu: {e}. Tien hanh train moi model...")
    else:
        print("[Retrain] Khong tim thay model hien tai. Khoi tao model moi...")

    model = model.to(device)

    # 5. Thiết lập Loss & Optimizer
    # Sử dụng learning rate nhỏ hơn để thực hiện fine-tuning tránh làm mất tri thức cũ
    retrain_lr = lr * 0.5
    criterion = nn.CrossEntropyLoss(weight=loss_weights, label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=retrain_lr, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)

    # 6. Thiết lập MLflow Tracking
    mlflow_uri = os.getenv("MLFLOW_TRACKING_URI")
    if mlflow_uri:
        print(f"[MLflow] Su dung Remote Tracking URI: {mlflow_uri}")
        mlflow.set_tracking_uri(mlflow_uri)
    else:
        mlflow.set_tracking_uri("file:experiments/mlruns")
        print("[MLflow] Su dung Local Tracking: experiments/mlruns")

    mlflow.set_experiment("Knee_Osteoarthritis_Retrain")

    best_acc = 0.0
    history = {"train_loss": [], "val_loss": [], "val_acc": []}

    with mlflow.start_run():
        mlflow.log_params(CFG['train'])
        mlflow.log_param("retrain_lr", retrain_lr)
        mlflow.log_param("feedback_samples_added", len(fb_paths))
        mlflow.log_param("total_training_samples", len(combined_dataset))

        for epoch in range(epochs):
            # Pha huấn luyện
            model.train()
            train_loss = 0.0
            for imgs, labels_batch in train_loader:
                imgs, labels_batch = imgs.to(device), labels_batch.to(device)
                optimizer.zero_grad()
                outputs = model(imgs)
                loss = criterion(outputs, labels_batch)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * imgs.size(0)

            # Pha đánh giá
            model.eval()
            val_loss = 0.0
            all_preds, all_labels = [], []
            with torch.no_grad():
                for imgs, labels_batch in val_loader:
                    imgs, labels_batch = imgs.to(device), labels_batch.to(device)
                    outputs = model(imgs)
                    loss = criterion(outputs, labels_batch)
                    val_loss += loss.item() * imgs.size(0)
                    all_preds.extend(outputs.argmax(1).cpu().numpy())
                    all_labels.extend(labels_batch.cpu().numpy())

            avg_train_loss = train_loss / len(combined_dataset)
            avg_val_loss = val_loss / len(val_loader.dataset)
            val_acc = np.mean(np.array(all_preds) == np.array(all_labels))

            scheduler.step(avg_val_loss)
            current_lr = optimizer.param_groups[0]['lr']

            print(f"[Retrain] Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Acc: {val_acc:.4f} | LR: {current_lr:.1e}")

            mlflow.log_metrics({
                "train_loss": avg_train_loss, 
                "val_loss": avg_val_loss, 
                "val_acc": val_acc,
                "learning_rate": current_lr
            }, step=epoch)

            history["train_loss"].append(avg_train_loss)
            history["val_loss"].append(avg_val_loss)
            history["val_acc"].append(val_acc)

            # Lưu trọng số mới nếu đạt hiệu năng tốt hơn hoặc bằng hiệu năng cũ
            if val_acc >= best_acc:
                best_acc = val_acc
                torch.save(model.state_dict(), save_path)
                print(f"[Retrain OK] Da cap nhat weight tot nhat vao: {save_path}")
                mlflow.log_artifact(save_path)

        # 7. Gửi thông báo kết quả lên Discord
        webhook_url = os.getenv("api_alerts_webhook") or os.getenv("ai_prediction_webhook")
        if webhook_url:
            try:
                final_train_loss = history["train_loss"][-1]
                final_val_loss = history["val_loss"][-1]
                
                embed = {
                    "title": "🔄 [MLOps] Tự động Retrain Model hoàn tất!",
                    "color": 3447003,  # Màu xanh dương
                    "description": "Mô hình đã được tự động huấn luyện lại bằng cách kết hợp dữ liệu feedback và thực hiện fine-tune.",
                    "fields": [
                        {"name": "Dữ liệu Feedback mới", "value": f"`{len(fb_paths)}` ảnh", "inline": True},
                        {"name": "Tổng mẫu huấn luyện", "value": f"`{len(combined_dataset)}` ảnh", "inline": True},
                        {"name": "Số Epochs", "value": f"`{epochs}`", "inline": True},
                        {"name": "Train Loss", "value": f"`{final_train_loss:.4f}`", "inline": True},
                        {"name": "Val Loss", "value": f"`{final_val_loss:.4f}`", "inline": True},
                        {"name": "Best Accuracy", "value": f"**{best_acc:.2%}**", "inline": True}
                    ],
                    "footer": {"text": "Retraining Engine | MLflow Run Captured"}
                }
                
                headers = {"User-Agent": "Mozilla/5.0"}
                res = requests.post(webhook_url, json={"embeds": [embed]}, headers=headers, timeout=10)
                print(f"[Discord] Da gui ket qua retrain len Discord! (Status: {res.status_code})")
            except Exception as e:
                print(f"[WARN] Khong the gui alert retrain len Discord: {e}")

    print("=" * 60)
    print("[Retrain] HOAN TAT QUA TRINH HUAN LUYEN LAI.")
    print("=" * 60)
    return True

if __name__ == "__main__":
    run_retraining()
