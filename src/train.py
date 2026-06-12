import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import mlflow
import matplotlib.pyplot as plt
import requests
from dotenv import load_dotenv

load_dotenv()
from .config_loader import CFG
from .model import build_model
from .data_loader import get_data_loaders

def train():
    device = CFG['train']['device']
    epochs = CFG['train']['epochs']
    lr = CFG['train']['learning_rate']
    
    # 1. Chuẩn bị Dữ liệu & Mô hình
    print(f"[...] Bat dau qua trinh huan luyen tren: {device}")
    train_loader, val_loader, loss_weights = get_data_loaders()
    loss_weights = loss_weights.to(device)
    
    model = build_model()
    
    # 2. Thiết lập Loss & Optimizer (Hàm mất mát và Bộ tối ưu)
    criterion = nn.CrossEntropyLoss(weight=loss_weights, label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.1, patience=3
    )

    # 3. Thiết lập MLflow (Hỗ trợ Tracking từ xa hoặc tự động chuyển về local)
    mlflow_uri = os.getenv("MLFLOW_TRACKING_URI")
    if mlflow_uri:
        print(f"[MLflow] Su dung Remote Tracking URI: {mlflow_uri}")
        mlflow.set_tracking_uri(mlflow_uri)
    else:
        mlflow.set_tracking_uri("file:experiments/mlruns")
        print("[MLflow] Su dung Local Tracking: experiments/mlruns")

    mlflow.set_experiment(CFG['train']['mlflow_exp_name'])

    print(f"[...] Training {epochs} epochs...")
    
    with mlflow.start_run():
        # Log toàn bộ cấu hình lên MLflow để đối chiếu sau này
        mlflow.log_params(CFG['train'])
        mlflow.log_params(CFG['data'])
        mlflow.log_param("model_name", CFG['model']['name'])

        best_acc = 0.0
        
        # Lưu lịch sử loss/acc để vẽ biểu đồ
        history = {"train_loss": [], "val_loss": [], "val_acc": []}
        
        for epoch in range(epochs):
            # --- PHA HUẤN LUYỆN (TRAIN) ---
            model.train()
            train_loss = 0.0
            for imgs, labels in train_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * imgs.size(0)

            # --- PHA ĐÁNH GIÁ (VALIDATION) ---
            model.eval()
            val_loss = 0.0
            all_preds, all_labels = [], []
            with torch.no_grad():
                for imgs, labels in val_loader:
                    imgs, labels = imgs.to(device), labels.to(device)
                    outputs = model(imgs)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item() * imgs.size(0)
                    all_preds.extend(outputs.argmax(1).cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())

            # Tính toán các chỉ số (Metrics)
            avg_train_loss = train_loss / len(train_loader.dataset)
            avg_val_loss = val_loss / len(val_loader.dataset)
            val_acc = np.mean(np.array(all_preds) == np.array(all_labels))

            # Cập nhật learning rate qua scheduler
            scheduler.step(val_acc)
            current_lr = optimizer.param_groups[0]['lr']
            
            # In thông tin và log metric lên MLflow
            print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Acc: {val_acc:.4f} | LR: {current_lr:.1e}")
            
            mlflow.log_metrics({
                "train_loss": avg_train_loss, 
                "val_loss": avg_val_loss, 
                "val_acc": val_acc,
                "learning_rate": current_lr
            }, step=epoch)
            
            # Ghi nhận lịch sử để vẽ đồ thị
            history["train_loss"].append(avg_train_loss)
            history["val_loss"].append(avg_val_loss)
            history["val_acc"].append(val_acc)

            # Lưu model tốt nhất nếu đạt accuracy cao hơn
            if val_acc > best_acc:
                best_acc = val_acc
                save_name = CFG['train']['save_name']
                save_path = os.path.join(CFG['paths']['models'], save_name)
                torch.save(model.state_dict(), save_path)
                print(f"[OK] Da luu model tot nhat: {save_path}")
                mlflow.log_artifact(save_path)

        # 4. Vẽ biểu đồ sau khi train xong (Chỉ vẽ nếu có nhiều hơn 1 epoch)
        curve_path = None
        if epochs > 1:
            plt.figure(figsize=(12, 5))
            plt.subplot(1, 2, 1)
            plt.plot(history["train_loss"], label="Train Loss", marker="o")
            plt.plot(history["val_loss"], label="Val Loss", marker="o")
            plt.title("Loss Curve")
            plt.xlabel("Epochs")
            plt.legend()

            plt.subplot(1, 2, 2)
            plt.plot(history["val_acc"], label="Val Acc", color="green", marker="o")
            plt.title("Accuracy Curve")
            plt.xlabel("Epochs")
            plt.legend()
            
            plt.tight_layout()
            curve_path = "experiments/training_curves.png"
            plt.savefig(curve_path)
            mlflow.log_artifact(curve_path)
            print(f"[OK] Da luu va log bieu do vao: {curve_path}")

        # 5. Gửi thông báo kết quả lên Discord (Dùng kênh Gold Tier - ai_prediction_webhook để theo dõi)
        webhook_url = os.getenv("ai_prediction_webhook")
        if webhook_url:
            try:
                import json
                final_train_loss = history["train_loss"][-1]
                final_val_loss = history["val_loss"][-1]
                
                embed = {
                    "title": "✅ [MLOps] Huấn luyện Model hoàn tất!",
                    "color": 3066993, # Màu xanh lá cây
                    "fields": [
                        {"name": "Số Epochs", "value": f"`{epochs}`", "inline": True},
                        {"name": "Train Loss", "value": f"`{final_train_loss:.4f}`", "inline": True},
                        {"name": "Val Loss", "value": f"`{final_val_loss:.4f}`", "inline": True},
                        {"name": "Best Val Accuracy", "value": f"**{best_acc:.2%}**", "inline": False}
                    ],
                    "footer": {"text": "Model: EfficientNet-B0"}
                }
                
                if curve_path:
                    embed["image"] = {"url": "attachment://training_curves.png"}
                    with open(curve_path, "rb") as f:
                        files = {"file": ("training_curves.png", f, "image/png")}
                        payload = {"payload_json": json.dumps({"embeds": [embed]})}
                        requests.post(webhook_url, data=payload, files=files, timeout=10)
                else:
                    payload = {"embeds": [embed]}
                    requests.post(webhook_url, json=payload, timeout=10)
                    
                print("[OK] Da gui bao cao Training len Discord!")
            except Exception as e:
                print(f"[ERROR] Khong the gui Discord: {e}")

if __name__ == "__main__":
    train()