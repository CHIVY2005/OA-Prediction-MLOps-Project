import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import mlflow
from .config_loader import CFG
from .model import build_model
from .data_loader import get_data_loaders

def train():
    device = CFG['train']['device']
    epochs = CFG['train']['epochs']
    lr = CFG['train']['learning_rate']
    
    # 1. Prepare Data & Model
    print(f"🚀 Bắt đầu quá trình huấn luyện trên: {device}")
    train_loader, val_loader, loss_weights = get_data_loaders()
    loss_weights = loss_weights.to(device)
    
    model = build_model()
    
    # 2. Setup Loss & Optimizer
    criterion = nn.CrossEntropyLoss(weight=loss_weights, label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.1, patience=3
    )

    # 3. Setup MLflow
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment(CFG['train']['mlflow_exp_name'])

    print(f"🚀 Training {epochs} epochs...")
    
    with mlflow.start_run():
        # Log toàn bộ config lên MLflow để sau này đối chiếu
        mlflow.log_params(CFG['train'])
        mlflow.log_params(CFG['data'])
        mlflow.log_param("model_name", CFG['model']['name'])

        best_acc = 0.0
        
        for epoch in range(epochs):
            # --- TRAIN ---
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

            # --- VAL ---
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

            # Metrics
            avg_train_loss = train_loss / len(train_loader.dataset)
            avg_val_loss = val_loss / len(val_loader.dataset)
            val_acc = np.mean(np.array(all_preds) == np.array(all_labels))

            # Scheduler Step
            scheduler.step(avg_val_loss)
            current_lr = optimizer.param_groups[0]['lr']
            
            # Print & Log
            print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Acc: {val_acc:.4f} | LR: {current_lr:.1e}")
            
            mlflow.log_metrics({
                "train_loss": avg_train_loss, 
                "val_loss": avg_val_loss, 
                "val_acc": val_acc,
                "learning_rate": current_lr
            }, step=epoch)

            # Save Best Model
            if val_acc > best_acc:
                best_acc = val_acc
                save_name = CFG['train']['save_name']
                save_path = os.path.join(CFG['paths']['models'], save_name)
                torch.save(model.state_dict(), save_path)
                print(f"🌟 Đã lưu model tốt nhất: {save_path}")
                mlflow.log_artifact(save_path)

if __name__ == "__main__":
    train()