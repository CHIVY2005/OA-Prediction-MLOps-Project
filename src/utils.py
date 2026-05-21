import cv2
import numpy as np
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from PIL import Image

# src/utils.py (Sửa lại hàm get_heatmap)

def get_heatmap(model, input_tensor, original_image):
    # 1. QUAN TRỌNG: Đảm bảo model tính toán được Gradient
    # Bật requires_grad cho các lớp Convolution cuối cùng để GradCAM chạy được
    for param in model.features.parameters():
        param.requires_grad = True
    
    # 2. Setup GradCAM
    target_layers = [model.features[-1]]
    cam = GradCAM(model=model, target_layers=target_layers)
    
    # 3. Chạy GradCAM
    grayscale_cam = cam(input_tensor=input_tensor, targets=None)[0, :]
    
    # 4. Xử lý ảnh gốc để overlay
    if isinstance(original_image, Image.Image):
        original_image = np.array(original_image)
        
    original_image = cv2.resize(original_image, (224, 224))
    # Normalize về 0-1
    original_image = np.float32(original_image) / 255.0
    
    # 5. Tạo ảnh màu
    visualization = show_cam_on_image(original_image, grayscale_cam, use_rgb=True)
    
    # 6. Dọn dẹp: Đóng băng lại model (để không ảnh hưởng lần sau)
    for param in model.features.parameters():
        param.requires_grad = False
        
    return visualization
