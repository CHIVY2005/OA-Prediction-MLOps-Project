import torch.nn as nn
from torchvision import models
from .config_loader import CFG

def build_model():
    print(f"\n--- [INFO] Khoi tao {CFG['model']['name']} ---")
    
    pretrained = CFG['model']['pretrained']
    weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.efficientnet_b0(weights=weights)
    
    # Freeze layers nếu cấu hình yêu cầu
    if CFG['model']['freeze_feature_layers']:
        for param in model.parameters():
            param.requires_grad = False
    
    # Thay đổi Classifier Head
    num_classes = CFG['data']['num_classes']
    in_features = model.classifier[1].in_features
    
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, num_classes)
    )
    
    device = CFG['train']['device']
    return model.to(device)