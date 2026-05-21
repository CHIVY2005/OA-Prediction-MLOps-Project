import torch
import torchvision.models as models
from torchvision import transforms
from PIL import Image
import json
import urllib.request

# Download ImageNet class names
url = "https://raw.githubusercontent.com/AnishShah/neulab/master/imagenet_class_index.json"
try:
    with urllib.request.urlopen(url) as response:
        class_index = json.loads(response.read().decode("utf-8"))
    # class_index is a dict of index -> [id, name]
    categories = {int(k): v[1] for k, v in class_index.items()}
    print(f"Downloaded {len(categories)} class names.")
except Exception as e:
    categories = {}
    print("Failed to download categories:", e)

# Load MobileNetV3 Small
model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
model.eval()

# Prepare image transforms for ImageNet
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load a knee X-ray image
image_path = "data/kneeKL224/val/0/9006140L.png"
try:
    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0)
    
    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.nn.functional.softmax(output, dim=1)[0]
        
    top5_prob, top5_catid = torch.topk(probs, 5)
    
    print("\n--- TOP 5 PREDICTIONS FOR KNEE X-RAY ---")
    for i in range(top5_prob.size(0)):
        idx = top5_catid[i].item()
        name = categories.get(idx, f"unknown_{idx}")
        print(f"{i+1}. Class: {name} (Index: {idx}) - Prob: {top5_prob[i].item():.4f}")
        
except Exception as e:
    print(f"Failed to run inference on {image_path}: {e}")

# Create a random noise image and run inference
try:
    noise_img = Image.new("RGB", (224, 224), color="red")
    input_tensor = transform(noise_img).unsqueeze(0)
    
    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.nn.functional.softmax(output, dim=1)[0]
        
    top5_prob, top5_catid = torch.topk(probs, 5)
    
    print("\n--- TOP 5 PREDICTIONS FOR RED IMAGE (NON-XRAY) ---")
    for i in range(top5_prob.size(0)):
        idx = top5_catid[i].item()
        name = categories.get(idx, f"unknown_{idx}")
        print(f"{i+1}. Class: {name} (Index: {idx}) - Prob: {top5_prob[i].item():.4f}")
        
except Exception as e:
    print("Failed to run inference on noise image:", e)
