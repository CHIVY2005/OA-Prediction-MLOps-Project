import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset, WeightedRandomSampler
from torchvision import transforms
from PIL import Image
from collections import Counter
from sklearn.model_selection import train_test_split
from .config_loader import CFG  # Import Config mới

class KneeDataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None):
        self.root_dir = os.path.join(root_dir, split)
        self.transform = transform
        self.images = []
        self.labels = []
        
        # Dùng tên class từ Config
        self.classes = CFG['data']['class_names']
        
        if not os.path.exists(self.root_dir):
            print(f"[WARN] Canh bao: Khong tim thay {self.root_dir}")
            return

        for label in self.classes:
            class_path = os.path.join(self.root_dir, label)
            if os.path.isdir(class_path):
                files = [os.path.join(class_path, f) for f in os.listdir(class_path) 
                         if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                self.images.extend(files)
                self.labels.extend([int(label)] * len(files))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = Image.open(self.images[idx]).convert("RGB")
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

def get_transforms():
    size = CFG['data']['img_size']
    return {
        'train': transforms.Compose([
            transforms.Resize((size, size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }

def get_data_loaders():
    transforms_dict = get_transforms()
    data_path = CFG['paths']['data']
    fraction = CFG['data']['fraction']
    batch_size = CFG['train']['batch_size']
    
    # 1. Load Data
    full_train_ds = KneeDataset(data_path, 'train', transforms_dict['train'])
    full_val_ds = KneeDataset(data_path, 'val', transforms_dict['val'])
    
    # 2. Xử lý Subset (Chạy nhanh)
    if fraction < 1.0:
        def create_stratified_subset(dataset):
            if len(dataset) == 0: return dataset
            indices = np.arange(len(dataset))
            try:
                _, subset_indices = train_test_split(
                    indices, test_size=fraction, 
                    stratify=dataset.labels, random_state=42
                )
                return Subset(dataset, subset_indices)
            except ValueError:
                # Fallback nếu dữ liệu quá ít không stratify được
                return Subset(dataset, indices[:int(len(dataset)*fraction)])
            
        train_ds = create_stratified_subset(full_train_ds)
        val_ds = create_stratified_subset(full_val_ds)
        print(f"[INFO] Che do Fraction {fraction}: Train {len(train_ds)}, Val {len(val_ds)}")
    else:
        train_ds, val_ds = full_train_ds, full_val_ds

    # 3. Xử lý Imbalance Sampler
    # Lấy nhãn chuẩn từ subset
    if isinstance(train_ds, Subset):
        labels = [train_ds.dataset.labels[i] for i in train_ds.indices]
    else:
        labels = train_ds.labels
    
    counts = Counter(labels)
    num_classes = CFG['data']['num_classes']
    class_weights = [len(labels)/counts[i] if counts[i] > 0 else 1.0 for i in range(num_classes)]
    sample_weights = [class_weights[l] for l in labels]
    
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
    loss_weights = torch.tensor(class_weights, dtype=torch.float)

    # 4. DataLoader
    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    
    return train_loader, val_loader, loss_weights