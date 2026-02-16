import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import json
from sklearn.preprocessing import LabelEncoder
import numpy as np

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return torch.relu(out)

class AttentionModule(nn.Module):
    def __init__(self, channels):
        super(AttentionModule, self).__init__()
        self.query = nn.Conv2d(channels, channels // 8, 1)
        self.key = nn.Conv2d(channels, channels // 8, 1)
        self.value = nn.Conv2d(channels, channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))
    
    def forward(self, x):
        B, C, H, W = x.size()
        q = self.query(x).view(B, -1, H * W).permute(0, 2, 1)
        k = self.key(x).view(B, -1, H * W)
        attn = torch.softmax(torch.bmm(q, k), dim=-1)
        v = self.value(x).view(B, -1, H * W)
        out = torch.bmm(v, attn.permute(0, 2, 1)).view(B, C, H, W)
        return self.gamma * out + x

class MultiTaskCNN(nn.Module):
    def __init__(self, num_classes):
        super(MultiTaskCNN, self).__init__()
        # Backbone with residual blocks
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, 7, 2, 3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(3, 2, 1)
        )
        
        self.layer1 = self._make_layer(64, 128, 2)
        self.layer2 = self._make_layer(128, 256, 2)
        self.layer3 = self._make_layer(256, 512, 2)
        
        self.attention1 = AttentionModule(256)
        self.attention2 = AttentionModule(512)
        
        # Classification head
        self.cls_head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
        
        # Detection head
        self.det_head = nn.Sequential(
            nn.Conv2d(512, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 4),
            nn.Sigmoid()
        )
    
    def _make_layer(self, in_channels, out_channels, num_blocks):
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride=2))
        for _ in range(num_blocks - 1):
            layers.append(ResidualBlock(out_channels, out_channels))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.attention1(x)
        x = self.layer3(x)
        x = self.attention2(x)
        
        cls_out = self.cls_head(x)
        det_out = self.det_head(x)
        return cls_out, det_out

class CatDataset(Dataset):
    def __init__(self, images_dir, labels_dir, transform=None):
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.transform = transform
        self.images = []
        self.labels = []
        self.boxes = []
        
        for img_file in os.listdir(images_dir):
            if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                img_path = os.path.join(images_dir, img_file)
                label_file = img_file.rsplit('.', 1)[0] + '.txt'
                label_path = os.path.join(labels_dir, label_file)
                
                if os.path.exists(label_path):
                    with open(label_path, 'r') as f:
                        line = f.readline().strip()
                        if line:
                            parts = line.split()
                            class_id = int(parts[0])
                            # YOLO format: class_id x_center y_center width height
                            bbox = [float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])]
                            self.images.append(img_path)
                            self.labels.append(class_id)
                            self.boxes.append(bbox)
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = Image.open(self.images[idx]).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, self.labels[idx], torch.tensor(self.boxes[idx], dtype=torch.float32)

def train_lightweight_model():
    # Data transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Dataset
    train_dataset = CatDataset(
        r"S:\SE4AI\Cat-5\train\images",
        r"S:\SE4AI\Cat-5\train\labels",
        transform=transform
    )
    
    val_dataset = CatDataset(
        r"S:\SE4AI\Cat-5\valid\images", 
        r"S:\SE4AI\Cat-5\valid\labels",
        transform=transform
    )
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=2)
    
    # Model
    num_classes = len(set(train_dataset.labels))
    model = MultiTaskCNN(num_classes)
    
    # Training
    cls_criterion = nn.CrossEntropyLoss()
    det_criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    print(f"Training on {device}")
    print(f"Model size: {sum(p.numel() for p in model.parameters())} parameters")
    
    for epoch in range(5):
        model.train()
        train_loss = 0
        for batch_idx, (data, target, bbox) in enumerate(train_loader):
            data, target, bbox = data.to(device), target.to(device), bbox.to(device)
            optimizer.zero_grad()
            cls_out, det_out = model(data)
            cls_loss = cls_criterion(cls_out, target)
            det_loss = det_criterion(det_out, bbox)
            loss = cls_loss + 0.5 * det_loss
            loss.backward()
            optimizer.step()
            scheduler.step()
            train_loss += loss.item()
            
            if batch_idx % 10 == 0:
                print(f'Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f} (Cls: {cls_loss.item():.4f}, Det: {det_loss.item():.4f})')
        
        # Validation
        model.eval()
        val_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target, bbox in val_loader:
                data, target, bbox = data.to(device), target.to(device), bbox.to(device)
                cls_out, det_out = model(data)
                cls_loss = cls_criterion(cls_out, target)
                det_loss = det_criterion(det_out, bbox)
                val_loss += (cls_loss + 0.5 * det_loss).item()
                pred = cls_out.argmax(dim=1)
                correct += pred.eq(target).sum().item()
        
        print(f'Epoch {epoch}: Train Loss: {train_loss/len(train_loader):.4f}, '
              f'Val Loss: {val_loss/len(val_loader):.4f}, '
              f'Val Acc: {100.*correct/len(val_dataset):.2f}%')
        print()
    # Save model
    os.makedirs('lightweight_model', exist_ok=True)
    torch.save(model.state_dict(), 'lightweight_model/model.pth')
    
    # Save label mapping
    unique_labels = sorted(set(train_dataset.labels))
    label_mapping = {str(i): f"class_{label}" for i, label in enumerate(unique_labels)}
    with open('lightweight_model/label_mapping.json', 'w') as f:
        json.dump(label_mapping, f)
    
    # Save model config
    config = {
        'num_classes': num_classes,
        'input_size': [512, 512],
        'model_type': 'MultiTaskCNN'
    }
    with open('lightweight_model/config.json', 'w') as f:
        json.dump(config, f)
    
    print("Lightweight model saved!")
    
    # Check model size
    model_size = os.path.getsize('lightweight_model/model.pth') / (1024 * 1024)
    print(f"Model size: {model_size:.2f} MB")

if __name__ == "__main__":
    train_lightweight_model()