import torch
import torch.nn as nn
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms
import json
import os

class LightweightCNN(nn.Module):
    def __init__(self, num_classes):
        super(LightweightCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((4, 4))
        )
        # Modification to your LightweightCNN
        self.classifier = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(256 * 4 * 4, 256),
            nn.ReLU()
        )
        self.class_head = nn.Linear(256, num_classes) # For Breed
        self.bbox_head = nn.Linear(256, 4)            # For [x, y, w, h]
    
    def forward(self, x):
            x = self.features(x)
            x = x.view(x.size(0), -1)
            x = self.classifier(x)
            
            # Split the output here
            class_logits = self.class_head(x)
            bbox_coords = self.bbox_head(x) 
            
            return class_logits, bbox_coords

class SimpleCatAnalysis:
    def __init__(self, model_path='lightweight_model', device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        print(f"Using device: {self.device}")
        
        # Load config
        with open(f"{model_path}/config.json", 'r') as f:
            self.config = json.load(f)
        
        # Load model
        self.model = LightweightCNN(self.config['num_classes'])
        self.model.load_state_dict(torch.load(f"{model_path}/model.pth", map_location=device))
        self.model.eval().to(self.device)
        
        # Load labels
        with open(f"{model_path}/label_mapping.json", 'r') as f:
            self.labels = json.load(f)
        
        # Transform
        self.transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        print("Lightweight model loaded!")
    
    def classify_image(self, image_path):
        img = Image.open(image_path).convert('RGB')
        img_tensor = self.transform(img).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(img_tensor)
            probs = torch.softmax(outputs, dim=1)
            top_prob, top_idx = probs.max(1)
        
        breed_name = self.labels[str(top_idx.item())]
        return breed_name, top_prob.item()
    
    def run(self, image_path):
        print(f"Analyzing: {image_path}")
        breed, confidence = self.classify_image(image_path)
        print(f"Predicted breed: {breed} (confidence: {confidence:.2f})")
        
        # Display image with prediction
        img = cv2.imread(image_path)
        if img is not None:
            label = f"{breed}: {confidence:.2f}"
            cv2.putText(img, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("Cat Breed Classification", img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

if __name__ == "__main__":
    image_path = r"S:\SE4AI\Cat-5\test\images\Bengal_17_jpg.rf.8a416d669cebb095c0dee2108f41aedc.jpg"
    
    if os.path.exists('lightweight_model'):
        system = SimpleCatAnalysis()
        system.run(image_path)
    else:
        print("Please train the lightweight model first by running: python train_lightweight.py")