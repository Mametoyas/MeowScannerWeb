import torch
import cv2
import numpy as np
from PIL import Image
from effdet import get_efficientdet_config, EfficientDet, DetBenchPredict
from timm.data import create_transform 
from omegaconf import OmegaConf 
from transformers import ViTImageProcessor, ViTForImageClassification
import json


class CatAnalysisSystem:
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu', custom_model_path=None):
        self.device = device
        print(f"Using device: {self.device}")
        
        # --- STAGE 1: LOAD EFFICIENTDET (Object Detection) ---
        print("Loading EfficientDet...")
        self.det_config = get_efficientdet_config('tf_efficientdet_d0')
        self.det_net = EfficientDet(self.det_config, pretrained_backbone=False)
        checkpoint = torch.hub.load_state_dict_from_url(
            "https://github.com/rwightman/efficientdet-pytorch/releases/download/v0.1/tf_efficientdet_d0_34-f153e0cf.pth", 
            map_location=device
        )
        self.det_net.load_state_dict(checkpoint)
        OmegaConf.set_readonly(self.det_config, False) 
        self.det_config.num_classes = 90     
        self.det_config.image_size = [512, 512]
        self.det_model = DetBenchPredict(self.det_net) 
        self.det_model.eval().to(self.device)
        
        print("Loading Vision Transformer (ViT)...")
        if custom_model_path:
            # Use base model processor but custom trained model
            self.vit_processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
            self.vit_model = ViTForImageClassification.from_pretrained(custom_model_path)
            # Load custom label mapping
            with open(f"{custom_model_path}/label_mapping.json", 'r') as f:
                self.custom_labels = json.load(f)
        else:
            self.vit_model_name = 'google/vit-base-patch16-224' 
            self.vit_processor = ViTImageProcessor.from_pretrained(self.vit_model_name)
            self.vit_model = ViTForImageClassification.from_pretrained(self.vit_model_name)
            self.custom_labels = None
        self.vit_model.eval().to(self.device)

    def detect_cats(self, img_cv2, threshold=0.5):
            """Stage 1: Detect objects and filter only Cats"""
            img_rgb = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB)
            src_img = Image.fromarray(img_rgb)
            img_h, img_w = self.det_config.image_size[0], self.det_config.image_size[1]
            transform = create_transform(
                input_size=(3, img_h, img_w),  # ต้องเป็น format (C, H, W)
                mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225]
            )
            img_tensor = transform(src_img).unsqueeze(0).to(self.device)

            with torch.no_grad():
                output = self.det_model(img_tensor)
            results = output.cpu().numpy()[0]
            cat_boxes = []
            CAT_CLASS_ID = 17 
            
            for res in results:
                xmin, ymin, xmax, ymax, score, class_id = res
                if score > threshold and int(class_id) == CAT_CLASS_ID:
                    h_orig, w_orig = img_cv2.shape[:2]
                    h_model, w_model = self.det_config.image_size
                    
                    scale_x = w_orig / w_model
                    scale_y = h_orig / h_model
                    
                    cat_boxes.append([
                        int(xmin * scale_x), int(ymin * scale_y), 
                        int(xmax * scale_x), int(ymax * scale_y), 
                        score
                    ])
                    
            return cat_boxes

    def classify_breed(self, crop_img_cv2):
        """Stage 2: Classify breed using ViT"""
        if crop_img_cv2.size == 0: return "Unknown", 0.0
        
        img_rgb = cv2.cvtColor(crop_img_cv2, cv2.COLOR_BGR2RGB)
        inputs = self.vit_processor(images=img_rgb, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.vit_model(**inputs)
            logits = outputs.logits
            probs = logits.softmax(dim=1)
            top_prob, top_idx = probs.max(1)
        
        if self.custom_labels:
            breed_name = self.custom_labels[str(top_idx.item())]
        else:
            breed_name = self.vit_model.config.id2label[top_idx.item()]
            breed_name = breed_name.split(',')[0] 
        
        return breed_name, top_prob.item()

    def run(self, image_path):
        print(f"Reading image from: {image_path}")
        img = cv2.imread(image_path)
        if img is None:
            print("Error: Image not found. Please check the path.")
            return

        # 1. Detection
        boxes = self.detect_cats(img)
        print(f"Found {len(boxes)} cats.")

        for box in boxes:
            xmin, ymin, xmax, ymax, score = box
            
            # Clamp coordinates
            xmin, ymin = max(0, xmin), max(0, ymin)
            xmax, ymax = min(img.shape[1], xmax), min(img.shape[0], ymax)

            # 2. Crop Image
            cat_crop = img[ymin:ymax, xmin:xmax]
            
            # 3. Classification
            breed, conf = self.classify_breed(cat_crop)
            print(f"Cat at [{xmin},{ymin}] is likely: {breed} ({conf:.2f})")

            # Draw
            cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            label = f"{breed}: {conf:.2f}"
            cv2.putText(img, label, (xmin, ymin - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Show result
        cv2.imshow("Two-Stage Cat Analysis", img)
        print("Press any key on the image window to close...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()


# --- วิธีใช้งาน ---
if __name__ == "__main__":
    # ใส่ path รูปแมวของคุณที่นี่
    image_path = r"S:\SE4AI\Cat-5\test\images\Bengal_17_jpg.rf.8a416d669cebb095c0dee2108f41aedc.jpg" 
    system = CatAnalysisSystem(custom_model_path="cat_breed_model")
    system.run(image_path)