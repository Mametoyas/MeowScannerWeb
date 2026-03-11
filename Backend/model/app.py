from flask import Flask, jsonify, request
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.ops import nms
import base64
import cv2
import io
import numpy as np
from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS
import pillow_heif
from flask_cors import CORS
from Model import PretrainedCatDetector

# Register HEIC Opener
pillow_heif.register_heif_opener()

app = Flask(__name__)
CORS(app)

# Cat class mapping from database
CAT_CLASSES = {
    0: {"id": "C0001", "name": "Abyssinian"},
    1: {"id": "C0002", "name": "Bengal"},
    2: {"id": "C0003", "name": "Birman"},
    3: {"id": "C0004", "name": "Oriental"},
    4: {"id": "C0005", "name": "Other"},
    5: {"id": "C0006", "name": "Siamese"},
    6: {"id": "C0007", "name": "Somali"},
    7: {"id": "C0008", "name": "Sphynx"},
    8: {"id": "C0009", "name": "Toyger"}
}

# Load custom model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_classes = 9
model = PretrainedCatDetector(num_classes=num_classes,num_anchors=5)
model.load_state_dict(torch.load('..\\DenseNet121_CatV3_withCBAMv8.pth', map_location=device))
model.to(device)
model.eval()

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5121, 0.4615, 0.4059], std=[0.2247, 0.2278, 0.2331])
])

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_decimal_from_dms(dms, ref):
    """แปลง (องศา, ลิปดา, ฟิลิปดา) เป็นทศนิยม"""
    try:
        degrees = dms[0]
        minutes = dms[1]
        seconds = dms[2]
        
        decimal = degrees + (minutes / 60.0) + (seconds / 3600.0)
        
        # ถ้าอยู่ซีกโลกใต้ (S) หรือ ตะวันตก (W) ต้องติดลบ
        if ref in ['S', 'W']:
            decimal = -decimal
        return decimal
    except Exception:
        return None

def extract_gps(image):
    """Extract GPS coordinates from image EXIF data"""
    try:
        gps_info = {}
        
        # วิธีที่ 1: ดึงจาก get_ifd (เหมาะสำหรับ HEIC และไฟล์ใหม่ๆ)
        exif = image.getexif()
        gps_ifd = exif.get_ifd(34853) # 34853 คือ ID ของ GPS Info
        
        if gps_ifd:
            for key, value in gps_ifd.items():
                tag_name = GPSTAGS.get(key, key)
                gps_info[tag_name] = value

        # วิธีที่ 2: ดึงจาก _getexif (Fallback สำหรับ JPG เก่า)
        if not gps_info and hasattr(image, '_getexif'):
            exif_data = image._getexif()
            if exif_data:
                for tag, value in exif_data.items():
                    tag_name = TAGS.get(tag, tag)
                    if tag_name == 'GPSInfo':
                        for t in value:
                            sub_tag = GPSTAGS.get(t, t)
                            gps_info[sub_tag] = value[t]

        # ตรวจสอบและคำนวณพิกัด
        if 'GPSLatitude' in gps_info and 'GPSLongitude' in gps_info:
            lat = get_decimal_from_dms(gps_info['GPSLatitude'], gps_info.get('GPSLatitudeRef', 'N'))
            lon = get_decimal_from_dms(gps_info['GPSLongitude'], gps_info.get('GPSLongitudeRef', 'E'))
            
            if lat is not None and lon is not None:
                return {"lat": lat, "lon": lon}
        
        return None

    except Exception as e:
        print(f"Error extracting GPS: {e}")
        return None

def _draw_bounding_boxes_on_image(image, bboxes):
    """Draw bounding boxes directly on the image"""
    from PIL import ImageDraw, ImageFont
    
    # Create a copy of the image to draw on
    img_copy = image.copy()
    draw = ImageDraw.Draw(img_copy)
    
    # Try to load a font, fallback to default if not available
    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except:
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 20)
        except:
            font = ImageFont.load_default()
    
    for bbox in bboxes:
        x1, y1, x2, y2 = bbox['x1'], bbox['y1'], bbox['x2'], bbox['y2']
        label = bbox['label']
        confidence = bbox['confidence']
        
        # Scale coordinates from 224x224 to actual image size
        img_width, img_height = img_copy.size
        scale_x = img_width / 224
        scale_y = img_height / 224
        
        scaled_x1 = int(x1 * scale_x)
        scaled_y1 = int(y1 * scale_y)
        scaled_x2 = int(x2 * scale_x)
        scaled_y2 = int(y2 * scale_y)
        
        # Draw bounding box
        draw.rectangle([scaled_x1, scaled_y1, scaled_x2, scaled_y2], 
                        outline='red', width=3)
        
        # Draw label
        label_text = f"{label} ({confidence*100:.1f}%)"
        
        # Get text bounding box
        bbox_text = draw.textbbox((0, 0), label_text, font=font)
        text_width = bbox_text[2] - bbox_text[0]
        text_height = bbox_text[3] - bbox_text[1]
        
        # Position label
        label_x = scaled_x1
        label_y = scaled_y1 - text_height - 5
        
        # Adjust if label goes outside image
        if label_x + text_width > img_width:
            label_x = img_width - text_width - 5
        if label_y < 0:
            label_y = scaled_y2 + 5
        
        # Draw label background
        draw.rectangle([label_x - 2, label_y - 2, 
                       label_x + text_width + 2, label_y + text_height + 2], 
                      fill='red')
        
        # Draw label text
        draw.text((label_x, label_y), label_text, fill='white', font=font)
    
    return img_copy

@app.route("/", methods=["GET"])
def mainRoute():
    return "Model API Service"

@app.route("/health", methods=["GET"])
def check_health():
    return jsonify({"code": 200})

@app.route("/predict", methods=["POST"])
def predict():
    if 'file' not in request.files:
        return jsonify({"Error Text": "ไม่มีไฟล์"}), 400
        
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({"Error Text": "ไม่ได้เลือกไฟล์"}), 400
    
    if not allowed_file(file.filename):
        return jsonify({"Error Text": "ไฟล์ผิดประเภท กรุณาอัปโหลด PNG, JPG หรือ JPEG เท่านั้น"}), 400
    
    try:
        image = Image.open(io.BytesIO(file.read())).convert("RGB")
        
        # Extract GPS coordinates from image
        gps_location = extract_gps(image)
        
        # Preprocess image
        input_tensor = transform(image).unsqueeze(0).to(device)
        
        # Predict
        with torch.no_grad():
            det_out = model(input_tensor)
        
        pred = det_out[0]  # [A, H, W, 5+C]
        pred = pred.reshape(-1, pred.shape[-1])
        obj_probs = torch.sigmoid(pred[..., 4])
        threshold = 0.1
        mask = obj_probs > threshold
        
        if not mask.any():
            detections = [{"class": "ไม่เจอแมว", "conf": 0.0}]
        else:
            valid_preds = pred[mask]
            valid_obj_probs = obj_probs[mask]
            
            # Extract boxes
            boxes = []
            for p in valid_preds:
                cx, cy, bw, bh = p[:4]
                x1 = (cx - bw/2) * 224
                y1 = (cy - bh/2) * 224
                x2 = (cx + bw/2) * 224
                y2 = (cy + bh/2) * 224
                boxes.append(torch.stack([x1, y1, x2, y2]))
            
            boxes = torch.stack(boxes).to(device)
            keep = nms(boxes, valid_obj_probs, iou_threshold=0.3)
            
            detections = []
            bboxes = []
            for idx in keep:
                p = valid_preds[idx]
                class_probs = torch.softmax(p[5:], dim=0)
                cls_id = torch.argmax(class_probs)
                cls_conf = class_probs[cls_id]
                
                # Get bounding box coordinates
                cx, cy, bw, bh = p[:4]
                x1 = float((cx - bw/2) * 224)
                y1 = float((cy - bh/2) * 224)
                x2 = float((cx + bw/2) * 224)
                y2 = float((cy + bh/2) * 224)
                
                # Get cat info from database mapping
                cat_info = CAT_CLASSES.get(cls_id.item(), {"id": "C0005", "name": "Other"})
                
                detections.append({
                    "class": cat_info["name"],
                    "cat_id": cat_info["id"],
                    "conf": float(cls_conf.item()),
                    "bbox": [x1, y1, x2, y2]
                })
                
                bboxes.append({
                    "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                    "label": cat_info["name"],
                    "confidence": float(cls_conf.item())
                })
        
        # Convert original image to base64 with bounding boxes drawn
        img_with_boxes = _draw_bounding_boxes_on_image(image, bboxes if 'bboxes' in locals() else [])
        buffered = io.BytesIO()
        img_with_boxes.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
        
        response_data = {
            "detections": detections,
            "bboxes": bboxes if 'bboxes' in locals() else [],
            "imagedetect": img_str
        }
        
        # Add GPS location if available
        if gps_location:
            response_data["location"] = gps_location
            
        return jsonify(response_data)
    
    except Exception as e:
        return jsonify({"Error Text": f"เกิดข้อผิดพลาดในการประมวลผล: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)