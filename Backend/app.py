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
from flask_cors import CORS
from Model import PretrainedCatDetector

app = Flask(__name__)

# Load custom model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_classes = 9
model = PretrainedCatDetector(num_classes=num_classes,num_anchors=5)
model.load_state_dict(torch.load('Backend\DenseNet121_CatV3_withCBAMv8.pth', map_location=device))
model.to(device)
model.eval()

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5121, 0.4615, 0.4059], std=[0.2247, 0.2278, 0.2331])
])
CORS(app, resources={
    r"/*": {
        "origins": ["http://localhost:3000", "http://10.54.55.59:3000"],  # อนุญาตแค่ Frontend
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type"]
    }
})
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/", methods=["GET"])
def mainRoute():
    return "Hello Flask API"

@app.route("/health", methods=["GET"])
def check_health():
    return jsonify({
        "code": 200
    })


@app.route("/predict", methods=["POST"])
def predict():
    file = request.files['file']
    if file and allowed_file(file.filename):
        image = Image.open(io.BytesIO(file.read())).convert("RGB")
        
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
            detections = [{"class": "ไม่พบวัตถุ", "conf": 0.0}]
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
            for idx in keep:
                p = valid_preds[idx]
                class_probs = torch.softmax(p[5:], dim=0)
                cls_id = torch.argmax(class_probs)
                cls_conf = class_probs[cls_id]
                
                detections.append({
                    "class": f"Class_{cls_id.item()}",
                    "conf": float(cls_conf.item())
                })
        
        # Convert original image to base64
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
        
        return jsonify({
            "detections": detections,
            "imagedetect": img_str
        })
    else:
        return jsonify({
            "Error Text": "ไฟล์ผิดประเภท"
        })

@app.errorhandler(404)
def not_found(e):
    return jsonify({
        "error": "ไม่รู้จัก Route ที่เรียกใช้ครับ",
        "code": 404
    }), 404

@app.errorhandler(405)
def method_not_allowed(e):
    return jsonify({
        "error": "Method ไม่ถูกต้องครับ",
        "code": 405
    }), 405

@app.errorhandler(500)
def server_error(e):
    return jsonify({
        "error": "Internal Server Error",
        "code": 500
    }), 500

@app.errorhandler(502)
def bad_gateway(e):
    return jsonify({
        "error": "Bad Gateway",
        "code": 502
    }), 502

@app.errorhandler(503)
def service_unavailable(e):
    return jsonify({
        "error": "Service Unavailable",
        "code": 503
    }), 503
    
    
@app.errorhandler(400)
def worng_pattern(e):
    return jsonify({
        "error": "ส่งข้อมูลไม่ถูก pattern",
        "code": 400        
    }), 400


@app.errorhandler(504)
def gateway_timeout(e):
    return jsonify({
        "error": "Gateway Timeout",
        "code": 504
    }), 504

if __name__ == '__main__':
    
    app.run(host='0.0.0.0', port=2569, debug=True)