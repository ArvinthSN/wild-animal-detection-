
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
import io
from PIL import Image
import numpy as np
import cv2
import base64

app = FastAPI()

# Allow CORS for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load YOLO model
# Use best model if available, otherwise yolov8n
MODEL_PATH = "runs/detect/animal_intrusion_v1/weights/best.pt"
try:
    model = YOLO(MODEL_PATH)
    print(f"Loaded custom model from {MODEL_PATH}")
except Exception:
    print(f"Custom model not found at {MODEL_PATH}, using yolov8n.pt")
    model = YOLO("yolov8n.pt")

@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    
    # Run inference
    results = model(image)
    
    detections = []
    annotated_frame = results[0].plot() # Get image with bounding boxes
    
    # Convert annotated image to base64 for frontend display
    _, buffer = cv2.imencode('.jpg', annotated_frame)
    img_base64 = base64.b64encode(buffer).decode('utf-8')

    for result in results:
        boxes = result.boxes
        for box in boxes:
            cls_id = int(box.cls[0])
            label = model.names[cls_id]
            conf = float(box.conf[0])
            bbox = box.xyxy[0].tolist()  # [x1, y1, x2, y2]
            
            detections.append({
                "label": label,
                "confidence": conf,
                "bbox": bbox
            })
            
    return {"detections": detections, "image": img_base64}

# Mount the frontend directory to serve static files (if any)
app.mount("/static", StaticFiles(directory="front end"), name="static")

@app.get("/", response_class=HTMLResponse)
async def read_root():
    with open("front end/index.html", "r", encoding="utf-8") as f:
        return f.read()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
