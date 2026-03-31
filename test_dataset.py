from ultralytics import YOLO
import torch

# Test dataset loading
device = '0' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# Load model
print("Loading YOLOv8m...")
model = YOLO('yolov8m.pt')

# Try to validate on our dataset
print("Testing dataset loading...")
try:
    results = model.val(
        data='dataset/data.yaml',
        batch=1,
        imgsz=640,
        device=device,
        verbose=True
    )
    print(f"✅ Dataset loaded successfully!")
    print(f"mAP@0.5: {results.box.map50:.4f}")
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
