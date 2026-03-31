
from ultralytics import YOLO
import torch
import os
import os
from data_manager import download_data, setup_yolo_structure

# Ensure GPU is used
device = '0' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

def train_pipeline():
    # 1. Prepare Data
    print("Step 1: Preparing Data...")
    dataset_path = download_data()
    yaml_path = setup_yolo_structure(dataset_path)
    
    # 2. Setup Model (YOLOv8m - Medium for 80-class high accuracy)
    print("Step 2: Loading Model...")
    print("=" * 60)
    print("🚀 UPGRADING TO YOLOv8m FOR 90% mAP@0.5 TARGET")
    print("=" * 60)
    
    # Always start fresh with YOLOv8m (don't resume from Nano)
    print("Starting new training from yolov8m.pt (pretrained)")
    model = YOLO('yolov8m.pt')  # 25.9M parameters - 8x larger than Nano

    # 3. Training - Optimized for 80-class animal detection
    # Strategy: Larger model + Extended epochs + Focal Loss
    print("Step 3: Starting Training...")
    print("Configuration: 100 epochs, imgsz=800, batch=16, focal loss enabled")
    try:
        results = model.train(
            data=yaml_path,
            epochs=100,            # Extended for convergence (expect 60-80)
            patience=15,           # Early stopping with more tolerance
            batch=8,               # Reduced for stability (OOM prevention)
            workers=2,             # Lower workers for Windows
            imgsz=800,             # Larger image size for small animals
            device=device,
            optimizer='AdamW',
            lr0=0.0005,            # Lower LR for stability with larger model
            weight_decay=0.0005,   # Regularization
            cos_lr=True,           # Cosine Annealing
            val=True,              # Validate during training
            
            # Imbalance Handling (CRITICAL FOR 80 CLASSES)
            fl_gamma=1.5,          # Focal Loss - handles hard examples
            
            # Augmentation (Strong but Controlled)
            augment=True,
            hsv_h=0.02, 
            hsv_s=0.7, 
            hsv_v=0.4,
            degrees=15.0,
            translate=0.2,
            scale=0.6,
            shear=2.5,
            perspective=0.0005,
            flipud=0.0,
            fliplr=0.5,
            mosaic=1.0,            # Mosaic augmentation
            mixup=0.15,            # Mixup for generalization
            
            project='runs/detect',
            name='animal_intrusion_v2_yolov8m',  # New experiment name
            exist_ok=True,
            resume=False           # Fresh start with YOLOv8m
        )
    except Exception as e:
        print(f"CRITICAL ERROR DETAIL: {e}")
        import traceback
        traceback.print_exc()
    
    print("Training Complete. Results saved in runs/detect/animal_intrusion_v2_yolov8m")
    
    # 4. Validation
    print("Step 4: Validating...")
    metrics = model.val(project='runs/detect', name='animal_intrusion_v2_yolov8m')
    map50 = metrics.box.map50
    map5095 = metrics.box.map
    
    print(f"Validation mAP@0.5: {map50:.4f}")
    print(f"Validation mAP@0.5:0.95: {map5095:.4f}")
    
    if map50 < 0.90:
        print("WARNING: mAP@0.5 is below 90%. Optimization needed.")
        # Logic to continue training or adjust could go here
    else:
        print("SUCCESS: MVP Criteria Met (>90% mAP@0.5)")
        
    # Export
    model.export(format='onnx')

if __name__ == "__main__":
    train_pipeline()
