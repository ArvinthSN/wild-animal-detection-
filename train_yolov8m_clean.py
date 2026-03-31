from ultralytics import YOLO
import torch
import os

# Ensure GPU is used
device = '0' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

def train_yolov8m():
    print("=" * 60)
    print("🚀 TRAINING YOLOv8m FOR 90% mAP@0.5 TARGET")
    print("=" * 60)
    
    # Verify dataset exists
    data_yaml = r'D:\naveen project\naveen project\naveen project\dataset'
    if not os.path.exists(data_yaml):
        raise FileNotFoundError(f"Dataset YAML not found: {data_yaml}")
    
    print(f"✅ Dataset YAML found: {data_yaml}")
    
    # Load YOLOv8m pretrained model
    print("Loading YOLOv8m pretrained weights...")
    model = YOLO('yolov8m.pt')
    
    # Training configuration optimized for 80-class animal detection
    print("Starting training with optimized configuration...")
    print("  - Epochs: 100 (patience=15)")
    print("  - Image size: 800")
    print("  - Batch: 8")
    print("  - Learning rate: 0.0005")
    print("  - Focal loss: enabled (fl_gamma=1.5)")
    
    try:
        results = model.train(
            data=data_yaml,
            epochs=100,
            patience=15,
            batch=8,
            workers=2,
            imgsz=800,
            device=device,
            optimizer='AdamW',
            lr0=0.0005,
            weight_decay=0.0005,
            cos_lr=True,
            val=True,
            
            # Strong augmentation
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
            mosaic=1.0,
            mixup=0.15,
            
            project='runs/detect',
            name='animal_intrusion_v2_yolov8m',
            exist_ok=True,
            resume=False
        )
        
        print("\n" + "=" * 60)
        print("✅ TRAINING COMPLETE!")
        print("=" * 60)
        
        # Validation
        print("\nRunning final validation...")
        metrics = model.val()
        map50 = metrics.box.map50
        map5095 = metrics.box.map
        
        print(f"\n📊 FINAL RESULTS:")
        print(f"  mAP@0.5: {map50:.4f} ({map50*100:.2f}%)")
        print(f"  mAP@0.5:0.95: {map5095:.4f} ({map5095*100:.2f}%)")
        
        if map50 >= 0.90:
            print(f"\n🎉 SUCCESS! Achieved ≥90% mAP@0.5 target!")
        else:
            print(f"\n⚠️  Target not met. Current: {map50*100:.2f}%, Target: 90%")
            
        # Export model
        print("\nExporting model to ONNX...")
        model.export(format='onnx')
        
        return map50, map5095
        
    except Exception as e:
        print(f"\n❌ TRAINING FAILED: {e}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    train_yolov8m()
