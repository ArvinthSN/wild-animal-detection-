from ultralytics import YOLO
import torch
import os

# Ensure GPU is used
device = '0' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

def resume_training():
    print("=" * 60)
    print("🔄 RESUMING YOLOv8m TRAINING FROM CHECKPOINT")
    print("=" * 60)
    
    # Check for checkpoint - try multiple possible locations
    checkpoint_paths = [
        r'runs\detect\animal_intrusion_v2_yolov8m\weights\last.pt',
        r'runs\detect\animal_intrusion_v1\weights\last.pt',
        r'runs\detect\animal_intrusion_v2_yolov8m2\weights\last.pt',
        r'runs\detect\animal_intrusion_v2_yolov8m3\weights\last.pt',
        r'runs\detect\animal_intrusion_v2_yolov8m4\weights\last.pt',
        r'runs\detect\animal_intrusion_v2_yolov8m5\weights\last.pt'
    ]
    
    checkpoint_path = None
    for path in checkpoint_paths:
        if os.path.exists(path):
            checkpoint_path = path
            print(f"✅ Found checkpoint: {checkpoint_path}")
            break
    
    if checkpoint_path:
        model = YOLO(checkpoint_path)
        
        # Resume training
        print("Resuming training with same configuration...")
        try:
            results = model.train(
                resume=True  # This will automatically resume from the checkpoint
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
                
            return map50, map5095
            
        except Exception as e:
            print(f"\n❌ TRAINING FAILED: {e}")
            import traceback
            traceback.print_exc()
            raise
    else:
        print(f"❌ No checkpoint found in any of the expected locations")
        print("Starting fresh training instead...")
        
        # Start fresh training with relative path
        data_yaml = r'dataset\data.yaml'
        model = YOLO('yolov8m.pt')
        
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
        
        # Validation
        metrics = model.val()
        map50 = metrics.box.map50
        map5095 = metrics.box.map
        
        print(f"\n📊 FINAL RESULTS:")
        print(f"  mAP@0.5: {map50:.4f} ({map50*100:.2f}%)")
        print(f"  mAP@0.5:0.95: {map5095:.4f} ({map5095*100:.2f}%)")
        
        return map50, map5095

if __name__ == "__main__":
    resume_training()
