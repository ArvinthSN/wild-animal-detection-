from ultralytics import YOLO
import torch
import os

# Ensure GPU is used if available, otherwise CPU
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")
print(f"CUDA available: {torch.cuda.is_available()}")

def continue_training():
    print("=" * 60)
    print("🔄 CONTINUING YOLOv8m TRAINING")
    print("=" * 60)
    
    # Find the latest checkpoint automatically
    import glob
    
    # Search for all last.pt files in runs directory (recursive)
    checkpoint_files = []
    for root, dirs, files in os.walk('runs'):
        for file in files:
            if file == 'last.pt':
                full_path = os.path.join(root, file)
                checkpoint_files.append(full_path)
    
    checkpoint_path = None
    if checkpoint_files:
        # Sort by modification time (newest first)
        checkpoint_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
        checkpoint_path = checkpoint_files[0]
        
        print(f"✅ Found latest checkpoint: {checkpoint_path}")
        print(f"   Last modified: {os.path.getmtime(checkpoint_path)}")
        
        if len(checkpoint_files) > 1:
            print(f"   (Found {len(checkpoint_files)} checkpoints total, using newest)")
            print(f"   Second newest: {checkpoint_files[1]}")
    else:
        print("❌ No 'last.pt' files found in 'runs' directory.")
    
    if checkpoint_path:
        # Load the checkpoint
        print(f"Loading model from checkpoint: {checkpoint_path}")
        model = YOLO(checkpoint_path)
        
        print(f"Resuming training from checkpoint on {device}...")
        try:
            results = model.train(resume=True, device=device)
            
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
            batch=4 if device == 'cpu' else 8,
            workers=2,
            imgsz=640 if device == 'cpu' else 800,
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
            name='animal_intrusion_fresh',
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
    continue_training()
