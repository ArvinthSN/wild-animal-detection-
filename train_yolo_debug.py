import sys
import traceback

try:
    from ultralytics import YOLO
    import torch
    import os
    
    # Print system information
    print("=" * 60)
    print("SYSTEM INFORMATION")
    print("=" * 60)
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    print("=" * 60)
    
    # Get absolute path to data.yaml
    data_yaml_path = os.path.abspath(r'dataset\data.yaml')
    print(f"\nData YAML path: {data_yaml_path}")
    print(f"Data YAML exists: {os.path.exists(data_yaml_path)}")
    
    # Check for existing checkpoint
    checkpoint_path = r'runs\detect\animal_intrusion_v1\weights\last.pt'
    checkpoint_exists = os.path.exists(checkpoint_path)
    print(f"\nCheckpoint path: {os.path.abspath(checkpoint_path)}")
    print(f"Checkpoint exists: {checkpoint_exists}")
    
    print("\n" + "=" * 60)
    print("STARTING TRAINING")
    print("=" * 60)
    
    # Load model
    if checkpoint_exists:
        print(f"Loading from checkpoint: {checkpoint_path}")
        model = YOLO(checkpoint_path)
    else:
        print("Loading pretrained YOLOv8m model")
        model = YOLO('yolov8m.pt')
    
    # Training configuration
    train_config = {
        'data': data_yaml_path,
        'epochs': 100,
        'patience': 15,
        'batch': 4 if device == 'cpu' else 8,
        'workers': 2,
        'imgsz': 640 if device == 'cpu' else 800,
        'device': device,
        'optimizer': 'AdamW',
        'lr0': 0.0005,
        'weight_decay': 0.0005,
        'cos_lr': True,
        'val': True,
        'augment': True,
        'hsv_h': 0.02,
        'hsv_s': 0.7,
        'hsv_v': 0.4,
        'degrees': 15.0,
        'translate': 0.2,
        'scale': 0.6,
        'shear': 2.5,
        'perspective': 0.0005,
        'flipud': 0.0,
        'fliplr': 0.5,
        'mosaic': 1.0,
        'mixup': 0.15,
        'project': 'runs/detect',
        'name': 'animal_intrusion_continued',
        'exist_ok': True,
        'resume': False
    }
    
    print("\nTraining configuration:")
    for key, value in train_config.items():
        print(f"  {key}: {value}")
    
    print("\n" + "=" * 60)
    print("TRAINING STARTED - This will take a while...")
    print("=" * 60 + "\n")
    
    sys.stdout.flush()
    
    # Start training
    results = model.train(**train_config)
    
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
        
except Exception as e:
    print(f"\n❌ TRAINING FAILED!")
    print(f"Error type: {type(e).__name__}")
    print(f"Error message: {str(e)}")
    print("\nFull traceback:")
    traceback.print_exc()
    sys.exit(1)
