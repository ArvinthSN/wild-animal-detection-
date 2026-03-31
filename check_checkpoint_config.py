import torch

checkpoint_path = r'runs\detect\animal_intrusion_v1\weights\last.pt'
try:
    ckpt = torch.load(checkpoint_path)
    train_args = ckpt.get('train_args', {})
    
    print(f"Batch from checkpoint: {train_args.get('batch')}")
    print(f"ImgSz from checkpoint: {train_args.get('imgsz')}")
    print(f"Total Epochs from checkpoint: {train_args.get('epochs')}")
    print(f"Current Epoch from checkpoint: {ckpt.get('epoch')}")
except Exception as e:
    print(e)
