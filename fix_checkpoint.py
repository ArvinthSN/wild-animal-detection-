import torch
import argparse
import os

checkpoint_paths = [
    r'runs\detect\animal_intrusion_v1\weights\last.pt',
    r'runs\detect\animal_intrusion_v1\weights\best.pt'
]

for checkpoint_path in checkpoint_paths:
    print(f"\nProcessing checkpoint: {checkpoint_path}")
    if not os.path.exists(checkpoint_path):
        print(f"Skipping {checkpoint_path} (not found)")
        continue

    try:
        ckpt = torch.load(checkpoint_path)
        
        # Check what kind of object 'train_args' is
        if 'train_args' in ckpt:
            args = ckpt['train_args']
            print(f"train_args type: {type(args)}")
            
            updated = False
            if isinstance(args, dict):
                if 'data' in args:
                    print(f"Current data path: {args['data']}")
                    args['data'] = r'd:\naveen project\naveen project\naveen project\dataset\data.yaml'
                    print(f"Updated data path to: {args['data']}")
                    updated = True
                if 'project' in args:
                    args['project'] = 'runs/detect'
                    updated = True
                if 'save_dir' in args:
                    args['save_dir'] = r'd:\naveen project\naveen project\naveen project\runs\detect\animal_intrusion_v1'
                    updated = True
                
                # Extend epochs and match imgsz
                print(f"Current epochs: {args.get('epochs')}")
                if args.get('epochs') != 100:
                    args['epochs'] = 100
                    print(f"Updated epochs to: 100")
                    updated = True
                
                print(f"Current imgsz: {args.get('imgsz')}")
                if args.get('imgsz') != 640:
                    args['imgsz'] = 640
                    print(f"Updated imgsz to: 640")
                    updated = True
            else:
                # Assuming Namespace
                if hasattr(args, 'data'):
                    print(f"Current data path: {args.data}")
                    args.data = r'd:\naveen project\naveen project\naveen project\dataset\data.yaml'
                    print(f"Updated data path to: {args.data}")
                    updated = True
                if hasattr(args, 'project'):
                    args.project = 'runs/detect'
                    updated = True
                if hasattr(args, 'save_dir'):
                    args.save_dir = r'd:\naveen project\naveen project\naveen project\runs\detect\animal_intrusion_v1'
                    updated = True
            
            if updated:
                # Save back
                torch.save(ckpt, checkpoint_path)
                print(f"✅ {checkpoint_path} updated successfully.")
            else:
                 print(f"ℹ️ {checkpoint_path} no changes needed.")
            
            # Check epoch
            current_epoch = ckpt.get('epoch')
            print(f"Current logged epoch: {current_epoch}")
            if current_epoch == -1:
                ckpt['epoch'] = 49
                print("Updated epoch from -1 to 49 (assuming 50 epochs completed)")
                torch.save(ckpt, checkpoint_path)


    except Exception as e:
        print(f"❌ Error updating {checkpoint_path}: {e}")
        import traceback
        traceback.print_exc()
