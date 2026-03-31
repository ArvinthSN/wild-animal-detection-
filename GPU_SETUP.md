# GPU Setup Instructions

## Current Issue
Your PyTorch installation (2.9.1+cpu) is CPU-only and doesn't support CUDA/GPU.

## Steps to Enable GPU Training:

### 1. Check Your GPU
First, verify you have an NVIDIA GPU:
```powershell
nvidia-smi
```

If this command works, you have an NVIDIA GPU and can proceed.

### 2. Uninstall Current PyTorch
```powershell
pip uninstall torch torchvision torchaudio
```

### 3. Install PyTorch with CUDA Support

For CUDA 12.1 (recommended for most modern GPUs):
```powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

For CUDA 11.8 (if you have older drivers):
```powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 4. Verify GPU is Detected
```powershell
python check_gpu.py
```

### 5. Resume Training
Once GPU is detected, run:
```powershell
python train_yolo_debug.py
```

## Important Notes:
- Training on GPU will be 10-50x faster than CPU
- Make sure your NVIDIA drivers are up to date
- If nvidia-smi doesn't work, install/update NVIDIA drivers first
- Visit https://pytorch.org/get-started/locally/ for more options
