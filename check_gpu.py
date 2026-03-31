import torch

print("=" * 60)
print("GPU DIAGNOSTICS")
print("=" * 60)

print(f"\nPyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU count: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"  Memory: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.2f} GB")
else:
    print("\n⚠️  CUDA is NOT available!")
    print("\nPossible reasons:")
    print("1. PyTorch was installed without CUDA support (CPU-only version)")
    print("2. NVIDIA GPU drivers are not installed or outdated")
    print("3. CUDA toolkit is not installed")
    print("4. GPU is not NVIDIA (AMD/Intel GPUs need different setup)")
    
    print("\n" + "=" * 60)
    print("SOLUTION:")
    print("=" * 60)
    print("\nTo enable GPU training, you need to:")
    print("1. Check if you have an NVIDIA GPU")
    print("2. Install NVIDIA GPU drivers")
    print("3. Reinstall PyTorch with CUDA support:")
    print("   pip uninstall torch torchvision")
    print("   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121")
    print("\nOr visit: https://pytorch.org/get-started/locally/")
