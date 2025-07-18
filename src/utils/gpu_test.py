"""
GPU Detection Test Script
Run this to diagnose CUDA/GPU issues
"""

import torch
import sys

print("=== GPU Detection Test ===")
print(f"Python version: {sys.version}")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"cuDNN version: {torch.backends.cudnn.version()}")
    print(f"Number of GPUs: {torch.cuda.device_count()}")
    
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"  Memory: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.1f} GB")
        print(f"  Compute Capability: {torch.cuda.get_device_properties(i).major}.{torch.cuda.get_device_properties(i).minor}")
    
    # Test GPU functionality
    try:
        x = torch.randn(1000, 1000).cuda()
        y = torch.randn(1000, 1000).cuda()
        z = torch.matmul(x, y)
        print("✓ GPU tensor operations work correctly")
    except Exception as e:
        print(f"✗ GPU tensor operations failed: {e}")
        
else:
    print("\n=== Troubleshooting CUDA Issues ===")
    print("Possible reasons CUDA is not available:")
    print("1. No NVIDIA GPU installed")
    print("2. GPU drivers not installed or outdated")
    print("3. PyTorch was installed without CUDA support")
    print("4. CUDA toolkit version mismatch")
    
    print("\n=== Installation Check ===")
    print("To install PyTorch with CUDA support, use:")
    print("pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
    print("(Replace cu121 with your CUDA version)")
    
    print("\nTo check your NVIDIA driver version, run:")
    print("nvidia-smi")

print("\n=== Mixed Precision Support ===")
if torch.cuda.is_available():
    if torch.cuda.get_device_capability()[0] >= 7:  # Tensor cores available on RTX 20xx, 30xx, 40xx series
        print("✓ Your GPU supports Tensor Cores (mixed precision will be very fast)")
    else:
        print("○ Your GPU doesn't have Tensor Cores (mixed precision still works but less benefit)")
else:
    print("○ Mixed precision requires CUDA")
