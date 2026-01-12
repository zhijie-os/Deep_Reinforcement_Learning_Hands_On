#!/usr/bin/env python3
import torch

print("=== CUDA Test ===")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA device count: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"  Device {i}: {torch.cuda.get_device_name(i)}")
    
    # 测试简单的CUDA操作
    print("\nTesting CUDA operations...")
    x = torch.randn(3, 3).cuda()
    y = torch.randn(3, 3).cuda()
    z = x @ y  # 矩阵乘法
    print(f"  CUDA matrix multiplication successful: {z.shape}")
    print(f"  CUDA memory allocated: {torch.cuda.memory_allocated()/1024**2:.2f} MB")
    print(f"  CUDA memory cached: {torch.cuda.memory_reserved()/1024**2:.2f} MB")
else:
    print("CUDA not available. Check your installation:")
    print("1. NVIDIA drivers installed?")
    print("2. CUDA toolkit installed?")
    print("3. PyTorch installed with CUDA support?")
    print("\nRun: pip list | grep torch")
    print("Should show something like: torch ...+cu11x")