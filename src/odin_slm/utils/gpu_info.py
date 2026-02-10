"""GPU information and optimization utilities"""

import torch
import subprocess


def get_gpu_info():
    """Get detailed GPU information"""
    if not torch.cuda.is_available():
        return {"available": False, "message": "CUDA not available"}

    gpu_info = {
        "available": True,
        "device_count": torch.cuda.device_count(),
        "current_device": torch.cuda.current_device(),
        "device_name": torch.cuda.get_device_name(0),
        "total_memory_gb": torch.cuda.get_device_properties(0).total_memory / 1e9,
        "cuda_version": torch.version.cuda,
        "pytorch_version": torch.__version__,
    }

    return gpu_info


def print_gpu_info():
    """Print GPU information in a formatted way"""
    info = get_gpu_info()

    if not info["available"]:
        print(info["message"])
        return

    print("=" * 50)
    print("GPU Information")
    print("=" * 50)
    print(f"Device: {info['device_name']}")
    print(f"Total Memory: {info['total_memory_gb']:.2f} GB")
    print(f"CUDA Version: {info['cuda_version']}")
    print(f"PyTorch Version: {info['pytorch_version']}")
    print(f"Number of GPUs: {info['device_count']}")
    print("=" * 50)


def get_optimal_batch_size(model_size_gb: float, seq_length: int = 2048) -> int:
    """
    Estimate optimal batch size based on available GPU memory

    Args:
        model_size_gb: Approximate model size in GB
        seq_length: Sequence length for training

    Returns:
        Suggested batch size
    """
    if not torch.cuda.is_available():
        return 1

    total_memory = torch.cuda.get_device_properties(0).total_memory / 1e9

    # Reserve ~2GB for CUDA overhead and other processes
    available_memory = total_memory - 2.0

    # Rough estimation: activation memory scales with sequence length
    # For 16GB GPU with 4-bit quantization
    if model_size_gb <= 1.0 and seq_length <= 2048:
        return 4
    elif model_size_gb <= 2.0 and seq_length <= 2048:
        return 2
    else:
        return 1


if __name__ == "__main__":
    print_gpu_info()
    print(f"\nRecommended batch size for 1B model: {get_optimal_batch_size(1.0)}")
