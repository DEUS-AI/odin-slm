#!/bin/bash
# Setup script for Odin SLM project

set -e

echo "========================================="
echo "Odin SLM - Project Setup"
echo "========================================="
echo ""

# Check if UV is installed
if ! command -v uv &> /dev/null; then
    echo "Error: UV is not installed. Install it first:"
    echo "curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

echo "✓ UV is installed ($(uv --version))"
echo ""

# Check if NVIDIA GPU is available
if ! command -v nvidia-smi &> /dev/null; then
    echo "Warning: nvidia-smi not found. GPU may not be available."
else
    echo "✓ NVIDIA GPU detected:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    echo ""
fi

# Check CUDA version
if command -v nvcc &> /dev/null; then
    echo "✓ CUDA compiler found:"
    nvcc --version | grep "release"
    echo ""
else
    echo "Warning: CUDA compiler (nvcc) not found in PATH"
    echo ""
fi

# Install dependencies
echo "Installing Python dependencies with UV..."
uv sync
echo "✓ Dependencies installed"
echo ""

# Create virtual environment activation notice
echo "========================================="
echo "Setup Complete!"
echo "========================================="
echo ""
echo "Next steps:"
echo "1. Activate virtual environment:"
echo "   source .venv/bin/activate"
echo ""
echo "2. Check GPU configuration:"
echo "   uv run python src/odin_slm/utils/gpu_info.py"
echo ""
echo "3. Review configuration:"
echo "   configs/training_config.yaml"
echo ""
echo "4. Read documentation:"
echo "   CLAUDE.md"
echo ""
echo "Happy training! 🚀"
