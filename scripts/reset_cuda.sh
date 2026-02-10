#!/bin/bash
# Reset NVIDIA CUDA driver without rebooting

echo "=================================="
echo "Resetting NVIDIA CUDA Driver"
echo "=================================="
echo ""

echo "Step 1: Unloading NVIDIA kernel modules..."
sudo rmmod nvidia_uvm
sudo rmmod nvidia_drm
sudo rmmod nvidia_modeset
sudo rmmod nvidia

echo ""
echo "Step 2: Reloading NVIDIA kernel modules..."
sudo modprobe nvidia
sudo modprobe nvidia_modeset
sudo modprobe nvidia_drm
sudo modprobe nvidia_uvm

echo ""
echo "Step 3: Verifying GPU is available..."
nvidia-smi

echo ""
echo "Step 4: Testing CUDA with PyTorch..."
cd /home/pablo/code/odin-slm
source .venv/bin/activate
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"

echo ""
echo "=================================="
echo "✓ CUDA Reset Complete!"
echo "=================================="
echo ""
echo "Now you can launch training:"
echo "  uv run python scripts/train_medical_ner.py --config configs/medical_ner_re_config.yaml"
