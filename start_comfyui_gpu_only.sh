#!/bin/bash
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0
export PYTORCH_ENABLE_MPS_FALLBACK=1
# M4 Pro 64GB unified memory optimizations:
#   --use-pytorch-cross-attention : Metal-optimized SDPA kernels, faster than sub-quadratic on MPS
#   --gpu-only                    : forces text encoder/VAE/intermediates onto MPS instead of CPU
#                                    (replaces --highvram, which is a no-op on MPS: vram_state
#                                    always gets forced to SHARED regardless of that flag)
#   --bf16-vae                    : explicit bf16 VAE, native on M4
#   --fast fp16_accumulation autotune : fp16 attention accumulation + MPS kernel autotuning
source /Volumes/d/conda/conda_install/miniconda3/etc/profile.d/conda.sh
conda activate comfyui-torch215 || exit 1
cd "$(dirname "$0")"
python main.py --listen 192.168.1.2 --port 8188 \
  --use-pytorch-cross-attention \
  --gpu-only \
  --bf16-vae \
  --fast fp16_accumulation autotune
