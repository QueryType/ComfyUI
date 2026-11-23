#!/bin/bash
export PYTORCH_ENABLE_MPS_FALLBACK=1
# Runs ComfyUI in the comfyui-torch215 conda env (torch 2.15).
source /Volumes/d/conda/conda_install/miniconda3/etc/profile.d/conda.sh
conda activate comfyui-torch215 || exit 1
# Text encoders are stored in fp8 by default (about half the memory). Pass "full" first to keep them at full precision.
# Diffusion model fp8: set weight_dtype = fp8_e4m3fn on the Load Diffusion Model node.
# Usage: ./start_comfyui.sh [full] [host] [port] [extra comfyui flags...]
TE_DTYPE="--fp8_e4m3fn-text-enc"
if [ "$1" = "full" ]; then
  TE_DTYPE=""
  shift
fi
HOST="${1:-0.0.0.0}"
PORT="${2:-8188}"
shift 2 2>/dev/null || shift $#
cd "$(dirname "$0")"
# The GPU wired memory limit resets to the macOS default on reboot; large models need 55 GB.
WIRED_MB=55000
if [ "$(sysctl -n iogpu.wired_limit_mb)" -lt "$WIRED_MB" ]; then
  echo "iogpu.wired_limit_mb is $(sysctl -n iogpu.wired_limit_mb), setting it to $WIRED_MB (needs sudo)"
  sudo sysctl iogpu.wired_limit_mb=$WIRED_MB || exit 1
fi
python main.py --listen "$HOST" --port "$PORT" \
  --use-pytorch-cross-attention \
  --highvram \
  --reserve-vram 10 \
  --bf16-vae \
  --fast fp16_accumulation autotune \
  $TE_DTYPE \
  "$@"
