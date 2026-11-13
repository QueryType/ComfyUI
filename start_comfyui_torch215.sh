export PYTORCH_ENABLE_MPS_FALLBACK=1
# Same flags as start_comfyui.sh, but runs in the comfyui-torch215 conda env (torch 2.15).
# Text encoders are stored in fp8 by default (about half the memory). Pass "full" first to keep them at full precision.
# Diffusion model fp8: set weight_dtype = fp8_e4m3fn on the Load Diffusion Model node.
# Default port 8189 so the browser tab shows which env you are on. Run only one env at a time when benchmarking.
# Usage: ./start_comfyui_torch215.sh [full] [host] [port] [extra comfyui flags...]
TE_DTYPE="--fp8_e4m3fn-text-enc"
if [ "$1" = "full" ]; then
  TE_DTYPE=""
  shift
fi
HOST="${1:-0.0.0.0}"
PORT="${2:-8189}"
shift 2 2>/dev/null || shift $#
cd "$(dirname "$0")"
/Volumes/d/conda/conda_envs/comfyui-torch215/bin/python main.py --listen "$HOST" --port "$PORT" \
  --use-pytorch-cross-attention \
  --highvram \
  --reserve-vram 10 \
  --bf16-vae \
  --fast fp16_accumulation autotune \
  $TE_DTYPE \
  "$@"
