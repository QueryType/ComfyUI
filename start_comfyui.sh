#python main.py --force-upcast-attention  --force-fp16 --listen 10.0.0.4 --port 8188
#export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0
export PYTORCH_ENABLE_MPS_FALLBACK=1
#python main.py --use-split-cross-attention  --lowvram  --force-fp16 --listen 10.0.0.4 --port 8188
# M4 Pro 64GB unified memory optimizations:
#   --use-pytorch-cross-attention : Metal-optimized SDPA kernels, faster than sub-quadratic on MPS
#   --highvram                    : keep models resident between runs (64GB has headroom)
#   --reserve-vram 10             : keep 10GB headroom so the text encoder fully unloads before the diffusion model loads
#   --bf16-vae                    : explicit bf16 VAE, native on M4
#   --fast fp16_accumulation autotune : fp16 attention accumulation + MPS kernel autotuning
# Usage: ./start_comfyui.sh [host] [port]
HOST="${1:-0.0.0.0}"
PORT="${2:-8188}"
python main.py --listen "$HOST" --port "$PORT" \
  --use-pytorch-cross-attention \
  --highvram \
  --reserve-vram 10 \
  --bf16-vae \
  --fast fp16_accumulation autotune

