#!/usr/bin/env bash
# infra/mi300x_bootstrap.sh
# -------------------------
# Bootstrap a fresh DigitalOcean MI300X droplet for RadAgent VLM serving.
# Idempotent: safe to re-run.
#
# Usage on droplet (after first SSH):
#   bash mi300x_bootstrap.sh
#
# Pre-reqs the droplet should already have (DO image provides):
#   - Ubuntu 22.04 LTS
#   - ROCm 6.x with rocm-smi, AMDGPU drivers
#   - Python 3.10+
#
# What we add:
#   - System deps for vLLM-ROCm
#   - Python venv at /workspace/venv
#   - vLLM-ROCm + dependencies
#   - HF token from env, cache config

set -euo pipefail

WORKDIR=/workspace
HF_HOME_DIR=/workspace/hf_cache
PROJECT_DIR=/workspace/radagent

echo "==[bootstrap] start at $(date) =="

# ---------- ROCm sanity ----------
echo "==[bootstrap] ROCm sanity =="
if ! command -v rocm-smi >/dev/null; then
    echo "ERROR: rocm-smi not found. Are you on an AMD GPU droplet?"
    exit 1
fi
rocm-smi --showproductname --showmeminfo=vram | head -40

# ---------- system deps ----------
echo "==[bootstrap] apt update =="
sudo apt-get update -y
sudo apt-get install -y --no-install-recommends \
    git build-essential python3-venv python3-pip \
    libnuma-dev libsqlite3-dev rsync curl wget jq tmux htop

# ---------- workdir ----------
echo "==[bootstrap] workdir $WORKDIR =="
sudo mkdir -p "$WORKDIR"
sudo chown -R "$(id -u)":"$(id -g)" "$WORKDIR"
mkdir -p "$HF_HOME_DIR" "$PROJECT_DIR"

# ---------- python venv ----------
echo "==[bootstrap] python venv =="
if [[ ! -d "$WORKDIR/venv" ]]; then
    python3 -m venv "$WORKDIR/venv"
fi
# shellcheck disable=SC1091
source "$WORKDIR/venv/bin/activate"
pip install --upgrade pip wheel setuptools

# ---------- vLLM-ROCm ----------
# Install path varies by ROCm version. We use the published wheel for ROCm 6.x.
# If this fails, fall back to building from source per the ROCm/vllm README.
echo "==[bootstrap] installing vLLM-ROCm =="
pip install --pre vllm --index-url https://download.pytorch.org/whl/rocm6.2 || {
    echo "[bootstrap] WARN: pre wheel failed, trying public PyPI"
    pip install vllm
}

# Sanity import
python3 -c "import vllm; print(f'vllm {vllm.__version__} loaded')"

# ---------- HF cache + token ----------
echo "==[bootstrap] HF config =="
if [[ -z "${HF_TOKEN:-}" ]]; then
    echo "WARN: HF_TOKEN not set. Set it before running run_vllm.sh."
fi
mkdir -p "$HF_HOME_DIR"

cat > "$WORKDIR/env.sh" <<EOF
# Source this before run_vllm.sh
export HF_HOME=$HF_HOME_DIR
export HF_HUB_ENABLE_HF_TRANSFER=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
# HF_TOKEN must be exported by the user (DO NOT bake it into a file).
EOF
echo "[bootstrap] wrote $WORKDIR/env.sh"

# ---------- hf_transfer for fast downloads ----------
pip install hf_transfer

# ---------- final report ----------
echo
echo "==[bootstrap] DONE =="
echo "  venv:        $WORKDIR/venv"
echo "  hf cache:    $HF_HOME_DIR"
echo "  project dir: $PROJECT_DIR"
echo
echo "Next steps:"
echo "  1. export HF_TOKEN=hf_..."
echo "  2. source $WORKDIR/env.sh"
echo "  3. bash run_vllm.sh                  # default: 11B"
echo "     MODEL_SIZE=90b bash run_vllm.sh   # heavier: 90B"