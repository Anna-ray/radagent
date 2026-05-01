# RadAgent — Specialist CV Module

This subdirectory contains the **specialist CV head** of the RadAgent system.
Its job is to catch findings the Vision LLM misses (small nodules, subtle
infiltrates) and to provide grounded visual heatmaps for every finding.

## Quick start (RTX 4070 Ti SUPER, Windows / Linux)

```bash
# 1. Create env
conda create -n radagent python=3.11 -y
conda activate radagent

# 2. Install PyTorch (CUDA 12.4 wheels)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# 3. Rest of deps
pip install -r requirements.txt

# 4. Edit configs/nih14_convnextv2_base.yaml — set data paths
#    images_dir, labels_csv, train_split_txt, test_split_txt

# 5. Train
python -m scripts.train --config configs/nih14_convnextv2_base.yaml
```

Expected wall-clock on a 4070 Ti SUPER: **~75–95 minutes per epoch** at
batch 16 / 384px. With early stopping (patience 6) the typical run
converges in 10–14 epochs → ~14–20 hours total.

If VRAM is tight: drop `image_size` to 320 or `batch_size` to 12.

## MI300X (ROCm) port

```bash
# Install ROCm-built PyTorch
pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm6.2
pip install -r requirements.txt
```

The training code is **device-agnostic** — `torch.device("cuda")` resolves
to the HIP device on ROCm. Set `amp_dtype: "bf16"` (already default) — the
MI300X has full bf16 support. Expect **3–5× faster wall-clock** thanks to
the larger effective batch (you can raise batch to 48 with the 192GB HBM).

## Architecture summary

| Component | Choice | Rationale |
|---|---|---|
| Backbone | `convnextv2_base.fcmae_ft_in22k_in1k_384` | FCMAE pretraining transfers well to medical |
| Resolution | 384×384 | Small-finding sensitivity |
| Loss | Asymmetric Loss (γ⁻=4, γ⁺=1, clip=0.05) | SOTA on multi-label medical |
| Optimizer | AdamW, discriminative LR (head=5e-4, backbone=5e-5) | Stable fine-tuning |
| Schedule | Linear warmup → cosine | Standard for ConvNeXt |
| AMP | bf16 + grad checkpointing | Fits 384px on 16GB |
| EMA | decay 0.9999 | +0.3-0.7% AUC for free |
| Sampling | Weighted (rarest-positive) | Combats Hernia 0.2% imbalance |
| Validation | TTA (hflip) + per-class threshold search | Reportable F1 |
| Calibration | Temperature scaling | Downstream LLM consumes confidences |

## Outputs

After training, `runs/<exp_name>/` contains:
- `best.pt` — checkpoint at best val mean AUC
- `last.pt` — final checkpoint
- `calibration.json` — temperature + per-class F1-optimal thresholds
- `tb/` — TensorBoard logs (per-class AUC and F1 curves)

## Next modules (not in this drop)

- `radagent/inference/findings.py` — converts logits → structured clinical text
- `radagent/rag/` — BGE-M3 + FAISS over Radiopaedia + PubMed
- `radagent/agent/` — LangGraph planner with tool calls
- `radagent/serve/` — FastAPI + WebSocket dashboard
