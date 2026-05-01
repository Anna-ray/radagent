"""
scripts/predict_one.py
----------------------
Single-image end-to-end inference for the RadAgent specialist + RAG.

Pipeline:
  load image -> CLAHE -> eval transform -> forward + hflip TTA
  -> probabilities_to_findings -> (optional) RAG retrieval per
  above-threshold finding -> (optional) Grad-CAM heatmaps -> findings.json

Usage:
    python -m scripts.predict_one `
        --config configs/nih14_convnextv2_base.yaml `
        --image path/to/cxr.png `
        --checkpoint runs/nih14_convnextv2_base_384/best.pt `
        --calibration runs/nih14_convnextv2_base_384/calibration.json `
        --bands runs/nih14_convnextv2_base_384/calibration_bands.json `
        --rag-index data/rag/index.faiss `
        --rag-chunks data/rag/chunks.jsonl `
        --rag-manifest data/rag/manifest.json `
        --output-dir runs/nih14_convnextv2_base_384/predict_one `
        --gradcam
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import yaml
from torch.amp import autocast

from radagent.data.dataset import build_eval_transforms
from radagent.data.preprocessing import (
    apply_clahe,
    load_cxr_grayscale,
    to_three_channel,
)
from radagent.inference.findings import (
    load_calibration,
    probabilities_to_findings,
)
from radagent.inference.gradcam import GradCAMpp, overlay_heatmap
from radagent.models.specialist import SpecialistCXR


# ----------------------------- query templates -----------------------------
# Each finding gets a clinically informative query. The retriever embeds
# these the same way it embedded chunks (BGE-M3, normalized), so we want
# the query to read like a chest-imaging question, not a single keyword.
FINDING_QUERY_TEMPLATES: dict[str, str] = {
    "Atelectasis":
        "Atelectasis on chest radiograph: imaging features, causes, "
        "lobar collapse and differentials.",
    "Cardiomegaly":
        "Cardiomegaly on chest x-ray: cardiothoracic ratio, causes, "
        "heart failure association and differentials.",
    "Effusion":
        "Pleural effusion on chest radiograph: imaging features, "
        "blunting of costophrenic angle, transudate vs exudate causes.",
    "Infiltration":
        "Pulmonary infiltrate on chest radiograph: airspace and "
        "interstitial patterns, ground-glass opacification, differentials.",
    "Mass":
        "Pulmonary mass on chest radiograph: imaging features, "
        "lung cancer differentials and staging considerations.",
    "Nodule":
        "Solitary pulmonary nodule on chest radiograph: imaging "
        "features, malignancy risk factors and differential diagnosis.",
    "Pneumonia":
        "Pneumonia on chest radiograph: lobar consolidation, "
        "bronchopneumonia patterns and differentials.",
    "Pneumothorax":
        "Pneumothorax on chest radiograph: visceral pleural line, "
        "lung edge, tension pneumothorax and management urgency.",
    "Consolidation":
        "Pulmonary consolidation on chest radiograph: airspace "
        "opacification, air bronchograms, common causes.",
    "Edema":
        "Pulmonary edema on chest radiograph: Kerley lines, "
        "cardiogenic vs non-cardiogenic, bat-wing pattern.",
    "Emphysema":
        "Pulmonary emphysema on chest radiograph: hyperinflation, "
        "flattened diaphragms, COPD imaging features.",
    "Fibrosis":
        "Pulmonary fibrosis on chest radiograph: reticular opacities, "
        "honeycombing, idiopathic pulmonary fibrosis features.",
    "Pleural_Thickening":
        "Pleural thickening on chest radiograph: imaging features, "
        "asbestos-related pleural disease and differentials.",
    "Hernia":
        "Hiatal hernia on chest radiograph: retrocardiac air-fluid "
        "level, diaphragmatic hernia differentials.",
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--image", type=str, required=True)
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--calibration", type=str, required=True)
    p.add_argument("--bands", type=str, default=None)
    # RAG (all three required together; if any missing, RAG is disabled)
    p.add_argument("--rag-index", type=str, default=None)
    p.add_argument("--rag-chunks", type=str, default=None)
    p.add_argument("--rag-manifest", type=str, default=None)
    p.add_argument("--rag-k", type=int, default=3)
    p.add_argument("--no-rag", action="store_true")
    p.add_argument("--rag-device", type=str, default="cuda")
    # Output
    p.add_argument("--output-dir", type=str, required=True)
    p.add_argument("--no-tta", action="store_true")
    p.add_argument("--gradcam", action="store_true")
    p.add_argument("--clahe-clip", type=float, default=2.5)
    return p.parse_args()


def _amp_dtype(name: str) -> torch.dtype:
    return {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[name]


def _load_model(cfg: dict, ckpt_path: str, device: torch.device) -> SpecialistCXR:
    classes = cfg["data"]["classes"]
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = SpecialistCXR(
        timm_name=cfg["model"]["name"],
        num_classes=len(classes),
        pretrained=False,
        drop_path_rate=cfg["model"]["drop_path_rate"],
        grad_checkpointing=False,
    )
    state_key = "ema" if "ema" in ckpt else "model"
    model.load_state_dict(ckpt[state_key])
    model = model.to(device).eval()
    print(f"[ckpt] loaded '{state_key}' from {ckpt_path}", flush=True)
    return model


def _preprocess(image_path: str, image_size: int, clahe_clip: float):
    import albumentations as A
    import cv2

    eval_tfms = build_eval_transforms(image_size=image_size)
    gray = load_cxr_grayscale(image_path)
    gray = apply_clahe(gray, clip_limit=clahe_clip)
    rgb = to_three_channel(gray)
    out = eval_tfms(image=rgb)
    tensor = out["image"].float().unsqueeze(0)

    overlay_tfms = A.Compose([
        A.LongestMaxSize(max_size=image_size),
        A.PadIfNeeded(
            min_height=image_size,
            min_width=image_size,
            border_mode=cv2.BORDER_CONSTANT,
            fill=0,
        ),
    ])
    rgb_overlay = overlay_tfms(image=rgb)["image"]
    return tensor, rgb_overlay


@torch.no_grad()
def _forward_tta(model, x, amp_dt, use_tta: bool) -> np.ndarray:
    with autocast(device_type="cuda", dtype=amp_dt):
        logits = model(x)
        if use_tta:
            logits_flip = model(torch.flip(x, dims=[-1]))
            logits = (logits + logits_flip) / 2.0
    return logits.float().cpu().numpy().squeeze(0)


def _gradcam_for_findings(model, x, findings_dict, rgb_overlay, out_dir: Path):
    import cv2
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []
    cam_engine = GradCAMpp(model=model, target_module=model.backbone)
    try:
        for f in findings_dict["findings"]:
            if not f["above_threshold"]:
                continue
            x_req = x.clone().requires_grad_(True)
            cam = cam_engine(x_req, class_idx=f["class_index"])
            overlay = overlay_heatmap(rgb_overlay, cam, alpha=0.4)
            safe_name = f["name"].replace("/", "_").replace(" ", "_")
            p = out_dir / f"cam_{f['class_index']:02d}_{safe_name}.png"
            cv2.imwrite(str(p), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
            paths.append(str(p))
            print(f"[cam] {p.name}", flush=True)
    finally:
        cam_engine.remove()
    return paths


def _maybe_load_retriever(args):
    """Return RadRetriever or None if RAG is disabled / artifacts missing."""
    if args.no_rag:
        return None
    paths = [args.rag_index, args.rag_chunks, args.rag_manifest]
    if any(p is None for p in paths):
        print("[rag] disabled (one of --rag-index/--rag-chunks/--rag-manifest "
              "not provided)", flush=True)
        return None
    for p in paths:
        if not Path(p).exists():
            print(f"[rag] disabled (missing artifact: {p})", flush=True)
            return None
    from radagent.rag.retriever import RadRetriever
    print(f"[rag] loading retriever ...", flush=True)
    r = RadRetriever(
        index_path=args.rag_index,
        chunks_path=args.rag_chunks,
        manifest_path=args.rag_manifest,
        device=args.rag_device,
    )
    print(f"[rag] ready  embed_dim={r.embed_dim}  n_chunks={len(r.chunks)}",
          flush=True)
    return r


def _retrieve_for_findings(retriever, findings_dict: dict, k: int):
    """Mutates findings_dict in place: each above-threshold finding gets
    a retrieved_passages field with up to k Passage dicts.
    """
    if retriever is None:
        return
    for f in findings_dict["findings"]:
        if not f["above_threshold"]:
            continue
        name = f["name"]
        query = FINDING_QUERY_TEMPLATES.get(
            name, f"{name} on chest radiograph: imaging features and differentials."
        )
        passages = retriever.query(query, k=k, finding_filter=[name])
        if not passages:
            # Fall back to unfiltered retrieval if nothing matches the metadata
            passages = retriever.query(query, k=k)
        f["retrieved_passages"] = [p.to_dict() for p in passages]
        f["retrieval_query"] = query
        print(f"[rag] {name}: {len(passages)} passages "
              f"(top score {passages[0].score:.3f})" if passages else
              f"[rag] {name}: 0 passages", flush=True)


def main():
    args = parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    classes: list[str] = cfg["data"]["classes"]
    image_size: int = cfg["data"]["image_size"]

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available.")
    device = torch.device("cuda")
    amp_dt = _amp_dtype(cfg["train"]["amp_dtype"])

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load specialist + calibration
    model = _load_model(cfg, args.checkpoint, device)
    calibration = load_calibration(
        calibration_path=args.calibration,
        class_names=classes,
        bands_path=args.bands,
    )
    if not calibration.bands:
        print("[cal] WARN: no bands_path provided -- using default. "
              "Run scripts/calibrate_bands.py for per-class bands.",
              flush=True)
    print(f"[cal] T={calibration.temperature:.4f}  "
          f"bands={'yes' if calibration.bands else 'default'}",
          flush=True)

    # Load retriever (optional)
    retriever = _maybe_load_retriever(args)

    # Preprocess
    image_path = Path(args.image).resolve()
    if not image_path.exists():
        raise FileNotFoundError(f"image not found: {image_path}")
    tensor, rgb_overlay = _preprocess(
        str(image_path), image_size=image_size, clahe_clip=args.clahe_clip
    )
    tensor = tensor.to(device, non_blocking=True)

    # Specialist forward + TTA
    use_tta = not args.no_tta
    t0 = time.time()
    logits = _forward_tta(model, tensor, amp_dt, use_tta=use_tta)
    fwd_ms = (time.time() - t0) * 1000.0
    print(f"[fwd] {fwd_ms:.1f} ms  TTA={use_tta}", flush=True)

    # Findings dict
    image_meta = {
        "path": str(image_path),
        "size": [int(rgb_overlay.shape[1]), int(rgb_overlay.shape[0])],
        "modality": "CXR",
    }
    model_meta = {
        "checkpoint": str(args.checkpoint),
        "tta": "hflip" if use_tta else "none",
        "amp_dtype": cfg["train"]["amp_dtype"],
        "image_size": image_size,
        "forward_ms": round(fwd_ms, 1),
    }
    findings = probabilities_to_findings(
        logits=logits,
        calibration=calibration,
        image_meta=image_meta,
        model_meta=model_meta,
    )

    # RAG retrieval per above-threshold finding
    t0 = time.time()
    _retrieve_for_findings(retriever, findings, k=args.rag_k)
    rag_ms = (time.time() - t0) * 1000.0
    if retriever is not None:
        findings["model_meta"]["rag_ms"] = round(rag_ms, 1)
        findings["model_meta"]["rag_manifest"] = retriever.manifest

    # Write findings.json (first pass)
    findings_path = out_dir / "findings.json"
    with open(findings_path, "w", encoding="utf-8") as f:
        json.dump(findings, f, indent=2, ensure_ascii=False)
    print(f"[out] {findings_path}", flush=True)

    # Console summary
    print()
    print(f"image: {image_path.name}")
    print(f"assessment: {findings['overall_assessment']}  "
          f"(n_above_threshold={findings['summary']['n_above_threshold']})")
    print(f"{'finding':<22} {'cal_p':>7} {'thr':>7}  {'level':>7}  above  passages")
    print("-" * 70)
    for f in findings["findings"]:
        mark = "*" if f["above_threshold"] else " "
        n_passages = len(f.get("retrieved_passages", []))
        print(f"{f['name']:<22} "
              f"{f['calibrated_probability']:>7.3f} "
              f"{f['threshold']:>7.3f}  "
              f"{f['confidence_level']:>7}  "
              f"  {mark}      {n_passages}")

    # Show RAG snippets for above-threshold findings
    if retriever is not None:
        print()
        print("=" * 70)
        print("Retrieved evidence")
        print("=" * 70)
        for f in findings["findings"]:
            if not f["above_threshold"]:
                continue
            print(f"\n# {f['name']} (calibrated p={f['calibrated_probability']:.3f})")
            for j, p in enumerate(f.get("retrieved_passages", []), 1):
                print(f"  [{j}] {p['title']} > {p['section']}  "
                      f"({p['source']}, score={p['score']:.3f})")
                print(f"      {p['source_url']}")
                snippet = p["text"][:240].replace("\n", " ")
                print(f"      {snippet}...")

    # Grad-CAM (optional)
    if args.gradcam:
        if findings["summary"]["n_above_threshold"] == 0:
            print("\n[cam] no above-threshold findings -- skipping Grad-CAM.",
                  flush=True)
        else:
            cam_paths = _gradcam_for_findings(
                model=model, x=tensor, findings_dict=findings,
                rgb_overlay=rgb_overlay, out_dir=out_dir / "cams",
            )
            findings["cams"] = {
                "note": "Grad-CAM++ on un-flipped forward; TTA-averaged "
                        "logits drive the findings list.",
                "files": cam_paths,
            }
            with open(findings_path, "w", encoding="utf-8") as f:
                json.dump(findings, f, indent=2, ensure_ascii=False)
            print(f"[out] updated {findings_path} with cam paths", flush=True)


if __name__ == "__main__":
    main()
