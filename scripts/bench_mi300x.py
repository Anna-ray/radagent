"""
scripts/bench_mi300x.py
-----------------------
End-to-end RadAgent benchmark on MI300X.

Pipeline measured per case:
  1. Specialist forward (CXR -> 14 calibrated probabilities)
  2. RAG retrieval (per above-threshold finding)
  3. VLM call (vLLM HTTP endpoint, OpenAI-compatible)

Outputs a JSON report + a CSV with per-image timings, and prints a
summary table.

Usage on droplet:
    python -u -m scripts.bench_mi300x \
        --config configs/nih14_convnextv2_base.yaml \
        --checkpoint runs/nih14_convnextv2_base_384/best.pt \
        --calibration runs/nih14_convnextv2_base_384/calibration.json \
        --bands runs/nih14_convnextv2_base_384/calibration_bands.json \
        --rag-index data/rag/index.faiss \
        --rag-chunks data/rag/chunks.jsonl \
        --rag-manifest data/rag/manifest.json \
        --image-dir data/samples \
        --vllm-url http://localhost:8000/v1 \
        --vllm-model meta-llama/Llama-3.2-11B-Vision-Instruct \
        --output-dir runs/bench_mi300x \
        --n-images 10
"""
from __future__ import annotations

import argparse
import base64
import json
import statistics
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
from radagent.models.specialist import SpecialistCXR


REPORT_PROMPT = """\
You are a radiologist's assistant. The following structured findings come
from a calibrated specialist model on a chest X-ray. Each finding includes
a calibrated probability and confidence level. Retrieved evidence passages
from public medical references are also provided per finding.

Your task: write a concise, structured chest radiograph report with:
- Findings (anatomical observations grounded in the provided evidence)
- Impression (1-2 sentences)
- Recommendations (if any)

You MUST cite the source URLs for any clinical claim drawn from the
retrieved passages. Use bracket notation like [1], [2], with the URL
list at the bottom of the report.

Do NOT invent findings beyond the provided structured input. If the
input says a finding is below threshold, do not assert it.
"""


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--calibration", type=str, required=True)
    p.add_argument("--bands", type=str, default=None)
    p.add_argument("--rag-index", type=str, required=True)
    p.add_argument("--rag-chunks", type=str, required=True)
    p.add_argument("--rag-manifest", type=str, required=True)
    p.add_argument("--image-dir", type=str, required=True)
    p.add_argument("--n-images", type=int, default=10)
    p.add_argument("--vllm-url", type=str, default="http://localhost:8000/v1")
    p.add_argument("--vllm-model", type=str,
                   default="meta-llama/Llama-3.2-11B-Vision-Instruct")
    p.add_argument("--vllm-max-tokens", type=int, default=400)
    p.add_argument("--output-dir", type=str, required=True)
    p.add_argument("--rag-k", type=int, default=3)
    return p.parse_args()


def _amp_dtype(name: str) -> torch.dtype:
    return {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[name]


def _build_vlm_messages(findings: dict, image_b64: str) -> list[dict]:
    above = [f for f in findings["findings"] if f["above_threshold"]]
    structured = []
    for f in above:
        block = {
            "name": f["name"],
            "calibrated_probability": round(f["calibrated_probability"], 3),
            "confidence_level": f["confidence_level"],
            "evidence": [
                {
                    "title": p["title"],
                    "section": p["section"],
                    "source_url": p["source_url"],
                    "snippet": p["text"][:600],
                }
                for p in f.get("retrieved_passages", [])
            ],
        }
        structured.append(block)
    user_text = (
        "Structured findings:\n"
        + json.dumps(structured, indent=2)
        + "\n\nWrite the report."
    )
    return [
        {"role": "system", "content": REPORT_PROMPT},
        {"role": "user", "content": [
            {"type": "image_url",
             "image_url": {"url": f"data:image/png;base64,{image_b64}"}},
            {"type": "text", "text": user_text},
        ]},
    ]


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    classes = cfg["data"]["classes"]
    image_size = cfg["data"]["image_size"]

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available.")
    device = torch.device("cuda")
    amp_dt = _amp_dtype(cfg["train"]["amp_dtype"])

    # ---- specialist ----
    print("[bench] loading specialist ...", flush=True)
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
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

    # ---- calibration ----
    calibration = load_calibration(
        calibration_path=args.calibration,
        class_names=classes,
        bands_path=args.bands,
    )

    # ---- retriever ----
    print("[bench] loading retriever ...", flush=True)
    from radagent.rag.retriever import RadRetriever
    retriever = RadRetriever(
        index_path=args.rag_index,
        chunks_path=args.rag_chunks,
        manifest_path=args.rag_manifest,
        device="cuda",
    )

    # ---- vllm client ----
    print("[bench] checking vllm endpoint ...", flush=True)
    from openai import OpenAI
    client = OpenAI(base_url=args.vllm_url, api_key="dummy")
    # Warm health check
    try:
        models = client.models.list()
        print(f"[bench] vllm models: {[m.id for m in models.data]}", flush=True)
    except Exception as e:
        raise RuntimeError(f"vLLM endpoint unreachable: {e}")

    # ---- pick images ----
    img_dir = Path(args.image_dir)
    image_paths = sorted(img_dir.glob("*.png"))[:args.n_images]
    if not image_paths:
        raise FileNotFoundError(f"No PNGs in {img_dir}")
    print(f"[bench] {len(image_paths)} images", flush=True)

    eval_tfms = build_eval_transforms(image_size=image_size)
    timings: list[dict] = []

    for i, ip in enumerate(image_paths):
        per = {"image": ip.name}
        # ---- preprocess ----
        t0 = time.perf_counter()
        gray = load_cxr_grayscale(str(ip))
        gray = apply_clahe(gray, clip_limit=2.5)
        rgb = to_three_channel(gray)
        out = eval_tfms(image=rgb)
        tensor = out["image"].float().unsqueeze(0).to(device, non_blocking=True)
        per["pre_ms"] = (time.perf_counter() - t0) * 1000

        # ---- specialist forward + TTA ----
        t0 = time.perf_counter()
        with torch.no_grad(), autocast(device_type="cuda", dtype=amp_dt):
            logits = model(tensor)
            logits_flip = model(torch.flip(tensor, dims=[-1]))
            logits = (logits + logits_flip) / 2.0
        logits = logits.float().cpu().numpy().squeeze(0)
        per["spec_ms"] = (time.perf_counter() - t0) * 1000

        # ---- findings ----
        t0 = time.perf_counter()
        findings = probabilities_to_findings(
            logits=logits,
            calibration=calibration,
            image_meta={"path": str(ip),
                        "size": [int(rgb.shape[1]), int(rgb.shape[0])],
                        "modality": "CXR"},
            model_meta={"checkpoint": str(args.checkpoint), "tta": "hflip"},
        )
        per["findings_ms"] = (time.perf_counter() - t0) * 1000
        per["n_above"] = findings["summary"]["n_above_threshold"]

        # ---- RAG ----
        t0 = time.perf_counter()
        for f in findings["findings"]:
            if not f["above_threshold"]:
                continue
            query = f"{f['name']} on chest radiograph: imaging features and differentials."
            passages = retriever.query(query, k=args.rag_k, finding_filter=[f["name"]])
            if not passages:
                passages = retriever.query(query, k=args.rag_k)
            f["retrieved_passages"] = [p.to_dict() for p in passages]
        per["rag_ms"] = (time.perf_counter() - t0) * 1000

        # ---- VLM call ----
        with open(ip, "rb") as fp:
            img_b64 = base64.b64encode(fp.read()).decode("ascii")
        messages = _build_vlm_messages(findings, img_b64)

        t0 = time.perf_counter()
        try:
            resp = client.chat.completions.create(
                model=args.vllm_model,
                messages=messages,
                max_tokens=args.vllm_max_tokens,
                temperature=0.2,
            )
            report = resp.choices[0].message.content
            usage = resp.usage
            per["vlm_ms"] = (time.perf_counter() - t0) * 1000
            per["vlm_prompt_tokens"] = usage.prompt_tokens
            per["vlm_completion_tokens"] = usage.completion_tokens
        except Exception as e:
            per["vlm_ms"] = -1.0
            per["vlm_error"] = str(e)[:300]
            report = None

        per["total_ms"] = sum(per.get(k, 0) for k in
                              ["pre_ms", "spec_ms", "findings_ms", "rag_ms", "vlm_ms"]
                              if isinstance(per.get(k), (int, float)) and per[k] > 0)
        timings.append(per)

        # Write per-image artifact
        out_path = out_dir / f"case_{i:03d}_{ip.stem}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump({"timings": per, "findings": findings, "report": report},
                      f, indent=2, ensure_ascii=False)
        print(f"[case {i+1:3d}/{len(image_paths)}] "
              f"pre={per['pre_ms']:.0f}  spec={per['spec_ms']:.0f}  "
              f"rag={per['rag_ms']:.0f}  vlm={per.get('vlm_ms',-1):.0f}  "
              f"total={per['total_ms']:.0f}ms  n_above={per['n_above']}",
              flush=True)

    # ---- summary ----
    def stat(key: str) -> dict:
        vals = [t[key] for t in timings if isinstance(t.get(key), (int, float)) and t[key] > 0]
        if not vals:
            return {"n": 0}
        return {
            "n": len(vals),
            "mean": statistics.mean(vals),
            "median": statistics.median(vals),
            "p95": float(np.percentile(vals, 95)),
            "min": min(vals), "max": max(vals),
        }

    summary = {k: stat(k) for k in
               ["pre_ms", "spec_ms", "findings_ms", "rag_ms", "vlm_ms", "total_ms"]}
    summary["n_cases"] = len(timings)
    summary["vlm_model"] = args.vllm_model

    with open(out_dir / "bench_summary.json", "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "timings": timings}, f, indent=2)

    print()
    print("=" * 64)
    print(f"RadAgent MI300X bench  N={len(timings)}  model={args.vllm_model}")
    print("=" * 64)
    print(f"{'stage':<14} {'mean':>8} {'median':>8} {'p95':>8} {'min':>6} {'max':>6}")
    for stage in ["pre_ms", "spec_ms", "rag_ms", "vlm_ms", "total_ms"]:
        s = summary[stage]
        if s["n"] == 0:
            print(f"{stage:<14}  (no data)")
            continue
        print(f"{stage:<14} {s['mean']:>8.0f} {s['median']:>8.0f} "
              f"{s['p95']:>8.0f} {s['min']:>6.0f} {s['max']:>6.0f}")
    print("=" * 64)


if __name__ == "__main__":
    main()
