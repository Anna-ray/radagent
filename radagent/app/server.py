"""
RadAgent dashboard backend.

Adapts scripts/predict_one.py into a FastAPI + WebSocket service.

Endpoints:
  GET  /              - serves static/index.html
  GET  /health        - device + model + RAG status
  POST /api/predict   - sync: upload CXR, get full structured result
  WS   /ws/predict    - streaming: emit one event per stage
"""
from __future__ import annotations
import asyncio
import base64
import io
import cv2
import json
import os
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from fastapi import FastAPI, File, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from torch.amp import autocast

# Project imports — match predict_one.py exactly
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

# ---------------------------------------------------------------------------
# Query templates (copied from predict_one.py — clinically informative per finding)
# ---------------------------------------------------------------------------
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
        "Emphysema on chest radiograph: hyperlucency, flattened "
        "diaphragm, increased retrosternal space.",
    "Fibrosis":
        "Pulmonary fibrosis on chest radiograph: reticular opacities, "
        "honeycombing, traction bronchiectasis.",
    "Pleural_Thickening":
        "Pleural thickening on chest radiograph: imaging features, "
        "asbestos exposure differentials.",
    "Hernia":
        "Diaphragmatic hernia on chest radiograph: imaging features "
        "and differential diagnosis.",
}

# ---------------------------------------------------------------------------
# Config (env vars override)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]

CONFIG_PATH      = Path(os.getenv("RADAGENT_CONFIG", PROJECT_ROOT / "configs" / "nih14_convnextv2_base.yaml"))
CHECKPOINT_PATH  = Path(os.getenv("RADAGENT_CHECKPOINT", PROJECT_ROOT / "runs" / "nih14_convnextv2_base_384" / "best.pt"))
CALIBRATION_PATH = Path(os.getenv("RADAGENT_CALIBRATION", PROJECT_ROOT / "runs" / "nih14_convnextv2_base_384" / "calibration.json"))
BANDS_PATH       = Path(os.getenv("RADAGENT_BANDS", PROJECT_ROOT / "runs" / "nih14_convnextv2_base_384" / "calibration_bands.json"))
RAG_INDEX_PATH   = Path(os.getenv("RADAGENT_RAG_INDEX", PROJECT_ROOT / "data" / "rag" / "index.faiss"))
RAG_CHUNKS_PATH  = Path(os.getenv("RADAGENT_RAG_CHUNKS", PROJECT_ROOT / "data" / "rag" / "chunks.jsonl"))
RAG_MANIFEST     = Path(os.getenv("RADAGENT_RAG_MANIFEST", PROJECT_ROOT / "data" / "rag" / "manifest.json"))

VLLM_URL    = os.getenv("VLLM_URL", "")
VLLM_MODEL  = os.getenv("VLLM_MODEL", "Qwen/Qwen2.5-VL-7B-Instruct")
DEVICE_NAME = "cuda" if torch.cuda.is_available() else "cpu"


def _amp_dtype(name: str) -> torch.dtype:
    return {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[name]


# ---------------------------------------------------------------------------
# App + global state
# ---------------------------------------------------------------------------
app = FastAPI(title="RadAgent", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

STATIC_DIR = Path(__file__).resolve().parent / "static"
STATIC_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


class State:
    cfg: dict
    classes: list[str]
    image_size: int
    amp_dt: torch.dtype
    device: torch.device
    model: SpecialistCXR
    eval_tfms: Any
    calibration: Any
    retriever: Any  # RadRetriever | None


STATE = State()


@app.on_event("startup")
async def _startup():
    print(f"[radagent] device={DEVICE_NAME}", flush=True)
    print(f"[radagent] config={CONFIG_PATH}", flush=True)
    cfg = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    STATE.cfg = cfg
    STATE.classes = list(cfg["data"]["classes"])
    STATE.image_size = int(cfg["data"]["image_size"])
    STATE.amp_dt = _amp_dtype(cfg["train"]["amp_dtype"])
    STATE.device = torch.device(DEVICE_NAME)

    print(f"[radagent] loading specialist from {CHECKPOINT_PATH} ...", flush=True)
    ckpt = torch.load(str(CHECKPOINT_PATH), map_location="cpu", weights_only=False)
    model = SpecialistCXR(
        timm_name=cfg["model"]["name"],
        num_classes=len(STATE.classes),
        pretrained=False,
        drop_path_rate=cfg["model"]["drop_path_rate"],
        grad_checkpointing=False,
    )
    state_key = "ema" if "ema" in ckpt else "model"
    model.load_state_dict(ckpt[state_key])
    model = model.to(STATE.device).eval()
    print(f"[ckpt] loaded '{state_key}' from {CHECKPOINT_PATH}", flush=True)
    STATE.model = model

    STATE.eval_tfms = build_eval_transforms(image_size=STATE.image_size)

    print(f"[radagent] loading calibration ...", flush=True)
    STATE.calibration = load_calibration(
        calibration_path=str(CALIBRATION_PATH),
        class_names=STATE.classes,
        bands_path=str(BANDS_PATH) if BANDS_PATH.exists() else None,
    )

    if RAG_INDEX_PATH.exists() and RAG_CHUNKS_PATH.exists():
        from radagent.rag.retriever import RadRetriever
        print(f"[radagent] loading retriever ...", flush=True)
        STATE.retriever = RadRetriever(
            index_path=str(RAG_INDEX_PATH),
            chunks_path=str(RAG_CHUNKS_PATH),
            manifest_path=str(RAG_MANIFEST) if RAG_MANIFEST.exists() else None,
        )
    else:
        print("[radagent] no RAG index found, retriever disabled", flush=True)
        STATE.retriever = None

    print("[radagent] ready.", flush=True)


# ---------------------------------------------------------------------------
# Pipeline (mirrors predict_one.py exactly)
# ---------------------------------------------------------------------------
def _preprocess_to_tensor(image_path: str, clahe_clip: float = 2.5):
    """Same idiom as predict_one._preprocess(): grayscale -> CLAHE -> RGB -> tfms."""
    gray = load_cxr_grayscale(image_path)
    gray = apply_clahe(gray, clip_limit=clahe_clip)
    rgb = to_three_channel(gray)
    out = STATE.eval_tfms(image=rgb)
    tensor = out["image"].float().unsqueeze(0)
    return tensor, rgb


def _forward_with_tta(x: torch.Tensor) -> np.ndarray:
    with torch.no_grad():
        with autocast(device_type=DEVICE_NAME, dtype=STATE.amp_dt):
            logits = STATE.model(x)
            logits_flip = STATE.model(torch.flip(x, dims=[-1]))
            logits = (logits + logits_flip) / 2.0
    return logits.squeeze(0).float().cpu().numpy()


def _gradcam_b64_for_class(image_tensor: torch.Tensor, rgb: np.ndarray, class_idx: int) -> str:
    """Compute Grad-CAM++ for a single class, render overlay on RGB, return base64 PNG."""
    cam_engine = GradCAMpp(model=STATE.model, target_module=STATE.model.backbone)
    try:
        x_req = image_tensor.clone().requires_grad_(True)
        cam = cam_engine(x_req, class_idx=class_idx)
        if isinstance(cam, torch.Tensor):
            cam = cam.detach().cpu().numpy()
        if cam.ndim == 3:
            cam = cam[0]
        if cam.shape[:2] != rgb.shape[:2]:
            cam = cv2.resize(cam, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_LINEAR)
        overlay = overlay_heatmap(rgb, cam, alpha=0.4)
        pil_img = Image.fromarray(overlay)
        buf = io.BytesIO()
        pil_img.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("ascii")
    finally:
        cam_engine.remove()
def _retrieve_for_findings(structured: dict, k: int = 3) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    if STATE.retriever is None:
        return out
    for f in structured["findings"]:
        if not f["above_threshold"]:
            continue
        name = f["name"]
        query = FINDING_QUERY_TEMPLATES.get(
            name, f"{name} on chest radiograph: imaging features and differentials."
        )
        passages = STATE.retriever.query(query, k=k, finding_filter=[name])
        if not passages:
            passages = STATE.retriever.query(query, k=k)
        out[name] = [p.to_dict() for p in passages]
    return out


async def _generate_report(structured: dict, retrieved: dict, language: str = "en") -> str | None:
    if not VLLM_URL:
        return None
    import httpx

    above = [f for f in structured["findings"] if f["above_threshold"]]
    above_lines = [
        f"- {f['name']}: cal_p={f['calibrated_probability']:.3f}, "
        f"confidence={f['confidence_level']}"
        for f in above
    ]
    passages_lines = []
    for finding_name, items in retrieved.items():
        for i, p in enumerate(items, start=1):
            preview = (p["text"][:240] + "...") if len(p.get("text", "")) > 240 else p.get("text", "")
            passages_lines.append(
                f"[{finding_name} #{i}] {p.get('title','?')} > {p.get('section','?')} "
                f"({p.get('source','?')}): {preview}  (URL: {p.get('source_url','')})"
            )

    if language == "ar":
        instruction = (
            "اكتب تقريرًا إشعاعيًا منظمًا للأشعة السينية للصدر باللغة العربية. "
            "اشمل الأقسام: النتائج، الانطباع، التوصيات. "
            "اربط كل ادعاء بإحدى الفقرات المسترجعة باستخدام علامات [1] [2] [3]. "
            "لا تخترع نتائج لم يرصدها المتخصص."
        )
    else:
        instruction = (
            "Write a structured chest radiograph report. "
            "Include sections: Findings, Impression, Recommendations. "
            "Ground every clinical claim in one of the retrieved passages using bracket "
            "citations [1] [2] [3]. Do not invent findings the specialist did not flag. "
            "End with a numbered citation list mapping each [n] to its source URL."
        )

    user = (
        f"{instruction}\n\n"
        f"SPECIALIST FINDINGS (above threshold):\n" + ("\n".join(above_lines) if above_lines else "  none") +
        f"\n\nOVERALL ASSESSMENT: {structured['overall_assessment']}\n\n"
        f"RETRIEVED PASSAGES:\n" + ("\n".join(passages_lines) if passages_lines else "  none")
    )

    payload = {
        "model": VLLM_MODEL,
        "messages": [
            {"role": "system", "content": "You are a careful radiology assistant."},
            {"role": "user", "content": user},
        ],
        "max_tokens": 600,
        "temperature": 0.2,
    }
    async with httpx.AsyncClient(timeout=60.0) as client:
        r = await client.post(f"{VLLM_URL.rstrip('/')}/chat/completions", json=payload)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]


async def _full_pipeline(raw_bytes: bytes, *, language: str = "en", emit=None) -> dict:
    async def _emit(stage: str, data: dict | None = None):
        if emit:
            await emit({"stage": stage, "data": data or {}})

    # write upload to a temp file because preprocess_cxr / load_cxr_grayscale take paths
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
        tmp.write(raw_bytes)
        tmp_path = tmp.name

    try:
        t_total = time.time()

        # PRE
        await _emit("preprocessing")
        t = time.time()
        x_cpu, rgb = _preprocess_to_tensor(tmp_path)
        x = x_cpu.to(STATE.device)
        pre_ms = (time.time() - t) * 1000.0

        # SPECIALIST
        await _emit("specialist")
        t = time.time()
        logits = _forward_with_tta(x)
        spec_ms = (time.time() - t) * 1000.0

        # FINDINGS
        t = time.time()
        image_meta = {
            "path": str(tmp_path),
            "image_size": STATE.image_size,
            "forward_ms": round(spec_ms, 1),
        }
        model_meta = {
            "backbone": STATE.cfg["model"]["name"],
            "amp_dtype": STATE.cfg["train"]["amp_dtype"],
            "image_size": STATE.image_size,
        }
        structured = probabilities_to_findings(
            logits=logits,
            calibration=STATE.calibration,
            image_meta=image_meta,
            model_meta=model_meta,
        )
        findings_ms = (time.time() - t) * 1000.0
        n_above = sum(1 for f in structured["findings"] if f["above_threshold"])
        await _emit("findings_done", {"n_above": n_above, "structured": structured})

        # RAG
        t = time.time()
        await _emit("retrieval")
        retrieved = _retrieve_for_findings(structured)
        rag_ms = (time.time() - t) * 1000.0
        await _emit("retrieval_done", {"retrieved": retrieved})

        # GRAD-CAM (top-3 above-threshold)
        t = time.time()
        await _emit("gradcam")
        cams_b64: dict[str, str] = {}
        above = [f for f in structured["findings"] if f["above_threshold"]][:3]
        for f in above:
            try:
                cams_b64[f["name"]] = _gradcam_b64_for_class(x, rgb, class_idx=f["class_index"])
            except Exception as e:
                print(f"[gradcam] failed for {f['name']}: {e}", flush=True)
        cam_ms = (time.time() - t) * 1000.0
        await _emit("gradcam_done", {"cams": cams_b64})

        # VLM
        t = time.time()
        report = None
        vlm_error = None
        if VLLM_URL:
            await _emit("vlm")
            try:
                report = await _generate_report(structured, retrieved, language=language)
            except Exception as e:
                vlm_error = str(e)
        vlm_ms = (time.time() - t) * 1000.0

        total_ms = (time.time() - t_total) * 1000.0

        # also include the original image as b64 so the UI can display it
        with open(tmp_path, "rb") as f_in:
            input_b64 = base64.b64encode(f_in.read()).decode("ascii")

        result = {
            "structured": structured,
            "retrieved": retrieved,
            "cams_b64": cams_b64,
            "input_b64": input_b64,
            "report": report,
            "vlm_error": vlm_error,
            "language": language,
            "vllm_enabled": bool(VLLM_URL),
            "timings_ms": {
                "pre": round(pre_ms, 1),
                "spec": round(spec_ms, 1),
                "findings": round(findings_ms, 1),
                "rag": round(rag_ms, 1),
                "gradcam": round(cam_ms, 1),
                "vlm": round(vlm_ms, 1),
                "total": round(total_ms, 1),
            },
        }
        await _emit("done", {
            "timings_ms": result["timings_ms"],
            "report": report,
            "vlm_error": vlm_error,
        })
        return result

    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/health")
async def health():
    return {
        "ok": True,
        "device": DEVICE_NAME,
        "vllm_enabled": bool(VLLM_URL),
        "vllm_url": VLLM_URL or None,
        "vllm_model": VLLM_MODEL if VLLM_URL else None,
        "n_classes": len(STATE.classes) if hasattr(STATE, "classes") else None,
        "rag_enabled": bool(getattr(STATE, "retriever", None)),
    }


@app.post("/api/predict")
async def predict(file: UploadFile = File(...), language: str = "en"):
    raw = await file.read()
    if not raw:
        return JSONResponse({"error": "empty upload"}, status_code=400)
    try:
        result = await _full_pipeline(raw, language=language)
        return JSONResponse(result)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse({"error": str(e)}, status_code=500)


@app.websocket("/ws/predict")
async def ws_predict(ws: WebSocket):
    await ws.accept()
    request_id = uuid.uuid4().hex[:8]
    try:
        # First frame: optional JSON metadata
        meta_text = await ws.receive_text()
        try:
            meta = json.loads(meta_text)
        except Exception:
            meta = {}
        language = meta.get("language", "en")

        # Second frame: image bytes
        msg = await ws.receive()
        raw = msg.get("bytes")
        if not raw:
            await ws.send_json({"stage": "error", "data": {"error": "expected image bytes after meta JSON"}})
            await ws.close()
            return

        async def emit(evt: dict):
            evt["request_id"] = request_id
            await ws.send_json(evt)

        await _full_pipeline(raw, language=language, emit=emit)
        await ws.close()
    except WebSocketDisconnect:
        pass
    except Exception as e:
        import traceback
        traceback.print_exc()
        try:
            await ws.send_json({"stage": "error", "data": {"error": str(e)}})
            await ws.close()
        except Exception:
            pass


@app.get("/", response_class=HTMLResponse)
async def index():
    f = STATIC_DIR / "index.html"
    if not f.exists():
        return HTMLResponse(
            "<h2>RadAgent backend live.</h2>"
            "<p>The dashboard frontend is not yet installed.</p>"
            "<p>Try <a href='/health'>/health</a> or POST a CXR to <code>/api/predict</code>.</p>"
        )
    return HTMLResponse(f.read_text(encoding="utf-8"))
