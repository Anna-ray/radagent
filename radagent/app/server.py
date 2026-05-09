"""
RadAgent dashboard backend — V3 with side-by-side, audit JSON, PDF download.

New in V3 vs V2:
  * Pipeline result cache (keyed by request_id) so /api/audit and /api/pdf
    can re-use the latest result without re-running the model.
  * New endpoints:
      GET  /api/audit/{request_id}     -> JSON audit trace download
      GET  /api/pdf/{request_id}       -> PDF report download
  * Side-by-side support:
      - language="bilingual" generates English first, then re-prompts in Arabic.
      - meta flag "compare_ungrounded": true triggers a parallel ungrounded VLM
        call after the grounded one, returned in evt.stage = "vlm_ungrounded_done".

Endpoints:
  GET  /                          - serves static/index.html
  GET  /health                    - device + model + RAG + cache size
  POST /api/predict               - sync: full pipeline, return JSON
  WS   /ws/predict                - streaming: emit one event per stage
  GET  /api/audit/{request_id}    - download audit.json for a past request
  GET  /api/pdf/{request_id}      - download report.pdf for a past request
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
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from fastapi import FastAPI, File, UploadFile, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, Response
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

# V3 deliverables — must live next to this file
from radagent.app.audit import build_audit_trace, audit_trace_to_json_bytes
from radagent.app.pdf_report import build_pdf_report_bytes

# Agentic RAG
from radagent.inference.agentic_rag import agentic_retrieve, self_audit_report


# ---------------------------------------------------------------------------
# Query templates (clinically informative per finding)
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
app = FastAPI(title="RadAgent", version="0.3.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

STATIC_DIR = Path(__file__).resolve().parent / "static"
STATIC_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


class _LRUCache:
    """Tiny size-bounded cache of pipeline results, keyed by request_id."""

    def __init__(self, max_size: int = 16):
        self.max_size = max_size
        self._d: OrderedDict[str, dict] = OrderedDict()

    def put(self, k: str, v: dict) -> None:
        self._d[k] = v
        self._d.move_to_end(k)
        while len(self._d) > self.max_size:
            self._d.popitem(last=False)

    def get(self, k: str) -> dict | None:
        v = self._d.get(k)
        if v is not None:
            self._d.move_to_end(k)
        return v

    def __len__(self) -> int:
        return len(self._d)


class State:
    cfg: dict
    classes: list[str]
    image_size: int
    amp_dt: torch.dtype
    device: torch.device
    model: SpecialistCXR
    eval_tfms: Any
    calibration: Any
    retriever: Any
    cache: _LRUCache


STATE = State()
STATE.cache = _LRUCache(max_size=16)


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
# Pipeline (mirrors predict_one.py)
# ---------------------------------------------------------------------------
def _preprocess_to_tensor(image_path: str, clahe_clip: float = 2.5):
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


async def _retrieve_for_findings_agentic(structured: dict, k: int = 3) -> tuple[dict[str, list[dict]], dict]:
    """Agentic retrieval: VLM evaluates sufficiency and refines queries if needed.
    
    Returns:
        Tuple of (retrieved passages dict, agentic trace dict)
    """
    out: dict[str, list[dict]] = {}
    agentic_traces = {}
    
    if STATE.retriever is None:
        return out, {"enabled": False, "reason": "no retriever"}
    
    if not VLLM_URL:
        # Fall back to non-agentic retrieval if VLM not available
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
        return out, {"enabled": False, "reason": "no vllm"}
    
    # Agentic retrieval for each above-threshold finding
    for f in structured["findings"]:
        if not f["above_threshold"]:
            continue
        name = f["name"]
        initial_query = FINDING_QUERY_TEMPLATES.get(
            name, f"{name} on chest radiograph: imaging features and differentials."
        )
        
        passages, trace = await agentic_retrieve(
            finding=f,
            retriever=STATE.retriever,
            initial_query=initial_query,
            vllm_url=VLLM_URL,
            vllm_model=VLLM_MODEL,
            k=k,
            max_iterations=2,
        )
        out[name] = passages
        agentic_traces[name] = trace
    
    return out, {"enabled": True, "traces": agentic_traces}


# ---------------------------------------------------------------------------
# VLM prompts — three modes: grounded EN/AR, plus ungrounded baseline
# ---------------------------------------------------------------------------
def _grounded_instruction(language: str) -> str:
    if language == "ar":
        return (
            "اكتب تقريرًا إشعاعيًا منظمًا للأشعة السينية للصدر باللغة العربية. "
            "اشمل الأقسام: النتائج، الانطباع، التوصيات. "
            "اربط كل ادعاء بإحدى الفقرات المسترجعة باستخدام علامات [1] [2] [3]. "
            "لا تخترع نتائج لم يرصدها المتخصص. "
            "قائمة المراجع في النهاية تربط [n] إلى رابط المصدر."
        )
    if language == "bilingual":
        return (
            "Write a structured chest radiograph report. Include sections: "
            "Findings, Impression, Recommendations. Ground every clinical claim "
            "in one of the retrieved passages using bracket citations [1] [2] [3]. "
            "Do not invent findings the specialist did not flag. End with a "
            "numbered citation list mapping each [n] to its source URL.\n\n"
            "After the English report, write the same report in Arabic under a "
            "heading '## التقرير بالعربية'."
        )
    return (
        "Write a structured chest radiograph report. "
        "Include sections: Findings, Impression, Recommendations. "
        "Ground every clinical claim in one of the retrieved passages using bracket "
        "citations [1] [2] [3]. Do not invent findings the specialist did not flag. "
        "End with a numbered citation list mapping each [n] to its source URL."
    )


async def _generate_grounded_report(structured: dict, retrieved: dict, language: str = "en") -> str | None:
    if not VLLM_URL:
        return None

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

    user = (
        f"{_grounded_instruction(language)}\n\n"
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
        "max_tokens": 800,
        "temperature": 0.2,
    }
    return await _post_with_retries(payload)


async def _post_with_retries(payload: dict, *, max_attempts: int = 3) -> str:
    """POST to VLLM_URL with retry-on-network-error.

    Windows in particular sometimes returns transient `getaddrinfo failed`
    on the first DNS resolution; subsequent attempts (after the resolver has
    cached) succeed. We catch that class of error and retry with a small
    backoff. Surfaces a useful error message including the URL we tried.
    """
    import httpx

    url = f"{VLLM_URL.rstrip('/')}/chat/completions"
    last_exc: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            async with httpx.AsyncClient(timeout=90.0) as client:
                r = await client.post(url, json=payload)
                r.raise_for_status()
                return r.json()["choices"][0]["message"]["content"]
        except httpx.ConnectError as e:
            last_exc = e
            print(f"[vlm] connect error attempt {attempt}/{max_attempts} → {url}: {e}", flush=True)
            await asyncio.sleep(0.4 * attempt)  # 0.4s, 0.8s
            continue
        except Exception as e:
            # non-network failures shouldn't be retried — surface immediately
            raise RuntimeError(f"VLM call failed against {url}: {e}") from e
    raise RuntimeError(f"VLM call failed against {url} after {max_attempts} attempts: {last_exc}")


async def _generate_ungrounded_report(image_b64: str, language: str = "en") -> str | None:
    """Vanilla VLM call: same image, no specialist findings, no retrieved
    evidence, no citation requirement. Used for the side-by-side comparison
    that demonstrates the value of grounding.
    """
    if not VLLM_URL:
        return None

    if language == "ar":
        prompt_text = (
            "صف هذه الأشعة السينية للصدر. اذكر النتائج المرضية إن وجدت، "
            "والانطباع التشخيصي، والتوصيات."
        )
    else:
        prompt_text = (
            "Describe this chest X-ray. State any pathological findings, "
            "the diagnostic impression, and recommendations."
        )

    payload = {
        "model": VLLM_MODEL,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {
                        "url": f"data:image/png;base64,{image_b64}"
                    }},
                    {"type": "text", "text": prompt_text},
                ],
            },
        ],
        "max_tokens": 600,
        "temperature": 0.2,
    }
    return await _post_with_retries(payload)


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------
async def _full_pipeline(
    raw_bytes: bytes,
    *,
    language: str = "en",
    compare_ungrounded: bool = False,
    image_filename: str | None = None,
    request_id: str | None = None,
    emit=None,
) -> dict:
    request_id = request_id or uuid.uuid4().hex[:8]

    async def _emit(stage: str, data: dict | None = None):
        if emit:
            await emit({"stage": stage, "data": data or {}})

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
        image_meta = {"path": str(tmp_path), "image_size": STATE.image_size, "forward_ms": round(spec_ms, 1)}
        model_meta = {"backbone": STATE.cfg["model"]["name"], "amp_dtype": STATE.cfg["train"]["amp_dtype"], "image_size": STATE.image_size}
        structured = probabilities_to_findings(
            logits=logits, calibration=STATE.calibration,
            image_meta=image_meta, model_meta=model_meta,
        )
        findings_ms = (time.time() - t) * 1000.0
        n_above = sum(1 for f in structured["findings"] if f["above_threshold"])
        await _emit("findings_done", {"n_above": n_above, "structured": structured})

        # RAG (agentic)
        t = time.time()
        await _emit("retrieval")
        retrieved, agentic_trace = await _retrieve_for_findings_agentic(structured)
        rag_ms = (time.time() - t) * 1000.0
        await _emit("retrieval_done", {"retrieved": retrieved})
        await _emit("agentic_retrieval_done", {"trace": agentic_trace})

        # GRAD-CAM (top-3 above-threshold)
        t = time.time()
        await _emit("gradcam")
        cams_b64: dict[str, str] = {}
        above_findings = [f for f in structured["findings"] if f["above_threshold"]][:3]
        for f in above_findings:
            try:
                cams_b64[f["name"]] = _gradcam_b64_for_class(x, rgb, class_idx=f["class_index"])
            except Exception as e:
                print(f"[gradcam] failed for {f['name']}: {e}", flush=True)
        cam_ms = (time.time() - t) * 1000.0
        await _emit("gradcam_done", {"cams": cams_b64})

        # also expose the original image as b64 for UI display + ungrounded VLM call
        with open(tmp_path, "rb") as f_in:
            input_b64 = base64.b64encode(f_in.read()).decode("ascii")

        # VLM: grounded (always if VLLM_URL) and optionally ungrounded baseline (parallel)
        t = time.time()
        report = None
        report_ungrounded = None
        vlm_error = None
        vlm_ungrounded_error = None
        vlm_ms = 0.0
        vlm_ungrounded_ms = 0.0

        if VLLM_URL:
            await _emit("vlm")
            t_g = time.time()

            # Grounded first — this is the headline output, ship it ASAP
            try:
                report = await _generate_grounded_report(structured, retrieved, language=language)
            except Exception as e:
                vlm_error = str(e)
                print(f"[vlm] grounded call failed: {e}", flush=True)

            # Notify the UI that grounded is ready (it can start rendering while
            # ungrounded is still in flight)
            await _emit("vlm_grounded_done", {
                "report": report,
                "vlm_error": vlm_error,
            })

            # Ungrounded baseline (only if requested) — sequential to avoid
            # asyncio.gather DNS/connection races on Windows.
            if compare_ungrounded:
                try:
                    report_ungrounded = await _generate_ungrounded_report(input_b64, language=language)
                except Exception as e:
                    vlm_ungrounded_error = str(e)
                    print(f"[vlm] ungrounded call failed: {e}", flush=True)

                await _emit("vlm_ungrounded_done", {
                    "report_ungrounded": report_ungrounded,
                    "vlm_ungrounded_error": vlm_ungrounded_error,
                })

            vlm_ms = (time.time() - t_g) * 1000.0

        # Self-audit (if VLM available and report generated)
        self_audit = None
        if VLLM_URL and report:
            try:
                t_audit = time.time()
                self_audit = await self_audit_report(
                    report_text=report,
                    structured_findings=structured,
                    retrieved_passages_by_finding=retrieved,
                    vllm_url=VLLM_URL,
                    vllm_model=VLLM_MODEL,
                )
                audit_ms = (time.time() - t_audit) * 1000.0
                await _emit("self_audit_done", {"audit": self_audit, "audit_ms": round(audit_ms, 1)})
            except Exception as e:
                print(f"[agentic-rag] self-audit failed: {e}", flush=True)
                self_audit = {"flags": [], "audit_summary": f"audit error: {e}"}

        total_ms = (time.time() - t_total) * 1000.0

        result = {
            "request_id": request_id,
            "image_filename": image_filename,
            "structured": structured,
            "retrieved": retrieved,
            "agentic_trace": agentic_trace,
            "self_audit": self_audit,
            "cams_b64": cams_b64,
            "input_b64": input_b64,
            "report": report,
            "report_ungrounded": report_ungrounded,
            "vlm_error": vlm_error,
            "vlm_ungrounded_error": vlm_ungrounded_error,
            "language": language,
            "compare_ungrounded": compare_ungrounded,
            "vllm_enabled": bool(VLLM_URL),
            "vllm_model": VLLM_MODEL if VLLM_URL else None,
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

        # cache the result for /api/audit + /api/pdf
        STATE.cache.put(request_id, result)

        await _emit("done", {
            "request_id": request_id,
            "timings_ms": result["timings_ms"],
            "report": report,
            "report_ungrounded": report_ungrounded,
            "vlm_error": vlm_error,
            "vlm_ungrounded_error": vlm_ungrounded_error,
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
        "cache_size": len(STATE.cache),
        "version": "0.3.0",
    }


@app.post("/api/predict")
async def predict(file: UploadFile = File(...), language: str = "en", compare_ungrounded: bool = False):
    raw = await file.read()
    if not raw:
        return JSONResponse({"error": "empty upload"}, status_code=400)
    try:
        result = await _full_pipeline(
            raw, language=language,
            compare_ungrounded=compare_ungrounded,
            image_filename=file.filename,
        )
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
        meta_text = await ws.receive_text()
        try:
            meta = json.loads(meta_text)
        except Exception:
            meta = {}
        language = meta.get("language", "en")
        compare_ungrounded = bool(meta.get("compare_ungrounded", False))
        image_filename = meta.get("filename")

        msg = await ws.receive()
        raw = msg.get("bytes")
        if not raw:
            await ws.send_json({"stage": "error", "data": {"error": "expected image bytes after meta JSON"}})
            await ws.close()
            return

        async def emit(evt: dict):
            evt["request_id"] = request_id
            await ws.send_json(evt)

        await _full_pipeline(
            raw,
            language=language,
            compare_ungrounded=compare_ungrounded,
            image_filename=image_filename,
            request_id=request_id,
            emit=emit,
        )
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


def _result_or_404(request_id: str) -> dict:
    res = STATE.cache.get(request_id)
    if res is None:
        raise HTTPException(status_code=404, detail=f"no cached result for request_id={request_id}")
    return res


def _config_summary() -> dict:
    """Lightweight subset of config for inclusion in audit traces."""
    cfg = getattr(STATE, "cfg", {}) or {}
    model = cfg.get("model", {}) or {}
    train = cfg.get("train", {}) or {}
    data = cfg.get("data", {}) or {}
    return {
        "model_backbone": model.get("name"),
        "image_size": data.get("image_size"),
        "classes": data.get("classes"),
        "amp_dtype": train.get("amp_dtype"),
    }


@app.get("/api/audit/{request_id}")
async def get_audit(request_id: str):
    res = _result_or_404(request_id)
    audit = build_audit_trace(
        request_id=request_id,
        image_filename=res.get("image_filename"),
        structured=res["structured"],
        retrieved=res["retrieved"],
        cams_b64=res["cams_b64"],
        report=res.get("report"),
        vlm_error=res.get("vlm_error"),
        language=res.get("language", "en"),
        timings_ms=res.get("timings_ms", {}),
        vllm_enabled=res.get("vllm_enabled", False),
        vllm_model=res.get("vllm_model"),
        config_summary=_config_summary(),
    )
    body = audit_trace_to_json_bytes(audit)
    return Response(
        content=body,
        media_type="application/json",
        headers={
            "Content-Disposition": f'attachment; filename="radagent_audit_{request_id}.json"',
        },
    )


@app.get("/api/pdf/{request_id}")
async def get_pdf(request_id: str):
    res = _result_or_404(request_id)
    pdf = build_pdf_report_bytes(
        request_id=request_id,
        image_filename=res.get("image_filename"),
        structured=res["structured"],
        retrieved=res["retrieved"],
        cams_b64=res["cams_b64"],
        report=res.get("report"),
        vlm_error=res.get("vlm_error"),
        language=res.get("language", "en"),
        timings_ms=res.get("timings_ms", {}),
        vllm_enabled=res.get("vllm_enabled", False),
        vllm_model=res.get("vllm_model"),
    )
    return Response(
        content=pdf,
        media_type="application/pdf",
        headers={
            "Content-Disposition": f'attachment; filename="radagent_report_{request_id}.pdf"',
        },
    )


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
