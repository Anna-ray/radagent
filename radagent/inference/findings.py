"""
radagent.inference.findings
---------------------------
Pure functional layer: specialist logits -> structured findings dict.

No file I/O. No model loading. No torch. Numpy in, dict out.

This is the "text bridge" of the RadAgent system: it converts the
specialist CV head's raw outputs into a stable schema the RAG retriever
and downstream VLM consume.

Editorial decisions (negation phrasing, omission of low-prob findings,
overall report tone) live in the VLM layer, NOT here. This module
reports all 14 classes faithfully.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

SCHEMA_VERSION = "1.0"


@dataclass(frozen=True)
class CalibrationBundle:
    """Everything needed to convert raw logits to a findings dict."""
    temperature: float
    thresholds: np.ndarray
    class_names: list[str]
    bands: list[tuple[float, float]] = field(default_factory=list)
    bands_method: list[str] = field(default_factory=list)


@dataclass
class StructuredFinding:
    name: str
    class_index: int
    raw_logit: float
    raw_probability: float
    calibrated_probability: float
    threshold: float
    above_threshold: bool
    confidence_level: str
    confidence_band: tuple[float, float]
    bands_method: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "class_index": self.class_index,
            "raw_logit": float(self.raw_logit),
            "raw_probability": float(self.raw_probability),
            "calibrated_probability": float(self.calibrated_probability),
            "threshold": float(self.threshold),
            "above_threshold": bool(self.above_threshold),
            "confidence_level": self.confidence_level,
            "confidence_band": [float(self.confidence_band[0]),
                                float(self.confidence_band[1])],
            "bands_method": self.bands_method,
        }


def _sigmoid(x: np.ndarray) -> np.ndarray:
    out = np.empty_like(x, dtype=np.float64)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    ex = np.exp(x[~pos])
    out[~pos] = ex / (1.0 + ex)
    return out


def _confidence_level(
    p: float,
    threshold: float,
    band: tuple[float, float] | None,
) -> tuple[str, tuple[float, float]]:
    if band is None or band[0] >= band[1]:
        low_cut, high_cut = float(threshold), 0.85
        method_band = (low_cut, high_cut)
    else:
        low_cut, high_cut = float(band[0]), float(band[1])
        method_band = (low_cut, high_cut)

    if p < low_cut:
        return "low", method_band
    if p < high_cut:
        return "medium", method_band
    return "high", method_band


def _overall_assessment(
    above_any: bool,
    max_calibrated_p: float,
    gray_zone_low: float = 0.30,
) -> str:
    if above_any:
        return "abnormal"
    if max_calibrated_p < gray_zone_low:
        return "normal"
    return "uncertain"


def probabilities_to_findings(
    logits: np.ndarray,
    calibration: CalibrationBundle,
    image_meta: dict[str, Any],
    model_meta: dict[str, Any],
) -> dict[str, Any]:
    if logits.ndim != 1:
        raise ValueError(f"logits must be 1-D, got shape {logits.shape}")
    C = logits.shape[0]
    if C != len(calibration.thresholds):
        raise ValueError(
            f"logits has {C} classes but calibration has "
            f"{len(calibration.thresholds)} thresholds"
        )
    if C != len(calibration.class_names):
        raise ValueError(
            f"logits has {C} classes but calibration has "
            f"{len(calibration.class_names)} class names"
        )

    T = max(float(calibration.temperature), 1e-3)
    raw_p = _sigmoid(logits.astype(np.float64))
    cal_p = _sigmoid(logits.astype(np.float64) / T)

    findings: list[StructuredFinding] = []
    for i in range(C):
        band = calibration.bands[i] if i < len(calibration.bands) else None
        method = (calibration.bands_method[i]
                  if i < len(calibration.bands_method) else "default")
        thr = float(calibration.thresholds[i])
        above = bool(cal_p[i] >= thr)
        conf, used_band = _confidence_level(float(cal_p[i]), thr, band)
        findings.append(StructuredFinding(
            name=calibration.class_names[i],
            class_index=i,
            raw_logit=float(logits[i]),
            raw_probability=float(raw_p[i]),
            calibrated_probability=float(cal_p[i]),
            threshold=thr,
            above_threshold=above,
            confidence_level=conf,
            confidence_band=used_band,
            bands_method=method,
        ))

    findings.sort(key=lambda f: f.calibrated_probability, reverse=True)

    above_any = any(f.above_threshold for f in findings)
    max_cal_p = float(max(f.calibrated_probability for f in findings))
    assessment = _overall_assessment(above_any, max_cal_p)

    return {
        "schema_version": SCHEMA_VERSION,
        "image_meta": dict(image_meta),
        "model_meta": {
            **dict(model_meta),
            "calibration_temperature": T,
        },
        "findings": [f.to_dict() for f in findings],
        "overall_assessment": assessment,
        "summary": {
            "n_above_threshold": int(sum(f.above_threshold for f in findings)),
            "max_calibrated_probability": max_cal_p,
        },
    }


def load_calibration(
    calibration_path: str,
    class_names: list[str],
    bands_path: str | None = None,
) -> CalibrationBundle:
    import json

    with open(calibration_path) as f:
        cal = json.load(f)
    temperature = float(cal["temperature"])
    thresholds = np.asarray(cal["thresholds"], dtype=np.float64)
    if len(thresholds) != len(class_names):
        raise ValueError(
            f"calibration.json has {len(thresholds)} thresholds but "
            f"{len(class_names)} class names provided."
        )

    bands: list[tuple[float, float]] = []
    methods: list[str] = []
    if bands_path is not None:
        with open(bands_path) as f:
            b = json.load(f)
        raw_bands = b.get("bands", [])
        raw_methods = b.get("method", [])
        if len(raw_bands) != len(class_names):
            raise ValueError(
                f"calibration_bands.json has {len(raw_bands)} bands but "
                f"{len(class_names)} classes."
            )
        bands = [(float(lo), float(hi)) for lo, hi in raw_bands]
        methods = ([str(m) for m in raw_methods] if raw_methods
                   else ["unknown"] * len(class_names))

    return CalibrationBundle(
        temperature=temperature,
        thresholds=thresholds,
        class_names=list(class_names),
        bands=bands,
        bands_method=methods,
    )
