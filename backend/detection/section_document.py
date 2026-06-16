from __future__ import annotations

from typing import Any

import cv2
import numpy as np
from PIL import Image

from analysis.model_registry import ModelRegistry


def _edge_sharpness_variance(cv_img_bgr: np.ndarray) -> tuple[float, list[str]]:
    signals: list[str] = []
    gray = cv2.cvtColor(cv_img_bgr, cv2.COLOR_BGR2GRAY)
    work = cv2.resize(gray, (512, 512), interpolation=cv2.INTER_AREA)
    lap = cv2.Laplacian(work.astype(np.float32), cv2.CV_32F)
    block_vars = []
    for r in range(0, 512, 64):
        for c in range(0, 512, 64):
            block_vars.append(float(np.var(lap[r : r + 64, c : c + 64])))
    if not block_vars:
        return 0.0, signals
    cv_val = float(np.std(block_vars) / (np.mean(block_vars) + 1e-8))
    if cv_val < 0.35:
        signals.append("edge_too_sharp")
    score = max(0.0, min((0.45 - cv_val) * 180, 80.0))
    return round(score, 2), signals


def _ocr_text_artifacts(cv_img_bgr: np.ndarray) -> tuple[float, list[str]]:
    signals: list[str] = []
    try:
        import easyocr

        reader = getattr(_ocr_text_artifacts, "_reader", None)
        if reader is None:
            reader = easyocr.Reader(["en"], gpu=False, verbose=False)
            _ocr_text_artifacts._reader = reader
        rgb = cv2.cvtColor(cv_img_bgr, cv2.COLOR_BGR2RGB)
        results = reader.readtext(rgb, detail=1, paragraph=False)
        if not results:
            return 15.0, signals
        confidences = [float(r[2]) for r in results if len(r) >= 3]
        if not confidences:
            return 20.0, signals
        avg_conf = float(np.mean(confidences))
        conf_std = float(np.std(confidences))
        if avg_conf < 0.55:
            signals.append("text_artifacts")
        if conf_std > 0.22:
            signals.append("font_inconsistency")
        score = max(0.0, min((0.65 - avg_conf) * 120 + conf_std * 80, 85.0))
        return round(score, 2), signals
    except Exception:
        gray = cv2.cvtColor(cv_img_bgr, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        text_like = float(np.mean(edges > 0))
        if text_like > 0.08 and text_like < 0.22:
            signals.append("text_region_detected")
        return round(min(text_like * 200, 40.0), 2), signals


def _document_efficientnet_score(
    registry: ModelRegistry, pil_img: Image.Image
) -> tuple[float, list[str]]:
    signals: list[str] = []
    if registry.hf_detector is None:
        return 50.0, signals
    score, details = registry.hf_detector.predict_ai_likelihood(pil_img)
    if score >= 60:
        signals.append("document_ai_classifier")
    if not details.get("available", True):
        return 50.0, signals
    return float(score), signals


def analyze_document_section(
    pil_img: Image.Image,
    cv_img_bgr: np.ndarray,
    image_bytes: bytes,
    registry: ModelRegistry | None = None,
    *,
    has_camera_exif: bool = False,
) -> dict[str, Any]:
    registry = registry or ModelRegistry.get_instance()
    signals: list[str] = []
    sub_scores: list[float] = []

    eff_score, eff_signals = _document_efficientnet_score(registry, pil_img)
    signals.extend(eff_signals)
    sub_scores.append(eff_score)

    ocr_score, ocr_signals = _ocr_text_artifacts(cv_img_bgr)
    signals.extend(ocr_signals)
    sub_scores.append(ocr_score)

    edge_score, edge_signals = _edge_sharpness_variance(cv_img_bgr)
    signals.extend(edge_signals)
    sub_scores.append(edge_score)

    if not has_camera_exif:
        signals.append("missing_exif")
        sub_scores.append(62.0)
    else:
        sub_scores.append(18.0)

    h, w = cv_img_bgr.shape[:2]
    ratio = w / max(h, 1)
    if 1.4 <= ratio <= 1.8:
        sub_scores.append(55.0)
        signals.append("document_aspect_ratio")

    final = float(np.mean(sub_scores)) if sub_scores else 50.0
    return {
        "score": round(min(max(final / 100.0, 0.0), 1.0), 4),
        "score_percent": round(min(max(final, 0.0), 100.0), 2),
        "signals": sorted(set(signals)),
        "model_scores": {
            "classifier": round(eff_score, 2),
            "ocr": ocr_score,
            "edge_sharpness": edge_score,
        },
    }
