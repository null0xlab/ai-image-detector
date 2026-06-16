"""
Layer 3 — AI source attribution (which tool likely generated or edited the image).
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from .ensemble_gates import ATTRIBUTION_CONFIDENCE_MIN, both_models_agree_ai
from .layer1_forensic import AI_SOFTWARE_EXIF_MAP

AI_SOFTWARE_MAP = {
    "Adobe Firefly": "Adobe Firefly",
    "DALL-E": "OpenAI DALL-E",
    "Stable Diffusion": "Stable Diffusion",
    "Midjourney": "Midjourney",
    "Canva": "Canva AI",
    "Bing Image Creator": "Microsoft Designer (DALL-E)",
    "ComfyUI": "Stable Diffusion (ComfyUI)",
    "Automatic1111": "Stable Diffusion (A1111)",
}

# SRM-style 3x3 high-pass kernels (subset of 30 SRM filters for speed)
_SRM_KERNELS = [
    np.array([[0, 0, 0], [0, 1, -1], [0, 0, 0]], dtype=np.float32),
    np.array([[0, 0, 0], [0, 1, 0], [0, -1, 0]], dtype=np.float32),
    np.array([[0, 0, 0], [-1, 1, 0], [0, 0, 0]], dtype=np.float32),
    np.array([[-1, 2, -1], [2, -4, 2], [-1, 2, -1]], dtype=np.float32) / 4.0,
    np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], dtype=np.float32) / 4.0,
]


def extract_srm_noise_pattern(gray: np.ndarray) -> dict[str, float]:
    """Residual noise statistics via SRM filter bank."""
    gray_f = gray.astype(np.float32)
    if gray_f.shape[0] > 384:
        gray_f = cv2.resize(gray_f, (384, 384), interpolation=cv2.INTER_AREA)
    residuals = []
    for k in _SRM_KERNELS:
        r = cv2.filter2D(gray_f, -1, k)
        residuals.append(float(np.std(r)))
    arr = np.array(residuals, dtype=np.float64)
    return {
        "srm_mean": float(np.mean(arr)),
        "srm_std": float(np.std(arr)),
        "srm_max": float(np.max(arr)),
    }


def extract_color_statistics(cv_img_bgr: np.ndarray) -> dict[str, float]:
    hsv = cv2.cvtColor(cv_img_bgr, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1].astype(np.float32)
    hue = hsv[:, :, 0].astype(np.float32)
    val = hsv[:, :, 2].astype(np.float32)
    b, g, r = cv2.split(cv_img_bgr.astype(np.float32))
    warmth = float(np.mean(r) - np.mean(b))
    return {
        "saturation_mean": float(np.mean(sat)),
        "saturation_std": float(np.std(sat)),
        "hue_std": float(np.std(hue)),
        "value_std": float(np.std(val)),
        "warmth": warmth,
    }


def frequency_fingerprint(freq_features: dict) -> dict[str, float]:
    """Map DCT / FFT-style features to generator family hints."""
    dct_std = freq_features.get("dct_std", 0.0)
    dct_mean = freq_features.get("dct_mean_abs", 0.0)
    hints = {
        "gan_checkerboard": 0.0,
        "diffusion_smooth": 0.0,
        "midjourney_saturation": 0.0,
    }
    if dct_std > 14:
        hints["gan_checkerboard"] = min((dct_std - 14) * 4, 80)
    if dct_std < 6 and dct_mean < 1.0:
        hints["diffusion_smooth"] = 60
    return hints


def attribute_ai_source(
    exif_data: dict,
    freq_features: dict,
    noise_pattern: dict,
    color_stats: dict,
    c2pa_data: dict,
    ml_result: dict | None = None,
    has_faces: bool = False,
    generative_score: float = 0.0,
) -> list[dict[str, Any]]:
    """
    Rule-based + probabilistic attribution.
    Returns top 3 candidates: [{"tool": str, "confidence": int}, ...]
    """
    confidence_map = {
        "OpenAI DALL-E 3": 0.0,
        "Midjourney": 0.0,
        "Stable Diffusion": 0.0,
        "Adobe Firefly": 0.0,
        "Google Imagen / ImageFX": 0.0,
        "Traditional Software Edit": 0.0,
        "Deepfake (Face Swap)": 0.0,
        "Other / Unknown AI": 0.0,
    }

    detected_sw = exif_data.get("detected_software")
    if detected_sw:
        for key in confidence_map:
            if detected_sw.lower() in key.lower() or key.lower() in detected_sw.lower():
                confidence_map[key] += 85
                break
        if "Stable Diffusion" in (detected_sw or ""):
            confidence_map["Stable Diffusion"] += 80
        if "DALL-E" in (detected_sw or "") or "OpenAI" in (detected_sw or ""):
            confidence_map["OpenAI DALL-E 3"] += 80
        if "Firefly" in (detected_sw or ""):
            confidence_map["Adobe Firefly"] += 85
        if "Midjourney" in (detected_sw or ""):
            confidence_map["Midjourney"] += 85

    if c2pa_data.get("has_c2pa"):
        producer = (c2pa_data.get("c2pa_producer") or "").lower()
        if "adobe" in producer or "firefly" in producer:
            confidence_map["Adobe Firefly"] += 75
        if "openai" in producer:
            confidence_map["OpenAI DALL-E 3"] += 75
        if "google" in producer:
            confidence_map["Google Imagen / ImageFX"] += 70

    sw_tag = (exif_data.get("tags") or {}).get("software", "") or ""
    sw_lower = sw_tag.lower()
    if any(x in sw_lower for x in ("photoshop", "gimp", "lightroom", "snapseed")):
        confidence_map["Traditional Software Edit"] += 65

    # Midjourney: oversaturated, tight hue spread
    if color_stats.get("saturation_mean", 0) > 100 and color_stats.get("saturation_std", 99) < 48:
        confidence_map["Midjourney"] += 35
    if color_stats.get("saturation_mean", 0) > 95:
        confidence_map["Midjourney"] += 15

    # DALL-E 3: smoother gradients, moderate warmth
    if color_stats.get("value_std", 99) < 42 and 5 < color_stats.get("warmth", 0) < 35:
        confidence_map["OpenAI DALL-E 3"] += 25

    # Stable Diffusion: texture noise from SRM
    if noise_pattern.get("srm_std", 0) > 1.2 and noise_pattern.get("srm_mean", 0) < 8:
        confidence_map["Stable Diffusion"] += 30

    freq_hints = frequency_fingerprint(freq_features)
    if freq_hints.get("diffusion_smooth", 0) > 40:
        confidence_map["Stable Diffusion"] += 20
        confidence_map["OpenAI DALL-E 3"] += 15
    if freq_hints.get("gan_checkerboard", 0) > 40:
        confidence_map["Other / Unknown AI"] += 25

    if has_faces and ml_result and generative_score < 52:
        eff = ml_result.get("model_scores", {}).get("efficientnet_deepfake", 0)
        clip_df = ml_result.get("model_scores", {}).get("clip_deepfake", 0)
        if eff >= 58 and clip_df >= 52:
            confidence_map["Deepfake (Face Swap)"] += min(eff * 0.75, 65)

    if exif_data.get("exif_anomaly") and not detected_sw and not c2pa_data.get("has_c2pa"):
        confidence_map["Other / Unknown AI"] += 15

    # Normalize to percentages summing ~100 for top entries
    total = sum(confidence_map.values())
    if total < 1e-3:
        return [{"tool": "Camera Capture / Authentic Photo", "confidence": 100}]

    ranked = sorted(confidence_map.items(), key=lambda x: x[1], reverse=True)
    top3 = []
    for tool, raw in ranked[:3]:
        pct = int(round(100 * raw / total)) if total > 0 else 0
        top3.append({"tool": tool, "confidence": min(max(pct, 0), 100)})

    # Renormalize top-3 to sum to 100
    s = sum(t["confidence"] for t in top3) or 1
    for t in top3:
        t["confidence"] = int(round(t["confidence"] * 100 / s))
    if top3:
        drift = 100 - sum(t["confidence"] for t in top3)
        top3[0]["confidence"] = max(0, min(100, top3[0]["confidence"] + drift))

    top3 = [t for t in top3 if t["confidence"] >= ATTRIBUTION_CONFIDENCE_MIN]
    if not top3:
        return [{"tool": "No high-confidence generator match", "confidence": 0}]
    return top3


def run_attribution(
    layer1: dict,
    ml_result: dict,
    cv_img_bgr: np.ndarray,
    generative_score: float = 0.0,
    hf_ai_score: float = 50.0,
    clip_ai_score: float = 50.0,
) -> list[dict[str, Any]]:
    gray = cv2.cvtColor(cv_img_bgr, cv2.COLOR_BGR2GRAY)
    noise = extract_srm_noise_pattern(gray)
    colors = extract_color_statistics(cv_img_bgr)
    freq = {
        "dct_std": ml_result.get("dct_features", {}).get("std", 0),
        "dct_mean_abs": ml_result.get("dct_features", {}).get("mean_abs", 0),
    }
    exif_data = {
        **layer1.get("exif", {}),
        "detected_software": layer1.get("exif", {}).get("detected_software"),
    }
    if not both_models_agree_ai(hf_ai_score, clip_ai_score):
        c2pa = layer1.get("c2pa", {})
        if c2pa.get("has_c2pa") and c2pa.get("c2pa_producer"):
            return attribute_ai_source(
                exif_data=exif_data,
                freq_features=freq,
                noise_pattern=noise,
                color_stats=colors,
                c2pa_data=c2pa,
                ml_result=ml_result,
                has_faces=ml_result.get("has_faces", False),
                generative_score=generative_score,
            )
        return [{"tool": "Camera Capture / Authentic Photo", "confidence": 100}]

    return attribute_ai_source(
        exif_data=exif_data,
        freq_features=freq,
        noise_pattern=noise,
        color_stats=colors,
        c2pa_data=layer1.get("c2pa", {}),
        ml_result=ml_result,
        has_faces=ml_result.get("has_faces", False),
        generative_score=generative_score,
    )
