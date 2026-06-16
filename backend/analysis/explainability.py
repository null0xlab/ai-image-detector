"""
Layer 4 — GradCAM explainability and human-readable explanations.
"""

from __future__ import annotations

import base64
from typing import Any

import cv2
import numpy as np
from PIL import Image

from .model_registry import ModelRegistry


def generate_gradcam(
    registry: ModelRegistry,
    pil_img: Image.Image,
    fallback_heatmap_b64: str = "",
) -> tuple[str, str | None]:
    """
    GradCAM overlay on the CNN/EfficientNet model when available.
    Returns (heatmap_base64, error_message).
    """
    model = registry.efficientnet_model or registry.cnn_model
    preprocess = registry.efficientnet_preprocess or registry.cnn_preprocess
    if model is None or preprocess is None:
        return fallback_heatmap_b64, "gradcam: no model loaded"

    try:
        import torch
        from pytorch_grad_cam import GradCAM
        from pytorch_grad_cam.utils.image import show_cam_on_image

        rgb = np.array(pil_img.convert("RGB")).astype(np.float32) / 255.0
        rgb_resized = cv2.resize(rgb, (256, 256))
        input_tensor = preprocess(Image.fromarray((rgb_resized * 255).astype(np.uint8)))
        input_tensor = input_tensor.unsqueeze(0).to(registry.device)

        target_layers = []
        for name, module in model.named_modules():
            if "layer4" in name or "blocks" in name:
                target_layers = [module]
        if not target_layers:
            children = list(model.children())
            if children:
                target_layers = [children[-2] if len(children) > 1 else children[-1]]

        cam = GradCAM(model=model, target_layers=target_layers)
        grayscale_cam = cam(input_tensor=input_tensor, targets=None)[0, :]
        visualization = show_cam_on_image(rgb_resized, grayscale_cam, use_rgb=True)
        vis_bgr = cv2.cvtColor((visualization * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        _, buffer = cv2.imencode(".png", vis_bgr)
        b64 = f"data:image/png;base64,{base64.b64encode(buffer).decode('utf-8')}"
        return b64, None
    except ImportError:
        return fallback_heatmap_b64, "gradcam: pytorch-grad-cam not installed"
    except Exception as e:
        return fallback_heatmap_b64, f"gradcam: {e}"


def _is_traditional_software(forensic: dict, exif_tags: dict | None) -> bool:
    sw = (forensic.get("detected_software") or "").lower()
    tag_sw = (exif_tags or {}).get("software", "").lower()
    combined = f"{sw} {tag_sw}"
    return any(
        x in combined
        for x in (
            "photoshop",
            "gimp",
            "lightroom",
            "snapseed",
            "affinity",
            "pixelmator",
            "capture one",
        )
    )


def derive_labels(
    *,
    has_faces: bool,
    metadata_score: float,
    generative_score: float,
    ml_score: float,
    visual_score: float,
    ela_score: float,
    compression_score: float,
    forensic: dict,
    exif_tags: dict | None,
    ai_tool_detected: str | None,
    deepfake_score: float = 0.0,
    clip_deepfake_score: float = 0.0,
) -> dict[str, Any]:
    """
    Multi-label interpretation:
    - Camera-capture authenticity (was this originally a camera photo?)
    - Manipulation types can co-exist: SOFTWARE_EDITED + DEEPFAKE, etc.
    """
    exif_tags = exif_tags or {}
    sw_edit = _is_traditional_software(forensic, exif_tags)
    has_camera = metadata_score <= 18 and bool(exif_tags.get("make"))
    ai_sw = (ai_tool_detected or forensic.get("detected_software") or "").lower()
    is_known_ai_tool = any(
        x in ai_sw
        for x in (
            "dall-e",
            "dalle",
            "midjourney",
            "stable diffusion",
            "firefly",
            "openai",
            "flux",
            "novelai",
        )
    )

    deepfake_signals = 0
    if has_faces:
        if ml_score >= 52 or clip_deepfake_score >= 45:
            deepfake_signals += 1
        if visual_score >= 40 or deepfake_score >= 40:
            deepfake_signals += 1
        if ela_score >= 35 or forensic.get("ela_anomaly_detected"):
            deepfake_signals += 1
        if compression_score >= 40 or forensic.get("double_compression"):
            deepfake_signals += 1
        if deepfake_score >= 55:
            deepfake_signals += 1

    is_deepfake = bool(has_faces and deepfake_signals >= 2 and generative_score < 78 and not is_known_ai_tool)
    is_ai_generated = bool(
        metadata_score >= 100
        or is_known_ai_tool
        or (forensic.get("has_c2pa") and not sw_edit)
        or (generative_score >= 88)
        or (generative_score >= 55 and metadata_score >= 65)
    )

    labels: list[str] = []
    if has_camera and not is_ai_generated:
        labels.append("AUTHENTIC_CAPTURE")
    if sw_edit:
        labels.append("SOFTWARE_EDITED")
    if is_deepfake:
        labels.append("DEEPFAKE")
    if is_ai_generated:
        labels.append("AI_GENERATED")

    # Camera authenticity likelihood: start from EXIF, then subtract strong AI proof
    camera_auth = 15.0
    if has_camera:
        camera_auth = 85.0
    if sw_edit and has_camera:
        camera_auth = 78.0
    if is_ai_generated and not has_camera:
        camera_auth = 10.0
    if is_ai_generated and has_camera:
        camera_auth = min(camera_auth, 55.0)  # could be rephotographed/embedded, be cautious
    if is_deepfake and has_camera:
        camera_auth = min(camera_auth, 70.0)

    return {
        "labels": labels,
        "camera_authenticity_likelihood": round(float(camera_auth), 1),
        "deepfake_signals": int(deepfake_signals),
        "is_known_ai_tool": bool(is_known_ai_tool),
    }


def determine_verdict(
    confidence: float,
    has_faces: bool,
    ml_score: float,
    metadata_score: float,
    generative_score: float,
    forensic: dict,
    *,
    exif_tags: dict | None = None,
    visual_score: float = 0.0,
    ela_score: float = 0.0,
    compression_score: float = 0.0,
    ai_tool_detected: str | None = None,
    is_messenger_shared: bool = False,
    deepfake_score: float = 0.0,
    clip_ai_score: float = 0.0,
    clip_deepfake_score: float = 0.0,
) -> str:
    sw_edit = _is_traditional_software(forensic, exif_tags)
    ai_sw = ai_tool_detected or forensic.get("detected_software") or ""
    is_known_ai_tool = any(
        x in ai_sw.lower()
        for x in (
            "dall-e",
            "dalle",
            "midjourney",
            "stable diffusion",
            "firefly",
            "openai",
            "flux",
            "novelai",
        )
    )

    # --- Tier 1: Definitive metadata signals (highest priority) ---
    if metadata_score >= 100 or is_known_ai_tool:
        return "AI_GENERATED"
    if forensic.get("has_c2pa") and not sw_edit:
        return "AI_GENERATED"

    # --- Tier 2: Strong visual generative evidence ---
    if generative_score >= 88:
        return "AI_GENERATED"
    if generative_score >= 55 and metadata_score >= 65 and confidence >= 50:
        return "AI_GENERATED"

    # --- Tier 3: Deepfake detection (face manipulation signals) ---
    # Use richer set of signals including dedicated deepfake forensics
    deepfake_signals = 0
    if has_faces:
        if ml_score >= 52 or clip_deepfake_score >= 45:
            deepfake_signals += 1
        if visual_score >= 40 or deepfake_score >= 40:
            deepfake_signals += 1
        if ela_score >= 35 or forensic.get("ela_anomaly_detected"):
            deepfake_signals += 1
        if compression_score >= 40 or forensic.get("double_compression"):
            deepfake_signals += 1
        if deepfake_score >= 55:
            deepfake_signals += 1  # strong dedicated deepfake signal counts extra

    # Deepfake verdict: face present, multiple manipulation signals, generative not dominant
    if has_faces and deepfake_signals >= 2 and generative_score < 78 and not is_known_ai_tool:
        # Extra check: if AI generation signals are stronger, call it AI_GENERATED
        if clip_ai_score > clip_deepfake_score * 1.5 and generative_score >= 60:
            return "AI_GENERATED"
        return "DEEPFAKE"

    if has_faces and deepfake_score >= 60 and generative_score < 72:
        return "DEEPFAKE"
    if has_faces and ml_score >= 62 and visual_score >= 35 and generative_score < 75:
        return "DEEPFAKE"

    # --- Tier 4: Traditional software edit ---
    if sw_edit:
        if generative_score < 70 and confidence >= 30:
            return "SOFTWARE_EDITED"
        if ela_score >= 25 or forensic.get("double_compression"):
            return "SOFTWARE_EDITED"

    # --- Tier 5: Messenger-shared (no EXIF) — forensics-only decision ---
    if is_messenger_shared and not is_known_ai_tool:
        # Need strong visual evidence to call non-authentic with stripped metadata
        if generative_score < 42 and ml_score < 58 and confidence < 55 and deepfake_score < 40:
            return "AUTHENTIC"
        if generative_score >= 58 and (ml_score >= 52 or visual_score >= 40):
            # Is it AI-generated or deepfake?
            if has_faces and deepfake_score >= 45 and generative_score < 75:
                return "DEEPFAKE"
            return "AI_GENERATED"
        # Deepfake with missing metadata
        if has_faces and deepfake_score >= 50 and ml_score >= 45:
            return "DEEPFAKE"

    # --- Tier 6: Camera EXIF present — lean authentic unless strong proof ---
    has_camera = metadata_score <= 18 and bool((exif_tags or {}).get("make"))
    if has_camera:
        if not is_known_ai_tool and generative_score < 65 and ml_score < 65 and deepfake_score < 50:
            return "AUTHENTIC"
        if not is_known_ai_tool and confidence < 72 and deepfake_score < 55:
            return "AUTHENTIC"

    # --- Tier 7: Score-based fallback ---
    if confidence >= 58:
        if has_faces and (deepfake_score >= 45 or ml_score >= 48) and generative_score < 80:
            return "DEEPFAKE"
        return "AI_GENERATED"

    if confidence >= 40:
        if sw_edit and generative_score < 55:
            return "SOFTWARE_EDITED"
        if has_faces and (deepfake_score >= 38 or ml_score >= 45) and generative_score < 70:
            return "DEEPFAKE"
        return "AI_GENERATED"

    # Camera EXIF present — trust authentic
    if has_camera:
        if not is_known_ai_tool and generative_score < 62 and ml_score < 62:
            return "AUTHENTIC"
        if not is_known_ai_tool and confidence < 68:
            return "AUTHENTIC"

    if confidence <= 22 and not forensic.get("exif_anomaly"):
        return "AUTHENTIC"
    if confidence < 32 and generative_score < 35 and ml_score < 40 and deepfake_score < 30:
        return "AUTHENTIC"
    if sw_edit:
        return "SOFTWARE_EDITED"
    return "AUTHENTIC"


def build_explanation(
    verdict: str,
    confidence: float,
    breakdown: dict,
    forensic_signals: dict,
    ai_attribution: list,
    model_errors: dict,
    *,
    model_disagreement: bool = False,
) -> str:
    parts = [f"Overall confidence in this classification: {confidence:.0f}%."]
    parts.append(f"Verdict: {verdict.replace('_', ' ').title()}.")

    if forensic_signals.get("has_c2pa"):
        prod = forensic_signals.get("c2pa_producer") or "unknown producer"
        parts.append(f"Content Credentials (C2PA) manifest detected from {prod}.")
    if forensic_signals.get("generator_watermark"):
        prov = forensic_signals.get("watermark_provider") or "a generative AI tool"
        parts.append(
            f"A visible Gemini-style sparkle watermark consistent with {prov} was detected "
            "(common on Gemini exports even after messenger re-sharing)."
        )
    elif forensic_signals.get("watermark_suspect"):
        parts.append(
            "A possible corner watermark was noted but neural classifiers indicate a natural "
            "photograph — the watermark cue was not used for the final verdict."
        )
    if forensic_signals.get("messenger_shared"):
        parts.append(
            "Image was likely shared via a messaging app (Telegram/WhatsApp/Messenger); "
            "metadata was stripped by the platform — the pretrained HF pixel classifier is weighted "
            "heavily in this mode to avoid false positives from compression artifacts."
        )
    if model_disagreement:
        parts.append(
            "Models disagree: the Hugging Face classifier indicates a real photograph while "
            "semantic or spectral heuristics show mixed signals — treat this result with caution "
            "for messenger-compressed images."
        )
    hf = breakdown.get("hf_ai_score", 0)
    clip = breakdown.get("clip_ai_score", 0)
    if hf > 0:
        parts.append(f"HF ViT AI score: {hf:.0f}%. CLIP semantic AI score: {clip:.0f}%.")
    elif forensic_signals.get("exif_anomaly"):
        parts.append("EXIF metadata is missing typical camera fields or shows generator software tags.")
    if forensic_signals.get("ela_anomaly_detected"):
        parts.append(
            "Error Level Analysis shows uneven compression — common in spliced or re-saved regions."
        )
    if forensic_signals.get("double_compression"):
        parts.append("Double JPEG compression artifacts suggest the image was saved more than once.")

    ml = breakdown.get("ml_ensemble_score", 0)
    df_score = breakdown.get("deepfake_score", 0)
    if verdict == "DEEPFAKE":
        parts.append(
            f"Face manipulation signals detected: deepfake forensics score={df_score:.0f}/100, "
            f"ML ensemble score={ml:.0f}/100."
        )
    elif ml >= 55:
        parts.append(f"The deep-learning ensemble scored {ml:.0f}/100 toward synthetic content.")
    elif ml <= 30:
        parts.append("Neural detectors found patterns consistent with a natural photograph.")

    pf = breakdown.get("pixel_forensics_score", 0)
    if pf >= 50:
        parts.append(
            f"Pixel-level forensics (PRNU/SRM/wavelet/chromatic aberration) scored {pf:.0f}/100 "
            "toward synthetic origin — analysis is metadata-independent."
        )

    if ai_attribution and verdict in ("AI_GENERATED", "DEEPFAKE"):
        top = ai_attribution[0]
        if top.get("confidence", 0) >= 70:
            parts.append(
                f"Most likely source: {top['tool']} ({top['confidence']}% attribution confidence)."
            )

    if model_errors:
        parts.append(
            "Note: some models were unavailable — "
            + ", ".join(f"{k}: {v}" for k, v in list(model_errors.items())[:3])
            + "."
        )

    return " ".join(parts)
