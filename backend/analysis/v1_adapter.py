"""Map v2 pipeline output to legacy /api/analyze response shape."""

from __future__ import annotations

from typing import Any


def v2_to_v1_response(v2: dict[str, Any]) -> dict[str, Any]:
    """Convert v2 JSON to v1-compatible payload for older clients."""
    verdict_map = {
        "AUTHENTIC": ("Likely Real Photo", False),
        "AI_GENERATED": ("Likely AI-Generated", True),
        "DEEPFAKE": ("Deepfake Detected", True),
        "SOFTWARE_EDITED": ("Uncertain / Possibly AI", True),
    }
    v2_verdict = v2.get("verdict", "AUTHENTIC")
    ui_verdict, is_ai = verdict_map.get(v2_verdict, ("Uncertain / Possibly AI", False))
    is_deepfake = v2_verdict == "DEEPFAKE"

    dual = v2.get("dual_detection") or {}
    legacy = v2.get("layer_details", {}).get("legacy_signals", {})
    pf = v2.get("layer_details", {}).get("pixel_forensics", {})
    df = v2.get("layer_details", {}).get("deepfake_forensics", {})
    ctx = v2.get("analysis_context", {})
    exif = v2.get("layer_details", {}).get("layer1", {}).get("exif", {})

    synthetic = v2.get("synthetic_likelihood", v2.get("confidence", 0))
    authentic = v2.get("authentic_likelihood", max(0, 100 - synthetic))

    return {
        "is_ai_generated": is_ai,
        "is_deepfake": is_deepfake,
        "confidence": synthetic,
        "synthetic_likelihood": synthetic,
        "authentic_likelihood": authentic,
        "verdict": ui_verdict,
        "dual_detection": dual,
        "signals": {
            "metadata": {
                "score": v2.get("breakdown", {}).get("metadata_score", 50),
                "details": exif.get("details", ""),
                "tags": exif.get("tags", {}),
            },
            "generative": legacy.get("generative", {"score": 50, "details": ""}),
            "frequency": legacy.get("frequency", {"score": 50, "details": ""}),
            "texture": legacy.get("texture", {"score": 50, "details": ""}),
            "compression": legacy.get("compression", {"score": 50, "details": ""}),
            "visual": legacy.get("visual", {"score": 50, "details": ""}),
            "pixel_forensics": {
                "score": v2.get("breakdown", {}).get("pixel_forensics_score", 50),
                "details": "; ".join((pf.get("all_findings") or [])[:4]) or "Pixel forensics computed.",
                "tags": {
                    **(pf.get("signal_scores") or {}),
                    "strong_signals": pf.get("strong_signal_count", 0),
                    "works_without_exif": True,
                },
            },
            "deepfake_forensics": {
                "score": v2.get("breakdown", {}).get("deepfake_score", 0),
                "details": "; ".join((df.get("all_findings") or [])[:4]) or "Face forensics computed.",
                "tags": {
                    "face_count": df.get("face_count", 0),
                    "clip_deepfake": v2.get("breakdown", {}).get("clip_deepfake_score", 0),
                },
            },
            "ml_semantic": {
                "score": v2.get("breakdown", {}).get("hf_ai_score", 50),
                "details": f"HF pretrained: {v2.get('breakdown', {}).get('hf_ai_score', 50):.0f}, CLIP AI: {v2.get('breakdown', {}).get('clip_ai_score', 50):.0f}",
            },
        },
        "heatmap_base64": v2.get("heatmap_base64", ""),
        "fft_base64": v2.get("fft_base64", ""),
        "compression_details": {
            "estimated_quality": ctx.get("estimated_jpeg_quality", 85),
            "analysis_mode": ctx.get("analysis_mode", "standard"),
            "is_compressed_mode": ctx.get("messenger_shared", False),
            "is_pristine_digital": ctx.get("analysis_mode") == "pristine_digital",
            "is_web_downloaded": ctx.get("analysis_mode") == "web_photo",
            "metadata_stripped": ctx.get("metadata_stripped", False),
            "jpeg_blockiness": 0,
            "dynamic_weights": v2.get("layer_details", {}).get("layer2", {}).get("weights", {}),
        },
        "explanation": v2.get("explanation", ""),
        "breakdown": v2.get("breakdown", {}),
    }
