"""
Evidence-based fusion for 3-class verdicts.

HF ViT gate: when HF &lt; 10% and deepfake &lt; 30%, do not let spectral/CLIP paths
override to a false AI verdict on messenger-compressed real photos.
"""

from __future__ import annotations

from .ensemble_gates import (
    AI_MODEL_AGREE_MIN,
    both_models_agree_ai,
    has_ai_provenance,
    hf_strongly_indicates_real,
    models_strongly_disagree,
    weighted_semantic_score,
)


def count_strong_signals(scores: dict[str, float], ai_threshold: float = 52.0) -> int:
    return sum(1 for v in scores.values() if v >= ai_threshold)


def count_authentic_signals(scores: dict[str, float], auth_threshold: float = 38.0) -> int:
    return sum(1 for v in scores.values() if v <= auth_threshold)


def check_explicit_ai_proof(
    *,
    metadata_score: float,
    generative_score: float,
    generative_details: str = "",
    forensic: dict | None = None,
) -> bool:
    forensic = forensic or {}
    details = (generative_details or "").lower()
    if metadata_score >= 100:
        return True
    if any(
        k in details
        for k in (
            "dall-e",
            "dalle",
            "midjourney",
            "openai",
            "embedded hint",
            "chatgpt",
            "gemini",
            "google imagen",
            "imagen",
        )
    ):
        return True
    if has_ai_provenance(
        has_c2pa=bool(forensic.get("has_c2pa")),
        c2pa_producer=forensic.get("c2pa_producer"),
        ai_tool_used=forensic.get("ai_tool_used"),
        metadata_score=metadata_score,
        generative_details=generative_details,
    ):
        return True
    return generative_score >= 95


def compute_synthetic_likelihood(
    *,
    weighted_score: float,
    generative_score: float,
    metadata_score: float,
    pf_score: float,
    npr_score: float,
    is_messenger: bool,
    has_explicit_ai: bool,
    hf_ai_score: float = 0.0,
    clip_ai_score: float = 0.0,
    diffusion_robust_score: float = 0.0,
    dual_combined: float | None = None,
    deepfake_score: float = 0.0,
    jpeg_quality: int = 85,
    provenance_ai: bool = False,
) -> float:
    """0–100: higher = more likely synthetic / AI."""
    hf = float(hf_ai_score)
    clip = float(clip_ai_score)
    dr = float(diffusion_robust_score)

    raw = float(weighted_score)
    if dual_combined is not None:
        raw = max(raw, float(dual_combined))

    if has_explicit_ai or provenance_ai:
        return round(min(max(raw, 90.0), 98.0), 1)

    if hf_strongly_indicates_real(
        hf,
        deepfake_score,
        clip_ai=clip,
        diffusion_robust=dr,
        generative_score=generative_score,
        is_messenger=is_messenger,
    ):
        return round(min(max(hf * 1.5, 3.0), 12.0), 1)

    sem_blend = weighted_semantic_score(hf, clip, jpeg_quality)
    semantic = sem_blend

    if both_models_agree_ai(hf, clip):
        raw = max(raw, sem_blend * 0.96, 62.0)
    elif hf >= 55 and clip >= 35:
        raw = max(raw, sem_blend * 0.92)
    elif semantic >= 68 and hf >= 30:
        raw = max(raw, 70.0, semantic * 0.90)

    if is_messenger:
        if both_models_agree_ai(hf, clip) and dr >= 48:
            raw = max(raw, dr * 0.85)
        elif dr >= 58 and hf >= 35 and clip >= 35:
            raw = max(raw, dr * 0.88)
        if models_strongly_disagree(hf, clip):
            raw = min(raw, 48.0)
    else:
        if generative_score < 40 and hf < 42 and pf_score < 40:
            factor = (generative_score / 40.0) * (hf / 42.0)
            raw = min(raw, 45.0) * (0.4 + 0.6 * factor)

    return round(min(max(raw, 0.0), 100.0), 1)


def messenger_verdict_from_likelihood(
    synthetic_likelihood: float,
    *,
    strong_ai_signals: int,
    strong_auth_signals: int,
    has_explicit_ai: bool,
    generative_score: float,
    hf_ai_score: float = 0.0,
    clip_ai_score: float = 0.0,
    deepfake_score: float = 0.0,
) -> tuple[str, bool]:
    """Returns (verdict_string_v1_style, is_ai_generated_bool)."""
    if has_explicit_ai:
        return "Likely AI-Generated", True

    hf = float(hf_ai_score)
    clip = float(clip_ai_score)

    if hf_strongly_indicates_real(
        hf, deepfake_score, clip_ai=clip, generative_score=generative_score
    ):
        return "Likely Real Photo", False

    if both_models_agree_ai(hf, clip) and synthetic_likelihood >= 52:
        return "Likely AI-Generated", True

    if models_strongly_disagree(hf, clip) and hf < AI_MODEL_AGREE_MIN:
        return "Uncertain / Possibly AI", True

    if synthetic_likelihood <= 42 and strong_auth_signals >= 2 and hf < 42:
        return "Likely Real Photo", False
    if synthetic_likelihood <= 45 and hf < 45 and strong_ai_signals < 2:
        return "Likely Real Photo", False

    if synthetic_likelihood > 52 and both_models_agree_ai(hf, clip):
        return "Likely AI-Generated", True
    if synthetic_likelihood > 50 and hf >= 35:
        return "Uncertain / Possibly AI", True
    return "Likely Real Photo", False


def v2_verdict_from_likelihood(
    synthetic_likelihood: float,
    *,
    has_faces: bool,
    deepfake_score: float,
    generative_score: float,
    is_messenger: bool,
    has_explicit_ai: bool,
    strong_ai_signals: int,
    strong_auth_signals: int,
    ai_generated_score: float = 0.0,
    hf_ai_score: float = 0.0,
    clip_ai_score: float = 0.0,
    diffusion_robust_score: float = 0.0,
) -> str:
    if has_explicit_ai:
        return "AI_GENERATED"

    hf = float(hf_ai_score)
    clip = float(clip_ai_score)

    if hf_strongly_indicates_real(
        hf,
        deepfake_score,
        clip_ai=clip,
        diffusion_robust=diffusion_robust_score,
        generative_score=generative_score,
    ):
        return "AUTHENTIC"

    if models_strongly_disagree(hf, clip) and not both_models_agree_ai(hf, clip):
        if synthetic_likelihood < 55 and hf < AI_MODEL_AGREE_MIN:
            return "UNCERTAIN"

    if both_models_agree_ai(hf, clip) and (
        ai_generated_score >= 52 or synthetic_likelihood >= 55
    ):
        return "AI_GENERATED"

    if has_faces and deepfake_score >= 60 and deepfake_score > ai_generated_score + 10:
        if generative_score < 55 and hf < 50:
            return "DEEPFAKE"

    if ai_generated_score >= 58 and hf >= 35 and clip >= 35:
        return "AI_GENERATED"

    if is_messenger:
        if both_models_agree_ai(hf, clip) and synthetic_likelihood >= 52:
            return "AI_GENERATED"
        if synthetic_likelihood <= 42 and strong_auth_signals >= 2 and hf < 42:
            return "AUTHENTIC"
        if hf < 10 and deepfake_score < 30:
            return "AUTHENTIC"
        if models_strongly_disagree(hf, clip):
            return "UNCERTAIN"
        return "AUTHENTIC" if synthetic_likelihood <= 48 and hf < 35 else "UNCERTAIN"

    if synthetic_likelihood >= 55 and hf >= 40:
        return "AI_GENERATED"
    if synthetic_likelihood <= 38:
        return "AUTHENTIC"
    return "AUTHENTIC" if synthetic_likelihood < 50 else "UNCERTAIN"
