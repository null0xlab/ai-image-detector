"""
Dual-path detection: AI-generated (full image) vs deepfake (face manipulation).

HF ViT is the primary gate for messenger-compressed real photos; CLIP and
spectral forensics only elevate scores when HF is not strongly "real" or when
multiple models agree.
"""

from __future__ import annotations

from typing import Any

from .ensemble_gates import (
    AI_MODEL_AGREE_MIN,
    cap_ai_score_when_hf_real,
    forensic_may_override_hf,
    has_ai_provenance,
    hf_strongly_indicates_real,
    models_strongly_disagree,
    weighted_semantic_score,
)


def _layer_contributions(
    *,
    hf_ai: float,
    clip_ai: float,
    ml_score: float,
    gen_score: float,
    pf_score: float,
    freq_score: float,
    tex_score: float,
    diffusion_robust: float = 0.0,
    weighted_semantic: float = 0.0,
    is_messenger: bool,
) -> dict[str, float]:
    """Per-layer synthetic scores used for UI breakdown."""
    if is_messenger:
        return {
            "weighted_semantic": round(weighted_semantic, 1),
            "compression_robust": round(diffusion_robust, 1),
            "semantic_hf": round(hf_ai, 1),
            "semantic_clip": round(clip_ai, 1),
            "ml_ensemble": round(ml_score, 1),
            "generative_heuristic": round(gen_score, 1),
            "frequency": round(freq_score, 1),
            "texture": round(tex_score, 1),
            "pixel_forensics": round(pf_score, 1),
        }
    return {
        "weighted_semantic": round(weighted_semantic, 1),
        "semantic_hf": round(hf_ai, 1),
        "semantic_clip": round(clip_ai, 1),
        "ml_ensemble": round(ml_score, 1),
        "generative_heuristic": round(gen_score, 1),
        "frequency": round(freq_score, 1),
        "texture": round(tex_score, 1),
        "pixel_forensics": round(pf_score, 1),
    }


def compute_ai_generated_score(
    *,
    hf_ai: float,
    clip_ai: float,
    ml_score: float,
    gen_score: float,
    pf_score: float,
    freq_score: float,
    tex_score: float,
    diffusion_robust: float = 0.0,
    is_messenger: bool,
    has_explicit_ai: bool,
    jpeg_quality: int = 85,
    deepfake_score: float = 0.0,
    watermark_score: float = 0.0,
) -> tuple[float, list[str]]:
    """0-100 likelihood the image is fully AI-generated (not a real photo)."""
    signals: list[str] = []

    if has_explicit_ai:
        return 96.0, ["explicit_ai_metadata"]

    if watermark_score >= 72:
        return round(min(max(watermark_score * 0.95, 80.0), 96.0), 1), ["generator_watermark"]

    if hf_strongly_indicates_real(
        hf_ai,
        deepfake_score,
        clip_ai=clip_ai,
        diffusion_robust=diffusion_robust,
        generative_score=gen_score,
        ml_score=ml_score,
        is_messenger=is_messenger,
        watermark_score=watermark_score,
    ):
        return round(min(max(hf_ai * 1.8, 2.0), 12.0), 1), ["hf_pretrained_real"]

    sem_blend = weighted_semantic_score(hf_ai, clip_ai, jpeg_quality)

    if is_messenger:
        dr_weight = 0.12 if hf_ai < 20 else 0.22
        raw = (
            sem_blend * 0.42
            + hf_ai * 0.18
            + clip_ai * 0.10
            + diffusion_robust * dr_weight
            + ml_score * 0.10
            + gen_score * 0.08
            + pf_score * 0.05
            + freq_score * 0.03
            + tex_score * 0.02
        )
    else:
        raw = (
            sem_blend * 0.38
            + hf_ai * 0.22
            + clip_ai * 0.18
            + ml_score * 0.12
            + gen_score * 0.05
            + pf_score * 0.03
            + freq_score * 0.01
            + tex_score * 0.01
        )

    if hf_ai >= 60:
        signals.append("hf_pretrained_model")
    if clip_ai >= 58:
        signals.append("clip_semantic_ai")
    if gen_score >= 50 and hf_ai >= 20:
        signals.append("generative_heuristic")

    strong_forensics = 0
    if gen_score >= 55:
        strong_forensics += 1
    if pf_score >= 48:
        strong_forensics += 1
    if freq_score >= 48:
        strong_forensics += 1
    if tex_score >= 48:
        strong_forensics += 1
    if diffusion_robust >= 55 and hf_ai >= 20:
        strong_forensics += 1

    if forensic_may_override_hf(
        hf_ai,
        clip_ai,
        generative_score=gen_score,
        diffusion_robust=diffusion_robust,
        strong_forensic_count=strong_forensics,
    ):
        if strong_forensics >= 4:
            raw = max(raw, 72.0)
            signals.append("strong_forensic_consensus")
        elif strong_forensics >= 3 and hf_ai >= 25:
            raw = max(raw, 62.0)
            signals.append("moderate_forensic_consensus")

    if hf_ai >= 55 and clip_ai >= 40:
        raw = max(raw, sem_blend * 0.95)
    elif hf_ai >= AI_MODEL_AGREE_MIN and clip_ai >= AI_MODEL_AGREE_MIN:
        raw = max(raw, sem_blend * 0.98, 58.0)
        signals.append("hf_clip_agreement")

    if ml_score >= 65 and clip_ai >= 50 and hf_ai >= 30:
        raw = max(raw, 65.0)
        signals.append("ml_ensemble")

    raw = cap_ai_score_when_hf_real(
        raw,
        hf_ai,
        deepfake_score,
        clip_ai=clip_ai,
        diffusion_robust=diffusion_robust,
        generative_score=gen_score,
        watermark_score=watermark_score,
        ml_score=ml_score,
        is_messenger=is_messenger,
    )
    return round(min(max(raw, 0.0), 100.0), 1), signals


def compute_deepfake_score(
    *,
    deepfake_ml: float,
    df_forensics: float,
    clip_df: float,
    vis_score: float,
    comp_score: float,
    has_faces: bool,
    is_messenger: bool,
) -> tuple[float, list[str]]:
    """0-100 likelihood of face-swap / face-reenactment manipulation."""
    signals: list[str] = []
    if not has_faces:
        return round(min(clip_df * 0.35, 28.0), 1), []

    if is_messenger:
        raw = (
            deepfake_ml * 0.38
            + df_forensics * 0.32
            + clip_df * 0.22
            + vis_score * 0.08
        )
    else:
        raw = (
            deepfake_ml * 0.40
            + df_forensics * 0.35
            + clip_df * 0.15
            + vis_score * 0.07
            + comp_score * 0.03
        )

    if df_forensics >= 55:
        signals.append("face_boundary_forensics")
    if clip_df >= 55:
        signals.append("clip_deepfake")
    if deepfake_ml >= 58:
        signals.append("face_ml_ensemble")

    return round(min(max(raw, 0.0), 100.0), 1), signals


def fuse_dual_detection(
    *,
    ai_score: float,
    deepfake_score: float,
    has_faces: bool,
    has_explicit_ai: bool,
    hf_ai: float = 50.0,
    clip_ai: float = 50.0,
) -> dict[str, Any]:
    """Final combined verdict from parallel AI + deepfake pipelines."""
    if has_explicit_ai:
        return {
            "verdict": "AI_GENERATED",
            "primary": "ai_generated",
            "ai_generated_score": ai_score,
            "deepfake_score": deepfake_score,
            "combined_synthetic_likelihood": max(ai_score, 94.0),
            "model_disagreement": False,
        }

    if hf_strongly_indicates_real(
        hf_ai,
        deepfake_score,
        clip_ai=clip_ai,
    ):
        authentic_score = round(100.0 - min(ai_score, 15.0), 1)
        return {
            "verdict": "AUTHENTIC",
            "primary": "authentic",
            "ai_generated_score": cap_ai_score_when_hf_real(
                ai_score, hf_ai, deepfake_score, clip_ai=clip_ai
            ),
            "deepfake_score": deepfake_score,
            "combined_synthetic_likelihood": round(min(ai_score, 12.0), 1),
            "model_disagreement": models_strongly_disagree(hf_ai, clip_ai),
        }

    if models_strongly_disagree(hf_ai, clip_ai) and ai_score < 55:
        uncertain = round(max(ai_score, clip_ai * 0.45), 1)
        return {
            "verdict": "UNCERTAIN",
            "primary": "uncertain",
            "ai_generated_score": uncertain,
            "deepfake_score": deepfake_score,
            "combined_synthetic_likelihood": uncertain,
            "model_disagreement": True,
        }

    combined = max(ai_score, deepfake_score if has_faces else 0.0)

    if has_faces and deepfake_score >= 58 and deepfake_score > ai_score + 8 and hf_ai >= 25:
        verdict = "DEEPFAKE"
        primary = "deepfake"
    elif ai_score >= 52 and hf_ai >= AI_MODEL_AGREE_MIN and clip_ai >= AI_MODEL_AGREE_MIN:
        verdict = "AI_GENERATED"
        primary = "ai_generated"
    elif ai_score >= 58 and hf_ai >= 35:
        verdict = "AI_GENERATED"
        primary = "ai_generated"
    elif combined <= 42:
        verdict = "AUTHENTIC"
        primary = "authentic"
    elif combined <= 48 and ai_score < 45 and deepfake_score < 45:
        verdict = "AUTHENTIC"
        primary = "authentic"
    elif models_strongly_disagree(hf_ai, clip_ai):
        verdict = "UNCERTAIN"
        primary = "uncertain"
    else:
        verdict = "AI_GENERATED" if combined > 52 and hf_ai >= 28 else "AUTHENTIC"
        primary = "ai_generated" if verdict == "AI_GENERATED" else "authentic"

    return {
        "verdict": verdict,
        "primary": primary,
        "ai_generated_score": ai_score,
        "deepfake_score": deepfake_score,
        "combined_synthetic_likelihood": round(combined, 1),
        "model_disagreement": models_strongly_disagree(hf_ai, clip_ai),
    }


def run_dual_detection(
    *,
    hf_ai: float,
    clip_ai: float,
    ml_score: float,
    gen_score: float,
    pf_score: float,
    freq_score: float,
    tex_score: float,
    diffusion_robust: float = 0.0,
    deepfake_ml: float,
    df_forensics: float,
    clip_df: float,
    vis_score: float,
    comp_score: float,
    has_faces: bool,
    is_messenger: bool,
    has_explicit_ai: bool,
    jpeg_quality: int = 85,
    hf_available: bool = True,
    provenance_ai: bool = False,
    watermark_score: float = 0.0,
    watermark_method: str | None = None,
) -> dict[str, Any]:
    """Run both pipelines and return scores, signals, layer breakdown, fused verdict."""
    df_score, df_signals = compute_deepfake_score(
        deepfake_ml=deepfake_ml,
        df_forensics=df_forensics,
        clip_df=clip_df,
        vis_score=vis_score,
        comp_score=comp_score,
        has_faces=has_faces,
        is_messenger=is_messenger,
    )
    ai_score, ai_signals = compute_ai_generated_score(
        hf_ai=hf_ai,
        clip_ai=clip_ai,
        ml_score=ml_score,
        gen_score=gen_score,
        pf_score=pf_score,
        freq_score=freq_score,
        tex_score=tex_score,
        diffusion_robust=diffusion_robust,
        is_messenger=is_messenger,
        has_explicit_ai=has_explicit_ai,
        jpeg_quality=jpeg_quality,
        deepfake_score=df_score,
        watermark_score=watermark_score,
    )
    if not hf_available and ai_score < 52 and clip_ai >= 48:
        ai_score = round(max(ai_score, clip_ai * 0.75), 1)
    sem_blend = weighted_semantic_score(hf_ai, clip_ai, jpeg_quality)
    fused = fuse_dual_detection(
        ai_score=ai_score,
        deepfake_score=df_score,
        has_faces=has_faces,
        has_explicit_ai=has_explicit_ai,
        hf_ai=hf_ai,
        clip_ai=clip_ai,
    )

    if (
        hf_available
        and hf_ai < 10
        and clip_ai < 42
        and not has_explicit_ai
        and not provenance_ai
        and watermark_score < 58
        and gen_score < 48
        and ml_score < 52
        and diffusion_robust < 45
    ):
        fused["verdict"] = "AUTHENTIC"
        fused["primary"] = "authentic"
        fused["ai_generated_score"] = cap_ai_score_when_hf_real(
            ai_score,
            hf_ai,
            df_score,
            clip_ai=clip_ai,
            diffusion_robust=diffusion_robust,
            generative_score=gen_score,
            ml_score=ml_score,
            is_messenger=is_messenger,
        )
        fused["combined_synthetic_likelihood"] = fused["ai_generated_score"]
        fused["model_disagreement"] = False

    from .ensemble_gates import messenger_synthetic_consensus

    if (
        is_messenger
        and hf_available
        and hf_ai < 12
        and not has_explicit_ai
        and not provenance_ai
        and messenger_synthetic_consensus(
            hf_ai=hf_ai,
            clip_ai=clip_ai,
            ml_score=ml_score,
            gen_score=gen_score,
            diffusion_robust=diffusion_robust,
            watermark_score=watermark_score,
            watermark_method=watermark_method,
        )
        and fused["verdict"] == "AUTHENTIC"
    ):
        boosted = round(
            max(ai_score, clip_ai * 0.75, watermark_score * 0.9, ml_score * 0.7, 62.0), 1
        )
        fused["verdict"] = "AI_GENERATED"
        fused["primary"] = "ai_generated"
        fused["ai_generated_score"] = boosted
        fused["combined_synthetic_likelihood"] = boosted
        fused["model_disagreement"] = hf_ai < 12 and clip_ai >= 25

    if (
        hf_available
        and hf_ai < 15
        and not has_explicit_ai
        and not provenance_ai
        and watermark_score < 58
        and ml_score >= 52
        and gen_score >= 42
        and fused["verdict"] == "AUTHENTIC"
    ):
        boosted = round(max(ai_score, ml_score * 0.8, gen_score * 0.85, 58.0), 1)
        fused["verdict"] = "AI_GENERATED"
        fused["primary"] = "ai_generated"
        fused["ai_generated_score"] = boosted
        fused["combined_synthetic_likelihood"] = boosted
        fused["model_disagreement"] = True

    if (
        hf_available
        and hf_ai < 10
        and clip_ai >= 48
        and (diffusion_robust >= 45 or gen_score >= 50)
        and fused["verdict"] in ("UNCERTAIN", "AUTHENTIC")
        and not provenance_ai
        and not has_explicit_ai
    ):
        boosted = round(max(ai_score, clip_ai * 0.88, diffusion_robust * 0.82, 58.0), 1)
        fused["verdict"] = "AI_GENERATED"
        fused["primary"] = "ai_generated"
        fused["ai_generated_score"] = boosted
        fused["combined_synthetic_likelihood"] = boosted
        fused["model_disagreement"] = True
    layers = _layer_contributions(
        hf_ai=hf_ai,
        clip_ai=clip_ai,
        ml_score=ml_score,
        gen_score=gen_score,
        pf_score=pf_score,
        freq_score=freq_score,
        tex_score=tex_score,
        diffusion_robust=diffusion_robust,
        weighted_semantic=sem_blend,
        is_messenger=is_messenger,
    )
    return {
        **fused,
        "weighted_semantic_score": sem_blend,
        "ai_signals": ai_signals,
        "deepfake_signals": df_signals,
        "layer_scores": layers,
    }
