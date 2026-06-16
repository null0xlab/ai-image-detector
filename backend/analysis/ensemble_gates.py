"""
Central gates for HF/CLIP ensemble — prevents false positives on messenger-compressed real photos.
"""

from __future__ import annotations

HF_STRONGLY_REAL_MAX = 10.0
DEEPFAKE_REAL_MAX = 30.0
AI_MODEL_AGREE_MIN = 40.0
ATTRIBUTION_CONFIDENCE_MIN = 70
DISAGREE_HF_MAX = 15.0
DISAGREE_CLIP_MIN = 45.0

_AI_PROVENANCE_KEYWORDS = (
    "google",
    "gemini",
    "imagen",
    "openai",
    "dall-e",
    "dalle",
    "chatgpt",
    "adobe firefly",
    "firefly",
    "midjourney",
    "stable diffusion",
    "flux",
    "sora",
    "vertex",
)


def neural_models_indicate_real(hf_ai: float, clip_ai: float) -> bool:
    """HF fooled but CLIP still low — typical of real messenger photos, not GenAI."""
    return hf_ai < 12.0 and clip_ai < 36.0


def messenger_genai_fingerprint(
    hf_ai: float, clip_ai: float, diffusion_robust: float
) -> bool:
    """
  HF ~0–2%, CLIP ~30–33%, compression-robust ~39+ on Telegram-shared Gemini exports.
  Real messenger photos usually have CLIP >= 34 or CR < 39.
    """
    return (
        hf_ai < 3.0
        and 29.0 <= clip_ai <= 33.5
        and diffusion_robust >= 39.0
    )


def watermark_is_definitive(
    *,
    watermark_detected: bool = False,
    watermark_score: float = 0.0,
    has_star_mark: bool = False,
    watermark_confidence: str = "none",
    watermark_method: str | None = None,
    hf_ai: float = 50.0,
    clip_ai: float = 50.0,
) -> bool:
    """
    Only byte hints or bright corner sparkles may override neural 'real' scores.
    Faint post-Telegram sparkle detection is a consensus vote only — never alone.
    """
    if not watermark_detected:
        return False
    if watermark_method == "faint_sparkle":
        return False
    if neural_models_indicate_real(hf_ai, clip_ai):
        return False
    if watermark_method == "byte_hint":
        return True
    if watermark_method == "bright_sparkle" and has_star_mark and watermark_score >= 58:
        return True
    return watermark_confidence == "high" and watermark_score >= 72


def has_ai_provenance(
    *,
    has_c2pa: bool = False,
    c2pa_producer: str | None = None,
    ai_tool_used: str | None = None,
    metadata_score: float = 0.0,
    generative_details: str = "",
    watermark_detected: bool = False,
    watermark_provider: str | None = None,
    watermark_score: float = 0.0,
    has_star_mark: bool = False,
    watermark_confidence: str = "none",
    watermark_method: str | None = None,
    hf_ai: float = 50.0,
    clip_ai: float = 50.0,
) -> bool:
    """Definitive metadata / C2PA proof the image is from a GenAI tool."""
    if watermark_is_definitive(
        watermark_detected=watermark_detected,
        watermark_score=watermark_score,
        has_star_mark=has_star_mark,
        watermark_confidence=watermark_confidence,
        watermark_method=watermark_method,
        hf_ai=hf_ai,
        clip_ai=clip_ai,
    ):
        return True
    if metadata_score >= 100:
        return True
    blob = " ".join(
        str(x or "") for x in (c2pa_producer, ai_tool_used, generative_details, watermark_provider)
    ).lower()
    if has_c2pa and any(k in blob for k in _AI_PROVENANCE_KEYWORDS):
        return True
    if any(k in (generative_details or "").lower() for k in ("gemini", "google imagen", "dall-e")):
        return True
    return False


def weighted_semantic_score(hf_ai: float, clip_ai: float, jpeg_quality: int = 85) -> float:
    """Blend HF ViT + CLIP; trust HF more on high-quality shares when it says real."""
    hf = float(hf_ai)
    clip = float(clip_ai)
    if hf < 10.0 and jpeg_quality > 85:
        return round(hf * 0.75 + clip * 0.25, 2)
    return round(hf * 0.40 + clip * 0.60, 2)


def messenger_synthetic_consensus(
    *,
    hf_ai: float,
    clip_ai: float,
    ml_score: float,
    gen_score: float,
    diffusion_robust: float,
    watermark_score: float = 0.0,
    watermark_method: str | None = None,
) -> bool:
    """
    Telegram strips C2PA; use combined weak signals when HF is fooled on GenAI photos.
    Does not fire on messenger real photos where HF+CLIP both indicate authentic.
    """
    if hf_ai >= 18:
        return False
    if messenger_genai_fingerprint(hf_ai, clip_ai, diffusion_robust):
        return True
    votes = 0
    if clip_ai >= 30:
        votes += 1
    if ml_score >= 48:
        votes += 1
    if gen_score >= 40:
        votes += 1
    if diffusion_robust >= 38:
        votes += 1
    if watermark_score >= 50 and watermark_method != "faint_sparkle":
        votes += 1
    return votes >= 2


def hf_strongly_indicates_real(
    hf_ai: float,
    deepfake_score: float,
    *,
    clip_ai: float = 0.0,
    diffusion_robust: float = 0.0,
    generative_score: float = 0.0,
    hf_available: bool = True,
    watermark_score: float = 0.0,
    ml_score: float = 50.0,
    is_messenger: bool = False,
) -> bool:
    """
    Trust HF "real" only when companion signals do not indicate synthetic content.
    Prevents false positives on messenger real photos while still catching AI when
    CLIP/forensics disagree with a fooled HF ViT.
    """
    if watermark_score >= 68:
        return False
    if not hf_available or hf_ai >= HF_STRONGLY_REAL_MAX:
        return False
    if deepfake_score >= DEEPFAKE_REAL_MAX:
        return False
    if is_messenger and messenger_synthetic_consensus(
        hf_ai=hf_ai,
        clip_ai=clip_ai,
        ml_score=ml_score,
        gen_score=generative_score,
        diffusion_robust=diffusion_robust,
        watermark_score=watermark_score,
    ):
        return False
    if clip_ai >= 50 or diffusion_robust >= 55 or generative_score >= 55:
        return False
    if clip_ai >= 45 and (diffusion_robust >= 48 or generative_score >= 50):
        return False
    if clip_ai >= DISAGREE_CLIP_MIN and hf_ai < DISAGREE_HF_MAX:
        return False
    return True


def both_models_agree_ai(hf_ai: float, clip_ai: float) -> bool:
    return hf_ai >= AI_MODEL_AGREE_MIN and clip_ai >= AI_MODEL_AGREE_MIN


def models_strongly_disagree(hf_ai: float, clip_ai: float) -> bool:
    """Pixel classifier says real; semantic path says synthetic."""
    return hf_ai < DISAGREE_HF_MAX and clip_ai >= DISAGREE_CLIP_MIN


def forensic_may_override_hf(
    hf_ai: float,
    clip_ai: float,
    *,
    generative_score: float,
    diffusion_robust: float,
    strong_forensic_count: int,
) -> bool:
    """
    Allow classical/spectral paths to raise AI score only when HF is not strongly real
    OR multiple independent paths agree.
    """
    if hf_strongly_indicates_real(hf_ai, 99.0, clip_ai=clip_ai, diffusion_robust=diffusion_robust):
        return False
    if both_models_agree_ai(hf_ai, clip_ai):
        return True
    if hf_ai >= 25:
        return True
    if strong_forensic_count >= 4 and generative_score >= 55 and diffusion_robust >= 55:
        return True
    return False


def cap_ai_score_when_hf_real(
    ai_score: float,
    hf_ai: float,
    deepfake_score: float,
    *,
    clip_ai: float = 0.0,
    diffusion_robust: float = 0.0,
    generative_score: float = 0.0,
    watermark_score: float = 0.0,
    ml_score: float = 50.0,
    is_messenger: bool = False,
) -> float:
    if hf_strongly_indicates_real(
        hf_ai,
        deepfake_score,
        clip_ai=clip_ai,
        diffusion_robust=diffusion_robust,
        generative_score=generative_score,
        watermark_score=watermark_score,
        ml_score=ml_score,
        is_messenger=is_messenger,
    ):
        return round(min(ai_score, max(hf_ai * 2.0, 8.0)), 1)
    return ai_score


def finalize_display_scores(
    verdict: str,
    *,
    ai_generated_score: float,
    synthetic_likelihood: float,
    explicit_ai: bool,
    provenance_ai: bool,
) -> tuple[float, float, float]:
    """
    Align synthetic_likelihood, UI confidence, and dual-card AI score.
    Returns (synthetic_likelihood, confidence_for_api, aligned_ai_card_score).
    """
    ai = float(ai_generated_score)
    syn = float(synthetic_likelihood)

    if verdict == "AI_GENERATED":
        floor = 88.0 if (explicit_ai or provenance_ai) else 58.0
        aligned = min(max(ai, syn, floor), 98.0)
        return round(aligned, 1), round(aligned, 1), round(aligned, 1)

    if verdict == "DEEPFAKE":
        aligned = min(max(ai, syn, 62.0), 96.0)
        return round(aligned, 1), round(aligned, 1), round(aligned, 1)

    if verdict == "UNCERTAIN":
        aligned = round(min(max(ai, syn * 0.6, 35.0), 48.0), 1)
        return aligned, aligned, aligned

    # AUTHENTIC
    aligned = round(min(ai, syn, 22.0), 1)
    auth_conf = round(100.0 - aligned, 1)
    return aligned, auth_conf, aligned
