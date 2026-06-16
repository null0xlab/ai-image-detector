from __future__ import annotations

from typing import Any


GENERATOR_HINTS = {
    "gan_spectrum": "GAN / StyleGAN (likely)",
    "diffusion_smooth_spectrum": "Stable Diffusion / Flux (likely)",
    "clip_semantic_ai": "Midjourney / Stable Diffusion (likely)",
    "hf_ai_portrait": "AI portrait generator (likely)",
    "document_ai_classifier": "AI-generated document (likely)",
    "sky_artifact": "Diffusion scene generator (likely)",
    "text_artifacts": "AI document renderer (likely)",
}


def _infer_generator_hint(signals: list[str], c2pa_data: dict | None) -> str | None:
    if c2pa_data:
        tool = c2pa_data.get("ai_tool_used") or c2pa_data.get("c2pa_producer")
        if tool:
            return str(tool)
    for signal in signals:
        if signal in GENERATOR_HINTS:
            return GENERATOR_HINTS[signal]
    if "frequency_anomaly" in signals:
        return "Unknown AI generator (frequency fingerprint)"
    return None


def _confidence_label(ai_score: float, signal_count: int) -> str:
    if ai_score >= 0.75 and signal_count >= 3:
        return "high"
    if ai_score >= 0.55 or signal_count >= 2:
        return "medium"
    if ai_score <= 0.35:
        return "low"
    return "medium"


def run_ensemble(
    *,
    section: str,
    section_result: dict[str, Any],
    universal_scores: dict[str, float],
    metadata_result: dict[str, Any],
    edit_result: dict[str, Any],
    possibly_recompressed: bool,
    signals: list[str] | None = None,
) -> dict[str, Any]:
    signals = list(signals or [])
    signals.extend(section_result.get("signals", []))
    signals.extend(edit_result.get("signals", []))
    if metadata_result.get("suspicious"):
        signals.append("missing_exif" if not metadata_result.get("has_camera_exif") else "metadata_suspicious")
    if metadata_result.get("c2pa_data", {}).get("has_c2pa"):
        signals.append("c2pa_present")

    hf_score = universal_scores.get("hf_ai", section_result.get("model_scores", {}).get("hf_ai", 50.0))
    eff_score = universal_scores.get(
        "efficientnet",
        section_result.get("model_scores", {}).get("efficientnet", 50.0),
    )
    clip_score = universal_scores.get(
        "clip",
        section_result.get("model_scores", {}).get("clip_semantic", section_result.get("model_scores", {}).get("clip_ai", 50.0)),
    )
    freq_score = universal_scores.get(
        "frequency",
        section_result.get("model_scores", {}).get("frequency", 50.0),
    )
    metadata_score = metadata_result.get("metadata_score", 50.0)

    w_hf = 1.3
    w_eff = 1.0
    w_clip = 1.5 if possibly_recompressed else 0.8
    w_freq = 1.0 if not possibly_recompressed else 0.4
    w_meta = 0.5

    weighted_sum = (
        hf_score * w_hf
        + eff_score * w_eff
        + clip_score * w_clip
        + freq_score * w_freq
        + metadata_score * w_meta
    )
    weight_total = w_hf + w_eff + w_clip + w_freq + w_meta
    section_percent = float(section_result.get("score_percent", section_result.get("score", 0.5) * 100))
    blended = 0.55 * (weighted_sum / weight_total) + 0.45 * section_percent
    ai_score = round(min(max(blended / 100.0, 0.0), 1.0), 4)

    edit_score = float(edit_result.get("edit_score", 0.0))
    generator_hint = _infer_generator_hint(signals, metadata_result.get("c2pa_data"))
    confidence = _confidence_label(ai_score, len(set(signals)))

    return {
        "ai_score": ai_score,
        "edit_score": edit_score,
        "section": section,
        "generator_hint": generator_hint,
        "signals": sorted(set(signals)),
        "confidence": confidence,
        "weights": {
            "hf_ai": w_hf,
            "efficientnet": w_eff,
            "clip": w_clip,
            "frequency": w_freq,
            "metadata": w_meta,
        },
        "model_scores": {
            "hf_ai": round(float(hf_score), 2),
            "efficientnet": round(float(eff_score), 2),
            "clip": round(float(clip_score), 2),
            "frequency": round(float(freq_score), 2),
            "metadata": round(float(metadata_score), 2),
            "section": round(section_percent, 2),
        },
    }
