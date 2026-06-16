"""
4-layer hybrid detection pipeline orchestrator for /api/v2/analyze.

Layers:
  1 — Forensic triage (EXIF, ELA, C2PA)
  2 — ML ensemble (HF pretrained + CLIP + CNN + frequency + deepfake)
  3 — Dual detection (AI-generated vs deepfake, parallel fusion)
  4 — Explainability (GradCAM, natural language)
"""

from __future__ import annotations

from typing import Any

from . import (
    analyze_compression,
    analyze_frequency,
    analyze_generative,
    analyze_metadata,
    analyze_texture,
    analyze_visual_and_heatmap,
    classify_upload_context,
    estimate_jpeg_quality,
    load_image,
    preprocess_and_denoise,
    normalize_compressed_image,
    to_json_safe,
)
from .attribution import run_attribution
from .calibration import platt_calibrate, synthetic_and_authentic_likelihood
from .compression_robust import run_compression_robust_analysis
from .generator_watermark import detect_generator_watermark
from .deepfake_forensics import run_deepfake_forensics
from .dual_detection import run_dual_detection
from .ensemble_gates import (
    finalize_display_scores,
    has_ai_provenance,
    messenger_genai_fingerprint,
    neural_models_indicate_real,
)
from .evidence_fusion import (
    check_explicit_ai_proof,
    count_authentic_signals,
    count_strong_signals,
    compute_synthetic_likelihood,
    v2_verdict_from_likelihood,
)
from .explainability import build_explanation, derive_labels, determine_verdict, generate_gradcam
from .layer1_forensic import run_layer1_forensic
from .ml_ensemble import run_ml_ensemble
from .model_registry import ModelRegistry
from .pixel_forensics import run_pixel_forensics


def run_v2_pipeline(image_bytes: bytes, filename: str, registry: ModelRegistry | None = None) -> dict[str, Any]:
    """Execute all layers and return structured v2 JSON with dual AI/deepfake scores."""
    registry = registry or ModelRegistry.get_instance()

    pil_img, cv_img = load_image(image_bytes)
    ext = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ".jpg"
    if not ext.startswith("."):
        ext = f".{ext}"

    quality = estimate_jpeg_quality(pil_img)
    has_exif = hasattr(pil_img, "_getexif") and pil_img._getexif() is not None
    ctx = classify_upload_context(cv_img, ext, has_exif, quality)
    metadata_stripped = bool(ctx.get("metadata_stripped", not has_exif))
    # Messenger mode only for re-compressed JPEG shares — not every EXIF-less upload
    is_msg_mode = bool(ctx["is_messaging_compressed"])
    forensics_compressed = is_msg_mode or (
        metadata_stripped
        and ext in (".jpg", ".jpeg")
        and quality < 88
        and not ctx.get("is_pristine_digital")
    )

    cv_denoised = preprocess_and_denoise(cv_img, quality)

    metadata_res = analyze_metadata(
        image_bytes,
        is_messaging_compressed=ctx["is_messaging_compressed"],
        is_pristine_digital=ctx["is_pristine_digital"],
        is_web_downloaded=ctx["is_web_downloaded"],
    )
    generative_res = analyze_generative(
        cv_img, pil_img, image_bytes, is_messaging_compressed=forensics_compressed
    )
    watermark_res = detect_generator_watermark(cv_img, image_bytes)
    wm_score = float(watermark_res.get("score", 0))
    wm_method = watermark_res.get("method")
    wm_definitive = bool(watermark_res.get("definitive"))
    wm_confident = wm_definitive
    if wm_confident:
        generative_res = {
            **generative_res,
            "score": max(float(generative_res["score"]), wm_score, 72.0),
            "details": (
                generative_res.get("details", "")
                + f" Visible generator watermark ({watermark_res.get('provider', 'unknown')}"
                f" corner={watermark_res.get('corner', '')})."
            ).strip(),
        }
    layer1 = run_layer1_forensic(
        image_bytes,
        pil_img,
        cv_img,
        metadata_res,
        is_messaging_compressed=ctx["is_messaging_compressed"],
        is_pristine_digital=ctx["is_pristine_digital"],
        is_web_downloaded=ctx["is_web_downloaded"],
    )

    frequency_res = analyze_frequency(cv_denoised)
    texture_res = analyze_texture(
        cv_denoised,
        is_web_downloaded=ctx["is_web_downloaded"],
        is_messaging_compressed=is_msg_mode,
    )
    compression_res = analyze_compression(
        pil_img, cv_denoised, is_lossless_origin=ctx["is_lossless_origin"]
    )
    visual_res, visual_heatmap = analyze_visual_and_heatmap(
        cv_img, compression_res["score"], texture_res["score"]
    )

    pixel_forensics_res = run_pixel_forensics(
        cv_img,
        is_messaging_compressed=forensics_compressed,
        n_crops=3 if metadata_stripped else 2,
    )
    pf_score = float(pixel_forensics_res["score"])

    deepfake_forensics_res = run_deepfake_forensics(cv_img, registry.face_cascade)
    df_forensics_score = float(deepfake_forensics_res["score"])

    forensic_hint = (
        float(generative_res["score"]) * 0.28
        + float(frequency_res["score"]) * 0.20
        + float(texture_res["score"]) * 0.17
        + float(visual_res["score"]) * 0.15
        + pf_score * 0.13
        + df_forensics_score * 0.07
    )

    # Bypass manual sharpening for deep learning models to avoid distorting noise structures
    cv_img_dl = cv_img.copy()
    pil_img_dl = pil_img

    ml_result = run_ml_ensemble(
        pil_img_dl,
        cv_img_dl,
        registry=registry,
        forensic_hint=forensic_hint,
        pixel_forensics_score=pf_score,
        is_messaging_compressed=is_msg_mode,
    )

    metadata_score = layer1["metadata_score"]
    ela_score = float(layer1["ela"].get("ela_score", 50))
    exif_tags = layer1["exif"].get("tags") or metadata_res.get("tags") or {}
    has_camera_exif = metadata_score <= 20 and bool(exif_tags.get("make"))
    if has_camera_exif:
        ela_score = min(ela_score, 28.0)

    gen_score = float(generative_res["score"])
    tex_score = float(texture_res["score"])
    vis_score = float(visual_res["score"])
    comp_score = float(compression_res["score"])
    forensic_freq = float(frequency_res["score"])

    ml_score = float(ml_result["ml_ensemble_score"])
    freq_ml_score = float(ml_result["frequency_score"])
    clip_ai_score = float(ml_result.get("clip_ai_score", 0.0))
    clip_df_score = float(ml_result.get("clip_deepfake_score", 0.0))
    hf_ai_score = float(ml_result.get("hf_ai_score", 50.0))
    deepfake_ml_score = float(ml_result.get("deepfake_ml_score", 0.0))

    compression_robust_res = run_compression_robust_analysis(
        cv_img,
        is_messaging_compressed=is_msg_mode or forensics_compressed,
        hf_ai_score=hf_ai_score,
    )
    cr_score = float(compression_robust_res["score"])

    combined_deepfake_score = float(
        deepfake_ml_score * 0.40
        + df_forensics_score * 0.35
        + vis_score * 0.15
        + comp_score * 0.10
    )

    c2pa_data = layer1.get("c2pa") or {}
    provenance_ai = has_ai_provenance(
        has_c2pa=bool(c2pa_data.get("has_c2pa")),
        c2pa_producer=c2pa_data.get("c2pa_producer"),
        ai_tool_used=c2pa_data.get("ai_tool_used"),
        metadata_score=metadata_score,
        generative_details=generative_res.get("details", ""),
        watermark_detected=bool(watermark_res.get("detected")),
        watermark_provider=watermark_res.get("provider"),
        watermark_score=wm_score,
        has_star_mark=bool(watermark_res.get("has_star_mark")),
        watermark_confidence=str(watermark_res.get("confidence", "none")),
        watermark_method=wm_method,
        hf_ai=hf_ai_score,
        clip_ai=clip_ai_score,
    )
    explicit_ai = check_explicit_ai_proof(
        metadata_score=metadata_score,
        generative_score=gen_score,
        generative_details=generative_res.get("details", ""),
        forensic=layer1.get("forensic_signals"),
    ) or provenance_ai

    # Parallel dual detection: AI-generated vs deepfake
    dual = run_dual_detection(
        hf_ai=hf_ai_score,
        clip_ai=clip_ai_score,
        ml_score=ml_score,
        gen_score=gen_score,
        pf_score=pf_score,
        freq_score=forensic_freq,
        tex_score=tex_score,
        diffusion_robust=cr_score,
        deepfake_ml=deepfake_ml_score,
        df_forensics=df_forensics_score,
        clip_df=clip_df_score,
        vis_score=vis_score,
        comp_score=comp_score,
        has_faces=bool(ml_result.get("has_faces")),
        is_messenger=is_msg_mode,
        has_explicit_ai=explicit_ai,
        jpeg_quality=quality,
        hf_available=bool((ml_result.get("hf_details") or {}).get("available", True)),
        provenance_ai=provenance_ai,
        watermark_score=wm_score,
        watermark_method=wm_method,
    )

    ai_generated_score = float(dual["ai_generated_score"])
    deepfake_score = float(dual["deepfake_score"])

    legacy_forensic = (
        metadata_score * 0.20
        + ela_score * 0.25
        + float(compression_res["score"]) * 0.15
        + forensic_freq * 0.15
        + pf_score * 0.10
        + gen_score * 0.15
    )

    confidence_raw = (
        legacy_forensic * 0.18
        + ml_score * 0.22
        + ai_generated_score * 0.45
        + min(combined_deepfake_score * 0.05, 5.0)
    )

    if layer1["forensic_signals"].get("has_c2pa"):
        confidence_raw = max(confidence_raw, 80.0)
    if metadata_res["score"] >= 100:
        confidence_raw = max(confidence_raw, 94.0)

    has_camera = metadata_score <= 20 and bool((metadata_res.get("tags") or {}).get("make"))
    if has_camera and gen_score < 55 and combined_deepfake_score < 50 and hf_ai_score < 55:
        confidence_raw = min(confidence_raw, 38.0)

    signal_map = {
        "generative": gen_score,
        "frequency": forensic_freq,
        "texture": tex_score,
        "compression": comp_score,
        "visual": vis_score,
        "pixel_forensics": pf_score,
        "ml": ml_score,
        "hf_ai": hf_ai_score,
        "clip_ai": clip_ai_score,
    }
    # Do not double-dampen signal map to ensure accurate counting of strong AI signals under compression
    pass

    synthetic_likelihood = compute_synthetic_likelihood(
        weighted_score=confidence_raw,
        generative_score=gen_score,
        metadata_score=metadata_score,
        pf_score=pf_score,
        npr_score=float(pixel_forensics_res.get("signal_scores", {}).get("npr", 0)),
        is_messenger=is_msg_mode or forensics_compressed,
        has_explicit_ai=explicit_ai,
        hf_ai_score=hf_ai_score,
        clip_ai_score=clip_ai_score,
        diffusion_robust_score=cr_score,
        dual_combined=float(dual["combined_synthetic_likelihood"]),
        deepfake_score=combined_deepfake_score,
        jpeg_quality=quality,
        provenance_ai=provenance_ai,
    )
    strong_ai = count_strong_signals(signal_map, ai_threshold=52)
    strong_auth = count_authentic_signals(signal_map, auth_threshold=40)

    forensic_signals = {
        **layer1["forensic_signals"],
        "detected_software": layer1["exif"].get("detected_software"),
        "messenger_shared": is_msg_mode,
        "c2pa_producer": c2pa_data.get("c2pa_producer"),
        "ai_tool_used": c2pa_data.get("ai_tool_used"),
        "generator_watermark": wm_definitive,
        "watermark_suspect": bool(
            watermark_res.get("detected") and wm_method == "faint_sparkle" and not wm_definitive
        ),
        "watermark_provider": watermark_res.get("provider"),
        "watermark_confidence": watermark_res.get("confidence"),
        "watermark_method": wm_method,
    }

    verdict = dual["verdict"]
    model_disagreement = bool(dual.get("model_disagreement"))
    if (
        not is_msg_mode
        and not explicit_ai
        and not provenance_ai
        and verdict not in ("UNCERTAIN", "AUTHENTIC")
    ):
        verdict = determine_verdict(
            confidence=synthetic_likelihood,
            has_faces=ml_result.get("has_faces", False),
            ml_score=ml_score,
            metadata_score=metadata_score,
            generative_score=gen_score,
            forensic=forensic_signals,
            exif_tags=layer1["exif"].get("tags") or metadata_res.get("tags"),
            visual_score=vis_score,
            ela_score=ela_score,
            compression_score=comp_score,
            ai_tool_detected=layer1["exif"].get("detected_software"),
            is_messenger_shared=is_msg_mode,
            deepfake_score=combined_deepfake_score,
            clip_ai_score=clip_ai_score,
            clip_deepfake_score=clip_df_score,
        )
    elif is_msg_mode and not explicit_ai:
        verdict = dual["verdict"]

    if provenance_ai and verdict == "AUTHENTIC":
        verdict = "AI_GENERATED"

    if wm_confident and provenance_ai and verdict == "AUTHENTIC":
        verdict = "AI_GENERATED"
        ai_generated_score = round(max(ai_generated_score, wm_score * 0.92, 82.0), 1)

    if (
        not explicit_ai
        and not provenance_ai
        and not wm_confident
        and is_msg_mode
        and hf_ai_score < 15
        and ml_score >= 54
        and gen_score >= 44
        and verdict == "AUTHENTIC"
    ):
        verdict = "AI_GENERATED"
        ai_generated_score = round(max(ai_generated_score, ml_score * 0.82, 58.0), 1)

    synthetic_likelihood, confidence, ai_generated_score = finalize_display_scores(
        verdict,
        ai_generated_score=ai_generated_score,
        synthetic_likelihood=synthetic_likelihood,
        explicit_ai=explicit_ai,
        provenance_ai=provenance_ai,
    )
    deepfake_score = float(dual["deepfake_score"])
    likelihood = synthetic_and_authentic_likelihood(synthetic_likelihood)

    ai_attribution = run_attribution(
        layer1,
        ml_result,
        cv_img,
        generative_res["score"],
        hf_ai_score=hf_ai_score,
        clip_ai_score=clip_ai_score,
    )

    # Reconcile: weak corner heuristics must not override unanimous "real photo" models.
    attr_top = ai_attribution[0] if ai_attribution else {}
    camera_authentic = (
        attr_top.get("tool") == "Camera Capture / Authentic Photo"
        and int(attr_top.get("confidence", 0)) >= 90
    )
    if (
        verdict == "AI_GENERATED"
        and not explicit_ai
        and not bool(c2pa_data.get("has_c2pa"))
        and camera_authentic
        and neural_models_indicate_real(hf_ai_score, clip_ai_score)
        and not messenger_genai_fingerprint(hf_ai_score, clip_ai_score, cr_score)
        and (not wm_definitive or wm_method == "faint_sparkle")
    ):
        verdict = "AUTHENTIC"
        ai_generated_score = round(min(ai_generated_score, 18.0), 1)
        provenance_ai = False
        synthetic_likelihood, confidence, ai_generated_score = finalize_display_scores(
            verdict,
            ai_generated_score=ai_generated_score,
            synthetic_likelihood=ai_generated_score,
            explicit_ai=False,
            provenance_ai=False,
        )
        likelihood = synthetic_and_authentic_likelihood(synthetic_likelihood)
        forensic_signals["generator_watermark"] = False

    labels_info = derive_labels(
        has_faces=ml_result.get("has_faces", False),
        metadata_score=metadata_score,
        generative_score=gen_score,
        ml_score=ml_score,
        visual_score=vis_score,
        ela_score=ela_score,
        compression_score=comp_score,
        forensic=forensic_signals,
        exif_tags=layer1["exif"].get("tags") or metadata_res.get("tags"),
        ai_tool_detected=layer1["exif"].get("detected_software"),
        deepfake_score=combined_deepfake_score,
        clip_deepfake_score=clip_df_score,
    )

    ela_heatmap = layer1["ela"].get("ela_heatmap") or ""
    heatmap_b64, gradcam_err = generate_gradcam(
        registry, pil_img, fallback_heatmap_b64=ela_heatmap or visual_heatmap
    )

    model_warnings = dict(ml_result.get("model_errors") or {})
    if gradcam_err:
        model_warnings["gradcam"] = gradcam_err



    explanation = build_explanation(
        verdict=verdict,
        confidence=confidence,
        model_disagreement=model_disagreement,
        breakdown={
            "metadata_score": metadata_score,
            "ela_score": ela_score,
            "ml_ensemble_score": ml_score,
            "hf_ai_score": hf_ai_score,
            "clip_ai_score": clip_ai_score,
            "frequency_score": freq_ml_score,
            "pixel_forensics_score": pf_score,
            "deepfake_score": combined_deepfake_score,
            "ai_generated_score": ai_generated_score,
        },
        forensic_signals=forensic_signals,
        ai_attribution=ai_attribution,
        model_errors=model_warnings,
    )

    payload = {
        "verdict": verdict,
        "confidence": round(confidence, 1),
        "synthetic_likelihood": likelihood["synthetic_likelihood"],
        "authentic_likelihood": likelihood["authentic_likelihood"],
        "model_disagreement": model_disagreement,
        "weighted_semantic_score": dual.get("weighted_semantic_score"),
        "dual_detection": {
            "mode": "full",
            "ai_generated": {
                "score": ai_generated_score,
                "likely_fake": ai_generated_score >= 52,
                "signals": dual.get("ai_signals", []),
            },
            "deepfake": {
                "score": deepfake_score,
                "likely_fake": deepfake_score >= 52,
                "signals": dual.get("deepfake_signals", []),
                "face_count": ml_result.get("face_count", 0),
            },
            "primary": dual.get("primary", "ai_generated"),
            "layer_scores": dual.get("layer_scores", {}),
        },
        "labels": labels_info["labels"],
        "camera_authenticity_likelihood": labels_info["camera_authenticity_likelihood"],
        "breakdown": {
            "metadata_score": round(metadata_score, 1),
            "ela_score": round(ela_score, 1),
            "ml_ensemble_score": round(ml_score, 1),
            "hf_ai_score": round(hf_ai_score, 1),
            "frequency_score": round(freq_ml_score, 1),
            "pixel_forensics_score": round(pf_score, 1),
            "deepfake_score": round(combined_deepfake_score, 1),
            "clip_ai_score": round(clip_ai_score, 1),
            "clip_deepfake_score": round(clip_df_score, 1),
            "ai_generated_score": round(ai_generated_score, 1),
            "compression_robust_score": round(cr_score, 1),
        },
        "ai_attribution": ai_attribution,
        "forensic_signals": forensic_signals,
        "heatmap_base64": heatmap_b64 or visual_heatmap,
        "fft_base64": frequency_res.get("fft_base64", ""),
        "ela_heatmap_base64": ela_heatmap,
        "explanation": explanation,
        "analysis_context": {
            "analysis_mode": ctx["analysis_mode"],
            "estimated_jpeg_quality": quality,
            "face_count": ml_result.get("face_count", 0),
            "metadata_stripped": metadata_stripped,
            "messenger_shared": is_msg_mode,
            "forensics_compressed": forensics_compressed,
        },
        "layer_details": {
            "layer1": {
                "exif": layer1["exif"],
                "c2pa": layer1["c2pa"],
                "ela": {k: v for k, v in layer1["ela"].items() if k != "ela_heatmap"},
                "double_jpeg": layer1["double_jpeg"],
            },
            "layer2": {
                "model_scores": ml_result.get("model_scores"),
                "weights": ml_result.get("weights"),
                "hf_details": ml_result.get("hf_details"),
                "load_status": ml_result.get("load_status"),
            },
            "layer3_dual": dual,
            "legacy_signals": {
                "generative": generative_res,
                "frequency": {"score": frequency_res["score"], "details": frequency_res["details"]},
                "texture": texture_res,
                "compression": compression_res,
                "visual": visual_res,
            },
            "pixel_forensics": {
                "score": round(pf_score, 1),
                "signal_scores": pixel_forensics_res.get("signal_scores", {}),
                "strong_signal_count": pixel_forensics_res.get("strong_signal_count", 0),
                "strong_signals": pixel_forensics_res.get("strong_signals", []),
                "all_findings": pixel_forensics_res.get("all_findings", []),
            },
            "deepfake_forensics": {
                "score": round(df_forensics_score, 1),
                "has_faces": deepfake_forensics_res.get("has_faces", False),
                "face_count": deepfake_forensics_res.get("face_count", 0),
                "sub_scores": deepfake_forensics_res.get("sub_scores", {}),
                "all_findings": deepfake_forensics_res.get("all_findings", []),
            },
            "compression_robust": compression_robust_res,
        },
        "model_warnings": model_warnings,
    }
    return to_json_safe(payload)
