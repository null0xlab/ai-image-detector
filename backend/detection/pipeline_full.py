from __future__ import annotations

import os
import uuid
from typing import Any

import cv2
from PIL import Image

from analysis import estimate_jpeg_quality, load_image, to_json_safe
from analysis.ml_ensemble import _clip_score, _efficientnet_score, extract_dct_features, freq_domain_score, run_ml_ensemble
from analysis.model_registry import ModelRegistry

from .ensemble import run_ensemble
from .image_type_classifier import classify_image_type
from .metadata_analyzer import analyze_image_metadata
from .section_document import analyze_document_section
from .section_portrait import analyze_portrait_section
from .section_scene import analyze_scene_section
from .software_edit_detector import detect_software_edits

ELA_STATIC_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "static", "ela")


def _detect_recompression(
    image_bytes: bytes,
    filename: str,
    metadata_result: dict[str, Any],
    quality: int,
) -> bool:
    filesize_kb = len(image_bytes) / 1024.0
    has_exif = metadata_result.get("has_camera_exif", False)
    ext = os.path.splitext(filename)[1].lower()
    is_jpeg = ext in (".jpg", ".jpeg") or image_bytes[:3] == b"\xff\xd8\xff"
    return filesize_kb < 100 and not has_exif and is_jpeg and quality < 75


def _save_ela_heatmap(ela_image: Image.Image) -> str:
    os.makedirs(ELA_STATIC_DIR, exist_ok=True)
    name = f"{uuid.uuid4().hex}.png"
    path = os.path.join(ELA_STATIC_DIR, name)
    ela_image.save(path, format="PNG")
    return f"/static/ela/{name}"


def _run_section_pipeline(
    section: str,
    pil_img: Image.Image,
    cv_img_bgr: np.ndarray,
    image_bytes: bytes,
    registry: ModelRegistry,
    metadata_result: dict[str, Any],
    possibly_recompressed: bool,
) -> dict[str, Any]:
    if section == "portrait":
        return analyze_portrait_section(
            pil_img,
            cv_img_bgr,
            registry=registry,
            is_messaging_compressed=possibly_recompressed,
        )
    if section == "document":
        return analyze_document_section(
            pil_img,
            cv_img_bgr,
            image_bytes,
            registry=registry,
            has_camera_exif=metadata_result.get("has_camera_exif", False),
        )
    if section in ("crowd_scene", "location_scene", "artwork", "general"):
        scene_section = section if section != "general" else "location_scene"
        return analyze_scene_section(
            pil_img,
            cv_img_bgr,
            registry=registry,
            section=scene_section,
            is_messaging_compressed=possibly_recompressed,
        )
    return analyze_scene_section(
        pil_img,
        cv_img_bgr,
        registry=registry,
        section="location_scene",
        is_messaging_compressed=possibly_recompressed,
    )


def run_full_pipeline(
    image_bytes: bytes,
    filename: str,
    registry: ModelRegistry | None = None,
) -> dict[str, Any]:
    registry = registry or ModelRegistry.get_instance()
    pil_img, cv_img = load_image(image_bytes)
    quality = estimate_jpeg_quality(pil_img)

    type_result = classify_image_type(pil_img, cv_img, registry=registry)
    section = type_result["section"]

    metadata_result = analyze_image_metadata(image_bytes)
    possibly_recompressed = _detect_recompression(image_bytes, filename, metadata_result, quality)

    section_result = _run_section_pipeline(
        section,
        pil_img,
        cv_img,
        image_bytes,
        registry,
        metadata_result,
        possibly_recompressed,
    )

    edit_result = detect_software_edits(pil_img, cv_img, image_bytes)
    ela_url = _save_ela_heatmap(edit_result["ela_heatmap"])

    ml_result = run_ml_ensemble(
        pil_img,
        cv_img,
        registry=registry,
        is_messaging_compressed=possibly_recompressed,
    )
    rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
    dct_std, dct_mean = extract_dct_features(rgb)
    freq_score = freq_domain_score(dct_std, dct_mean)
    _, clip_ai, _, _ = _clip_score(registry, pil_img)
    faces = registry.detect_faces(cv_img)
    eff_score, _ = _efficientnet_score(registry, pil_img, faces)

    universal_scores = {
        "hf_ai": float(ml_result.get("hf_ai_score", 50.0)),
        "efficientnet": float(eff_score),
        "clip": float(clip_ai),
        "frequency": float(freq_score),
    }

    ensemble_result = run_ensemble(
        section=section,
        section_result=section_result,
        universal_scores=universal_scores,
        metadata_result=metadata_result,
        edit_result=edit_result,
        possibly_recompressed=possibly_recompressed,
    )

    ai_score = ensemble_result["ai_score"]
    ai_generated = ai_score >= 0.52

    return to_json_safe(
        {
            "ai_generated": ai_generated,
            "ai_score": ai_score,
            "edit_detected": bool(edit_result.get("edited")),
            "edit_score": ensemble_result["edit_score"],
            "section_detected": section,
            "section_confidence": type_result.get("confidence"),
            "section_probabilities": type_result.get("probabilities"),
            "signals": ensemble_result["signals"],
            "generator_hint": ensemble_result.get("generator_hint"),
            "ela_heatmap_url": ela_url,
            "recompressed": possibly_recompressed,
            "confidence": ensemble_result["confidence"],
            "metadata": {
                "has_camera_exif": metadata_result.get("has_camera_exif"),
                "software": metadata_result.get("software"),
                "c2pa_result": metadata_result.get("c2pa_result"),
                "suspicious": metadata_result.get("suspicious"),
            },
            "edit_details": {
                "software": edit_result.get("software"),
                "edit_score_percent": edit_result.get("edit_score_percent"),
            },
            "ensemble": ensemble_result,
            "section_analysis": section_result,
            "universal_ml": {
                "hf_ai_score": ml_result.get("hf_ai_score"),
                "clip_ai_score": ml_result.get("clip_ai_score"),
                "frequency_score": ml_result.get("frequency_score"),
                "ml_ensemble_score": ml_result.get("ml_ensemble_score"),
            },
            "analysis_context": {
                "estimated_jpeg_quality": quality,
                "face_count": type_result.get("face_count", 0),
                "aspect_ratio": type_result.get("aspect_ratio"),
                "filesize_kb": round(len(image_bytes) / 1024.0, 2),
            },
        }
    )
