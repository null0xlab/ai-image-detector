from __future__ import annotations

from typing import Any

import cv2
import numpy as np
from PIL import Image

from analysis.ml_ensemble import _clip_score, _efficientnet_score
from analysis.model_registry import ModelRegistry


def _face_symmetry_score(cv_img_bgr: np.ndarray, faces: list[tuple[int, int, int, int]]) -> tuple[float, list[str]]:
    signals: list[str] = []
    if not faces:
        return 0.0, signals

    try:
        import mediapipe as mp

        mp_face = mp.solutions.face_mesh
        h, w = cv_img_bgr.shape[:2]
        x, y, fw, fh = faces[0]
        pad = int(0.2 * max(fw, fh))
        x0 = max(0, x - pad)
        y0 = max(0, y - pad)
        x1 = min(w, x + fw + pad)
        y1 = min(h, y + fh + pad)
        crop = cv_img_bgr[y0:y1, x0:x1]
        if crop.size == 0:
            return 0.0, signals
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        with mp_face.FaceMesh(static_image_mode=True, max_num_faces=1) as mesh:
            result = mesh.process(rgb)
            if not result.multi_face_landmarks:
                return 0.0, signals
            lm = result.multi_face_landmarks[0].landmark
            left_eye = np.mean([(lm[i].x, lm[i].y) for i in (33, 133, 160, 159, 158, 144, 145, 153)], axis=0)
            right_eye = np.mean([(lm[i].x, lm[i].y) for i in (362, 263, 387, 386, 385, 384, 398, 373)], axis=0)
            nose = np.array([lm[1].x, lm[1].y])
            eye_mid = (left_eye + right_eye) / 2.0
            asym = float(np.linalg.norm(nose - eye_mid))
            if asym > 0.045:
                signals.append("ear_asymmetry")
            left_cheek = np.array([lm[234].x, lm[234].y])
            right_cheek = np.array([lm[454].x, lm[454].y])
            cheek_diff = abs(left_cheek[1] - right_cheek[1])
            if cheek_diff > 0.04:
                signals.append("facial_symmetry_anomaly")
            score = min(asym * 800 + cheek_diff * 400, 100.0)
            return round(score, 2), signals
    except Exception:
        pass

    gray = cv2.cvtColor(cv_img_bgr, cv2.COLOR_BGR2GRAY)
    x, y, fw, fh = faces[0]
    face = gray[y : y + fh, x : x + fw]
    if face.size == 0:
        return 0.0, signals
    left = face[:, : fw // 2]
    right = cv2.flip(face[:, fw // 2 :], 1)
    min_w = min(left.shape[1], right.shape[1])
    if min_w < 8:
        return 0.0, signals
    diff = float(np.mean(np.abs(left[:, :min_w].astype(float) - right[:, :min_w].astype(float))))
    score = min(diff * 2.5, 100.0)
    if diff > 18:
        signals.append("facial_symmetry_anomaly")
    return round(score, 2), signals


def _skin_texture_uniformity(cv_img_bgr: np.ndarray, faces: list[tuple[int, int, int, int]]) -> tuple[float, list[str]]:
    signals: list[str] = []
    if not faces:
        return 0.0, signals
    gray = cv2.cvtColor(cv_img_bgr, cv2.COLOR_BGR2GRAY)
    x, y, fw, fh = faces[0]
    face = gray[y : y + fh, x : x + fw]
    if face.size == 0:
        return 0.0, signals
    lap = cv2.Laplacian(face.astype(np.float32), cv2.CV_32F)
    detail_std = float(np.std(lap))
    if detail_std < 8.5:
        signals.append("skin_texture_uniform")
    score = max(0.0, min((10.0 - detail_std) * 8.0, 85.0))
    return round(score, 2), signals


def _eye_reflection_check(cv_img_bgr: np.ndarray, faces: list[tuple[int, int, int, int]]) -> tuple[float, list[str]]:
    signals: list[str] = []
    if not faces:
        return 0.0, signals
    gray = cv2.cvtColor(cv_img_bgr, cv2.COLOR_BGR2GRAY)
    x, y, fw, fh = faces[0]
    eye_h = max(8, fh // 5)
    left_eye = gray[y + fh // 4 : y + fh // 4 + eye_h, x + fw // 6 : x + fw // 2]
    right_eye = gray[y + fh // 4 : y + fh // 4 + eye_h, x + fw // 2 : x + 5 * fw // 6]
    if left_eye.size == 0 or right_eye.size == 0:
        return 0.0, signals
    left_bright = float(np.mean(left_eye > np.percentile(left_eye, 92)))
    right_bright = float(np.mean(right_eye > np.percentile(right_eye, 92)))
    mismatch = abs(left_bright - right_bright)
    if mismatch > 0.12:
        signals.append("eye_reflection")
    score = min(mismatch * 350, 75.0)
    return round(score, 2), signals


def _hair_boundary_artifacts(cv_img_bgr: np.ndarray, faces: list[tuple[int, int, int, int]]) -> tuple[float, list[str]]:
    signals: list[str] = []
    if not faces:
        return 0.0, signals
    gray = cv2.cvtColor(cv_img_bgr, cv2.COLOR_BGR2GRAY)
    x, y, fw, fh = faces[0]
    hair = gray[max(0, y - fh // 4) : y + fh // 6, x : x + fw]
    if hair.size == 0:
        return 0.0, signals
    edges = cv2.Canny(hair, 40, 120)
    edge_density = float(np.mean(edges > 0))
    if edge_density < 0.015:
        signals.append("hair_artifacts")
    score = max(0.0, min((0.025 - edge_density) * 2500, 60.0))
    return round(score, 2), signals


def analyze_portrait_section(
    pil_img: Image.Image,
    cv_img_bgr: np.ndarray,
    registry: ModelRegistry | None = None,
    *,
    is_messaging_compressed: bool = False,
) -> dict[str, Any]:
    registry = registry or ModelRegistry.get_instance()
    faces = registry.detect_faces(cv_img_bgr)
    signals: list[str] = []
    sub_scores: list[float] = []

    hf_score = 50.0
    if registry.hf_detector is not None:
        hf_score, hf_details = registry.hf_detector.predict_ai_likelihood(
            pil_img, is_messaging_compressed=is_messaging_compressed
        )
        if hf_score >= 65:
            signals.append("hf_ai_portrait")
        if hf_details.get("available") is False:
            hf_score = 50.0
    sub_scores.append(hf_score * 1.3)

    eff_score, _ = _efficientnet_score(registry, pil_img, faces)
    if eff_score >= 55 and faces:
        signals.append("efficientnet_face")
    sub_scores.append(eff_score)

    _, clip_ai, clip_df, _ = _clip_score(registry, pil_img)
    sub_scores.append(clip_ai * 0.85)
    if clip_df >= 55:
        signals.append("deepfake_likelihood")

    sym_score, sym_signals = _face_symmetry_score(cv_img_bgr, faces)
    signals.extend(sym_signals)
    sub_scores.append(sym_score * 0.7)

    skin_score, skin_signals = _skin_texture_uniformity(cv_img_bgr, faces)
    signals.extend(skin_signals)
    sub_scores.append(skin_score)

    eye_score, eye_signals = _eye_reflection_check(cv_img_bgr, faces)
    signals.extend(eye_signals)
    sub_scores.append(eye_score * 0.8)

    hair_score, hair_signals = _hair_boundary_artifacts(cv_img_bgr, faces)
    signals.extend(hair_signals)
    sub_scores.append(hair_score * 0.6)

    if not faces:
        signals.append("no_face_detected")
        final = float(np.mean(sub_scores[:3])) if sub_scores else 50.0
    else:
        final = float(np.mean(sub_scores)) if sub_scores else 50.0

    return {
        "score": round(min(max(final / 100.0, 0.0), 1.0), 4),
        "score_percent": round(min(max(final, 0.0), 100.0), 2),
        "signals": sorted(set(signals)),
        "face_count": len(faces),
        "model_scores": {
            "hf_ai": round(hf_score, 2),
            "efficientnet": round(eff_score, 2),
            "clip_ai": round(clip_ai, 2),
            "clip_deepfake": round(clip_df, 2),
            "symmetry": sym_score,
            "skin_texture": skin_score,
        },
    }
