"""
Layer 2 — Deep learning multi-model ensemble with frequency-domain features.
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np
from PIL import Image
from scipy.fft import dct

from .calibration import calibrate_ensemble_scores, platt_calibrate
from .model_registry import ML_INPUT_SIZE, ModelRegistry


def extract_dct_features(image_array: np.ndarray) -> tuple[float, float]:
    """DCT high-frequency statistics (AI vs real photo fingerprint)."""
    if image_array.ndim == 3:
        gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY).astype(float)
    else:
        gray = image_array.astype(float)
    h, w = gray.shape
    size = min(512, h, w)
    gray = cv2.resize(gray, (size, size), interpolation=cv2.INTER_AREA)
    dct_map = dct(dct(gray.T, norm="ortho").T, norm="ortho")
    mid = gray.shape[0] // 2
    high_freq = dct_map[mid:, mid:]
    return float(np.std(high_freq)), float(np.mean(np.abs(high_freq)))


def freq_domain_score(dct_std: float, dct_mean_abs: float) -> float:
    """Map DCT features to 0-100 fake-likelihood (heuristic calibration)."""
    # Diffusion/GAN outputs often show elevated high-freq std with moderate mean
    score = 0.0
    if dct_std > 12.0:
        score += min((dct_std - 12.0) * 3.5, 50)
    if dct_mean_abs > 1.8:
        score += min((dct_mean_abs - 1.8) * 25, 40)
    if dct_std < 4.0 and dct_mean_abs < 0.6:
        score += 25  # overly smooth spectrum
    return round(min(max(score, 0.0), 100.0), 2)


def _pil_to_tensor(pil_img: Image.Image, preprocess, device: str):
    import torch

    tensor = preprocess(pil_img.convert("RGB")).unsqueeze(0).to(device)
    return tensor


def _cnn_score(registry: ModelRegistry, pil_img: Image.Image) -> tuple[float, str | None]:
    if registry.cnn_model is None or registry.cnn_preprocess is None:
        return 50.0, "cnn_detection unavailable"
    try:
        import torch

        tensor = _pil_to_tensor(pil_img, registry.cnn_preprocess, registry.device)
        with torch.no_grad():
            logits = registry.cnn_model(tensor)
            if logits.shape[-1] == 2:
                probs = torch.softmax(logits, dim=-1)[0]
                fake_prob = float(probs[1])
            else:
                fake_prob = float(torch.sigmoid(logits[0, 0]))

        # Untrained head: blend with embedding norm heuristic
        arr = np.array(pil_img.resize((ML_INPUT_SIZE, ML_INPUT_SIZE))) / 255.0
        color_var = float(np.std(arr))
        heuristic = min(max((0.12 - color_var) * 400 + fake_prob * 40, 0), 100)
        raw = 0.55 * (fake_prob * 100) + 0.45 * heuristic
        return round(raw, 2), None
    except Exception as e:
        return 50.0, str(e)


def _clip_score(
    registry: ModelRegistry, pil_img: Image.Image
) -> tuple[float, float, float, str | None]:
    """
    Returns (fake_score, ai_score, deepfake_score, error).
    fake_score = combined; ai_score and deepfake_score are class-specific (0-100).
    """
    if registry.clip_model is None or registry.clip_preprocess is None:
        return 50.0, 50.0, 25.0, "clip_detector unavailable"
    try:
        import torch

        rgb_img = pil_img.convert("RGB")
        w, h = rgb_img.size

        # Three crops: full frame, center crop (86%), and a slightly offset crop
        # for robustness against framing effects.
        crops = [rgb_img]
        for scale in (0.86, 0.75):
            cw = max(64, int(w * scale))
            ch = max(64, int(h * scale))
            x0 = (w - cw) // 2
            y0 = (h - ch) // 2
            crops.append(rgb_img.crop((x0, y0, x0 + cw, y0 + ch)))

        tensors = torch.stack(
            [registry.clip_preprocess(c) for c in crops], dim=0
        ).to(registry.device)

        with torch.no_grad():
            image_features = registry.clip_model.encode_image(tensors)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            # text_features: [real_proto, ai_proto, deepfake_proto]
            sims = image_features @ registry.clip_text_features.T  # [3, 3]
            sims_mean = sims.mean(dim=0)

            # Independent binary pairwise classifications:
            real_sim = sims_mean[0]
            ai_sim = sims_mean[1]
            df_sim = sims_mean[2]

            # Softmax on [real, ai]
            ai_logits = torch.stack([real_sim, ai_sim]) * 32.0
            ai_prob = float(ai_logits.softmax(dim=-1)[1])

            # Softmax on [real, df]
            df_logits = torch.stack([real_sim, df_sim]) * 32.0
            df_prob = float(df_logits.softmax(dim=-1)[1])

        # Per-class scores (0-100), calibrated with slight temperature correction
        ai_score = round(min(ai_prob * 100 * 1.20, 100.0), 2)
        df_score = round(min(df_prob * 100 * 1.20, 100.0), 2)
        # Combined fake probability weighting deepfake slightly lower
        # (deepfake class is harder to zero-shot on image content alone)
        fake_prob = max(ai_prob, df_prob * 0.85)
        fake_score = round(min(fake_prob * 100 * 1.15, 100.0), 2)

        return fake_score, ai_score, df_score, None
    except Exception as e:
        return 50.0, 50.0, 25.0, str(e)


def _efficientnet_score(
    registry: ModelRegistry, pil_img: Image.Image, faces: list
) -> tuple[float, str | None]:
    if registry.efficientnet_model is None or registry.efficientnet_preprocess is None:
        return 50.0, "efficientnet_deepfake unavailable"
    if not faces:
        return 35.0, None  # low prior without faces
    try:
        import torch

        cv_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        face_scores = []
        for x, y, w, h in faces[:3]:
            pad = int(0.15 * max(w, h))
            x0 = max(0, x - pad)
            y0 = max(0, y - pad)
            x1 = min(cv_img.shape[1], x + w + pad)
            y1 = min(cv_img.shape[0], y + h + pad)
            crop = cv_img[y0:y1, x0:x1]
            if crop.size == 0:
                continue
            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            crop_pil = Image.fromarray(crop_rgb)
            tensor = _pil_to_tensor(crop_pil, registry.efficientnet_preprocess, registry.device)
            with torch.no_grad():
                logits = registry.efficientnet_model(tensor)
                probs = torch.softmax(logits, dim=-1)[0]
                face_scores.append(float(probs[1]) * 100)

        if not face_scores:
            return 40.0, None
        return round(float(np.mean(face_scores)), 2), None
    except Exception as e:
        return 50.0, str(e)


def _forensic_proxy_score(cv_img_bgr: np.ndarray, has_faces: bool) -> float:
    """When PyTorch models are unavailable, derive ML-like score from pixel forensics."""
    gray = cv2.cvtColor(cv_img_bgr, cv2.COLOR_BGR2GRAY)
    work = cv2.resize(gray, (256, 256), interpolation=cv2.INTER_AREA)
    lap = cv2.Laplacian(work.astype(np.float32), cv2.CV_32F)
    detail = float(np.std(lap))
    block_vars = []
    for r in range(0, 256, 32):
        for c in range(0, 256, 32):
            block_vars.append(float(np.var(lap[r : r + 32, c : c + 32])))
    sharp_cv = float(np.std(block_vars) / (np.mean(block_vars) + 1e-8))
    score = 0.0
    if detail < 11.0:
        score += min((11.0 - detail) * 6, 45)
    if sharp_cv < 0.42:
        score += min((0.42 - sharp_cv) * 120, 40)
    if has_faces and detail < 14.0:
        score += 22  # deepfake face regions often over-smooth
    return round(min(max(score, 0.0), 100.0), 2)


def run_ml_ensemble(
    pil_img: Image.Image,
    cv_img_bgr: np.ndarray,
    registry: ModelRegistry | None = None,
    forensic_hint: float | None = None,
    pixel_forensics_score: float | None = None,
    is_messaging_compressed: bool = False,
) -> dict[str, Any]:
    """Run Layer 2 ensemble; adjust weights when faces are present or metadata is stripped."""
    registry = registry or ModelRegistry.get_instance()
    faces = registry.detect_faces(cv_img_bgr)
    has_faces = len(faces) > 0

    rgb = cv2.cvtColor(cv_img_bgr, cv2.COLOR_BGR2RGB)
    dct_std, dct_mean = extract_dct_features(rgb)
    freq_raw = freq_domain_score(dct_std, dct_mean)

    errors: dict[str, str] = {}
    cnn_raw, err = _cnn_score(registry, pil_img)
    if err:
        errors["cnn"] = err
    clip_raw, clip_ai_raw, clip_df_raw, err = _clip_score(registry, pil_img)
    if err:
        errors["clip"] = err

    hf_ai_raw = 50.0
    hf_details: dict = {}
    if registry.hf_detector is not None:
        hf_ai_raw, hf_details = registry.hf_detector.predict_ai_likelihood(
            pil_img, is_messaging_compressed=is_messaging_compressed
        )
    else:
        errors["hf_ai"] = "hf_ai_detector unavailable"
    eff_raw, err = _efficientnet_score(registry, pil_img, faces)
    if err:
        errors["efficientnet"] = err

    models_missing = bool(errors)
    if models_missing:
        proxy = _forensic_proxy_score(cv_img_bgr, has_faces)
        hint = forensic_hint if forensic_hint is not None else proxy
        blend = 0.65 * proxy + 0.35 * hint
        if "cnn" in errors:
            cnn_raw = blend
        if "clip" in errors:
            clip_raw = blend + 5
            clip_ai_raw = blend + 5
            clip_df_raw = blend * 0.5 if has_faces else 10.0
        if "efficientnet" in errors:
            eff_raw = blend + (15 if has_faces else 0)

    # Include pixel forensics as an additional ensemble member
    pf_score = pixel_forensics_score if pixel_forensics_score is not None else 25.0

    raw_scores = {
        "cnn": cnn_raw,
        "clip": clip_raw,
        "hf_ai": hf_ai_raw,
        "frequency": freq_raw,
        "efficientnet": eff_raw,
        "pixel_forensics": pf_score,
    }
    calibrated = calibrate_ensemble_scores(
        {k: v for k, v in raw_scores.items() if k != "pixel_forensics"}
    )
    calibrated["pixel_forensics"] = pf_score  # pixel forensics already calibrated
    calibrated["clip_ai"] = round(float(clip_ai_raw), 2)
    calibrated["clip_deepfake"] = round(float(clip_df_raw), 2)
    calibrated["hf_ai"] = round(float(hf_ai_raw), 2)

    # Messenger mode: balance semantic models with frequency + pixel (HF often fails on re-JPEG)
    if is_messaging_compressed:
        w_hf, w_cnn, w_clip, w_freq, w_eff, w_pf = 1.3, 1.0, 1.5, 0.4, 1.0, 0.5
    elif has_faces:
        w_hf, w_cnn, w_clip, w_freq, w_eff, w_pf = 1.3, 1.0, 0.8, 1.0, 1.0, 0.5
    else:
        w_hf, w_cnn, w_clip, w_freq, w_eff, w_pf = 1.3, 1.0, 0.8, 1.0, 1.0, 0.5

    weight_total = w_hf + w_cnn + w_clip + w_freq + w_eff + w_pf
    ensemble_raw = (
        w_hf * raw_scores["hf_ai"]
        + w_cnn * raw_scores["cnn"]
        + w_clip * raw_scores["clip"]
        + w_freq * raw_scores["frequency"]
        + w_eff * raw_scores["efficientnet"]
        + w_pf * raw_scores["pixel_forensics"]
    ) / weight_total
    ml_ensemble_score = platt_calibrate(ensemble_raw)

    # Dedicated deepfake likelihood score combining face-aware model signals
    if has_faces:
        df_score_raw = (
            calibrated["clip_deepfake"] * 0.30
            + calibrated["efficientnet"] * 0.40
            + calibrated["cnn"] * 0.20
            + pf_score * 0.10
        )
    else:
        df_score_raw = calibrated["clip_deepfake"] * 0.5 + pf_score * 0.2
    deepfake_ml_score = round(min(df_score_raw, 100.0), 1)

    return {
        "ml_ensemble_score": ml_ensemble_score,
        "deepfake_ml_score": deepfake_ml_score,
        "hf_ai_score": calibrated["hf_ai"],
        "clip_ai_score": calibrated["clip_ai"],
        "clip_deepfake_score": calibrated["clip_deepfake"],
        "frequency_score": calibrated["frequency"],
        "pixel_forensics_score": pf_score,
        "has_faces": has_faces,
        "face_count": len(faces),
        "faces": faces,
        "model_scores": {
            "cnn_detection": calibrated["cnn"],
            "clip": calibrated["clip"],
            "hf_ai": calibrated["hf_ai"],
            "clip_ai": calibrated["clip_ai"],
            "clip_deepfake": calibrated["clip_deepfake"],
            "frequency_dct": calibrated["frequency"],
            "efficientnet_deepfake": calibrated["efficientnet"],
            "pixel_forensics": calibrated["pixel_forensics"],
        },
        "raw_scores": raw_scores,
        "hf_details": hf_details,
        "weights": {
            "hf_ai": w_hf,
            "cnn": w_cnn,
            "clip": w_clip,
            "frequency": w_freq,
            "efficientnet": w_eff,
            "pixel_forensics": w_pf,
        },
        "dct_features": {"std": dct_std, "mean_abs": dct_mean},
        "model_errors": errors,
        "load_status": dict(registry.load_status),
    }
