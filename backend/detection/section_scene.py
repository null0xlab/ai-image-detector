from __future__ import annotations

from typing import Any

import cv2
import numpy as np
from PIL import Image

from analysis.ml_ensemble import _clip_score, extract_dct_features, freq_domain_score
from analysis.model_registry import ModelRegistry

ARTWORK_PROMPTS = [
    "human made digital art created in Photoshop or Procreate",
    "hand painted illustration by an artist",
]
AI_ART_PROMPTS = [
    "AI generated art from Midjourney or Stable Diffusion",
    "synthetically generated digital artwork with cinematic lighting",
]


def _clip_semantic_distance(registry: ModelRegistry, pil_img: Image.Image, section: str) -> tuple[float, list[str]]:
    signals: list[str] = []
    if registry.clip_model is None or registry.clip_preprocess is None:
        return 50.0, signals

    import torch
    import open_clip

    real_prompts = [
        "a real unedited photograph taken by a camera",
        "an authentic outdoor photo with natural lighting",
        "a genuine camera photo with natural depth of field",
    ]
    ai_prompts = [
        "an AI generated synthetic image",
        "a computer generated scene from Midjourney or Stable Diffusion",
        "a diffusion model output with unnatural perfection",
    ]
    if section == "artwork":
        real_prompts = ARTWORK_PROMPTS
        ai_prompts = AI_ART_PROMPTS

    rgb = pil_img.convert("RGB")
    tensor = registry.clip_preprocess(rgb).unsqueeze(0).to(registry.device)
    tokenizer = registry.clip_tokenizer or open_clip.get_tokenizer("ViT-L-14")
    with torch.no_grad():
        image_features = registry.clip_model.encode_image(tensor)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        real_tokens = tokenizer(real_prompts).to(registry.device)
        ai_tokens = tokenizer(ai_prompts).to(registry.device)
        real_feat = registry.clip_model.encode_text(real_tokens)
        ai_feat = registry.clip_model.encode_text(ai_tokens)
        real_feat /= real_feat.norm(dim=-1, keepdim=True)
        ai_feat /= ai_feat.norm(dim=-1, keepdim=True)
        real_sim = float((image_features @ real_feat.T).mean())
        ai_sim = float((image_features @ ai_feat.T).mean())
        logits = torch.tensor([real_sim, ai_sim]) * 32.0
        ai_prob = float(logits.softmax(dim=-1)[1])

    score = round(min(ai_prob * 100 * 1.15, 100.0), 2)
    if score >= 55:
        signals.append("clip_semantic_ai")
    if real_sim < 0.18:
        signals.append("lighting_inconsistency")
    return score, signals


def _resnet_frequency_branch(cv_img_bgr: np.ndarray) -> tuple[float, list[str]]:
    signals: list[str] = []
    rgb = cv2.cvtColor(cv_img_bgr, cv2.COLOR_BGR2RGB)
    dct_std, dct_mean = extract_dct_features(rgb)
    score = freq_domain_score(dct_std, dct_mean)
    if score >= 50:
        signals.append("frequency_anomaly")
    if dct_std > 14:
        signals.append("gan_spectrum")
    elif dct_std < 5:
        signals.append("diffusion_smooth_spectrum")
    return score, signals


def _sky_region_analysis(cv_img_bgr: np.ndarray) -> tuple[float, list[str]]:
    signals: list[str] = []
    hsv = cv2.cvtColor(cv_img_bgr, cv2.COLOR_BGR2HSV)
    h, w = cv_img_bgr.shape[:2]
    upper = hsv[: max(h // 3, 1), :, :]
    sky_mask = (upper[:, :, 1] < 80) & (upper[:, :, 2] > 90)
    if float(np.mean(sky_mask)) < 0.08:
        return 25.0, signals
    sky = cv_img_bgr[: max(h // 3, 1), :, :]
    gray = cv2.cvtColor(sky, cv2.COLOR_BGR2GRAY).astype(np.float32)
    f = np.fft.fft2(gray - gray.mean())
    mag = np.abs(np.fft.fftshift(f))
    mid = mag.shape[0] // 2
    high = mag[mid:, mid:]
    ratio = float(np.mean(high) / (np.mean(mag[:mid, :mid]) + 1e-8))
    if ratio < 0.08:
        signals.append("sky_artifact")
    score = max(0.0, min((0.12 - ratio) * 500, 70.0))
    return round(score, 2), signals


def _texture_repetition(cv_img_bgr: np.ndarray) -> tuple[float, list[str]]:
    signals: list[str] = []
    gray = cv2.cvtColor(cv_img_bgr, cv2.COLOR_BGR2GRAY)
    work = cv2.resize(gray, (256, 256), interpolation=cv2.INTER_AREA)
    f = np.fft.fft2(work.astype(np.float32))
    mag = np.abs(f)
    peaks = mag > (np.mean(mag) + 3.5 * np.std(mag))
    peak_count = int(np.sum(peaks))
    if peak_count > 120:
        signals.append("texture_repetition")
    score = min(max((peak_count - 80) * 0.6, 0.0), 75.0)
    return round(score, 2), signals


def analyze_scene_section(
    pil_img: Image.Image,
    cv_img_bgr: np.ndarray,
    registry: ModelRegistry | None = None,
    *,
    section: str = "location_scene",
    is_messaging_compressed: bool = False,
) -> dict[str, Any]:
    registry = registry or ModelRegistry.get_instance()
    signals: list[str] = []
    sub_scores: list[float] = []

    clip_score, clip_signals = _clip_semantic_distance(registry, pil_img, section)
    signals.extend(clip_signals)
    clip_weight = 1.5 if is_messaging_compressed else 0.8
    sub_scores.append(clip_score * clip_weight)

    freq_score, freq_signals = _resnet_frequency_branch(cv_img_bgr)
    if not is_messaging_compressed:
        signals.extend(freq_signals)
        sub_scores.append(freq_score)
    else:
        sub_scores.append(clip_score * 0.5)

    sky_score, sky_signals = _sky_region_analysis(cv_img_bgr)
    signals.extend(sky_signals)
    sub_scores.append(sky_score * 0.7)

    tex_score, tex_signals = _texture_repetition(cv_img_bgr)
    if section in ("location_scene", "crowd_scene"):
        signals.extend(tex_signals)
        sub_scores.append(tex_score)

    _, clip_ai, _, _ = _clip_score(registry, pil_img)
    sub_scores.append(clip_ai * 0.6)

    if section == "crowd_scene":
        faces = registry.detect_faces(cv_img_bgr)
        if len(faces) >= 2:
            signals.append("multiple_people")
        elif len(faces) == 0:
            sub_scores.append(35.0)

    final = float(np.mean(sub_scores)) if sub_scores else 50.0
    return {
        "score": round(min(max(final / 100.0, 0.0), 1.0), 4),
        "score_percent": round(min(max(final, 0.0), 100.0), 2),
        "signals": sorted(set(signals)),
        "section_mode": section,
        "model_scores": {
            "clip_semantic": clip_score,
            "frequency": freq_score,
            "sky": sky_score,
            "texture_repetition": tex_score,
        },
    }
