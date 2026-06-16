from __future__ import annotations

from typing import Any

import numpy as np
from PIL import Image

from analysis.model_registry import ModelRegistry

SECTION_LABELS = [
    "portrait",
    "crowd_scene",
    "document",
    "location_scene",
    "artwork",
    "general",
]

SECTION_PROMPTS = {
    "portrait": [
        "a close-up portrait photo of a single person",
        "a selfie or profile picture of one person",
        "a passport or ID photo of a face",
    ],
    "crowd_scene": [
        "a street photo with people in the background",
        "an event or crowd scene with distant people",
        "a landscape photo with small people in the distance",
    ],
    "document": [
        "a photograph of an identity document or passport",
        "a scanned certificate receipt or invoice",
        "an ID card or license document photo",
    ],
    "location_scene": [
        "a photograph of a building landmark or tourist location",
        "an outdoor place or cityscape photo",
        "a landscape or architectural scene photo",
    ],
    "artwork": [
        "a digital illustration or concept art image",
        "a painting or artistic drawing",
        "a stylized digital artwork or comic illustration",
    ],
    "general": [
        "a general photograph of an object or scene",
        "an everyday photo that does not fit a specific category",
    ],
}


def _aspect_ratio_hint(width: int, height: int) -> dict[str, float]:
    if height <= 0 or width <= 0:
        return {label: 1.0 / len(SECTION_LABELS) for label in SECTION_LABELS}
    ratio = width / height
    hints = {label: 0.05 for label in SECTION_LABELS}
    if ratio < 0.85:
        hints["portrait"] += 0.35
        hints["document"] += 0.25
    elif ratio > 1.6:
        hints["location_scene"] += 0.30
        hints["crowd_scene"] += 0.25
    else:
        hints["artwork"] += 0.20
        hints["general"] += 0.15
    if min(width, height) >= 512:
        hints["location_scene"] += 0.05
        hints["crowd_scene"] += 0.05
    total = sum(hints.values())
    return {k: v / total for k, v in hints.items()}


def _clip_section_scores(registry: ModelRegistry, pil_img: Image.Image) -> dict[str, float]:
    if registry.clip_model is None or registry.clip_preprocess is None:
        return {label: 1.0 / len(SECTION_LABELS) for label in SECTION_LABELS}

    import torch
    import open_clip

    rgb = pil_img.convert("RGB")
    tensor = registry.clip_preprocess(rgb).unsqueeze(0).to(registry.device)
    all_prompts: list[str] = []
    prompt_map: list[str] = []
    for label, prompts in SECTION_PROMPTS.items():
        for prompt in prompts:
            all_prompts.append(prompt)
            prompt_map.append(label)

    tokenizer = registry.clip_tokenizer or open_clip.get_tokenizer("ViT-L-14")
    with torch.no_grad():
        image_features = registry.clip_model.encode_image(tensor)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        tokens = tokenizer(all_prompts).to(registry.device)
        text_features = registry.clip_model.encode_text(tokens)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        sims = (image_features @ text_features.T).squeeze(0).cpu().numpy()

    section_sims: dict[str, list[float]] = {label: [] for label in SECTION_LABELS}
    for sim, label in zip(sims, prompt_map):
        section_sims[label].append(float(sim))

    raw = {label: float(np.mean(values)) if values else 0.0 for label, values in section_sims.items()}
    arr = np.array([raw[label] for label in SECTION_LABELS], dtype=np.float64)
    arr = np.exp((arr - arr.max()) * 8.0)
    arr /= arr.sum() + 1e-8
    return {label: float(arr[i]) for i, label in enumerate(SECTION_LABELS)}


def classify_image_type(
    pil_img: Image.Image,
    cv_img_bgr: np.ndarray,
    registry: ModelRegistry | None = None,
) -> dict[str, Any]:
    registry = registry or ModelRegistry.get_instance()
    width, height = pil_img.size
    clip_probs = _clip_section_scores(registry, pil_img)
    aspect_probs = _aspect_ratio_hint(width, height)

    combined = {
        label: 0.72 * clip_probs[label] + 0.28 * aspect_probs[label]
        for label in SECTION_LABELS
    }

    faces = registry.detect_faces(cv_img_bgr)
    if faces:
        combined["portrait"] += 0.18
        combined["crowd_scene"] += 0.04
    elif len(faces) == 0 and max(width, height) > 800:
        combined["location_scene"] += 0.06
        combined["crowd_scene"] += 0.04

    total = sum(combined.values())
    combined = {k: v / total for k, v in combined.items()}
    section = max(combined, key=combined.get)
    confidence = combined[section]

    return {
        "section": section,
        "confidence": round(float(confidence), 4),
        "probabilities": {k: round(float(v), 4) for k, v in combined.items()},
        "face_count": len(faces),
        "aspect_ratio": round(width / max(height, 1), 3),
        "resolution": {"width": width, "height": height},
    }
