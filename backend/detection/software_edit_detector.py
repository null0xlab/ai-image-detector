from __future__ import annotations

import io
from typing import Any

import cv2
import numpy as np
import piexif
from PIL import Image

EDIT_SOFTWARE_KEYWORDS = (
    "adobe photoshop",
    "photoshop",
    "lightroom",
    "gimp",
    "facetune",
    "snapseed",
    "picsart",
    "capture one",
    "affinity photo",
)


def compute_ela(image: Image.Image, quality: int = 90) -> Image.Image:
    original = image.convert("RGB")
    buffer = io.BytesIO()
    original.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    compressed = Image.open(buffer).convert("RGB")
    orig_arr = np.array(original, dtype=np.float32)
    comp_arr = np.array(compressed, dtype=np.float32)
    ela_arr = np.abs(orig_arr - comp_arr)
    max_val = float(ela_arr.max()) or 1.0
    ela_arr = (ela_arr / max_val * 255).astype(np.uint8)
    return Image.fromarray(ela_arr)


def _ela_score(ela_image: Image.Image) -> tuple[float, bool]:
    arr = np.array(ela_image.convert("L"), dtype=np.float32)
    h, w = arr.shape
    block_vars = []
    for r in range(0, h, max(h // 8, 16)):
        for c in range(0, w, max(w // 8, 16)):
            block = arr[r : r + max(h // 8, 16), c : c + max(w // 8, 16)]
            if block.size:
                block_vars.append(float(np.var(block)))
    if not block_vars:
        return 0.0, False
    cv_val = float(np.std(block_vars) / (np.mean(block_vars) + 1e-8))
    score = min(max(cv_val * 45, 0.0), 100.0)
    edited = cv_val > 0.55 or float(np.max(block_vars)) > 900.0
    return round(score, 2), edited


def check_editing_software(image_bytes: bytes) -> str | None:
    try:
        exif = piexif.load(image_bytes)
        software = exif.get("0th", {}).get(piexif.ImageIFD.Software, b"")
        if isinstance(software, bytes):
            text = software.decode("utf-8", errors="ignore").strip()
            if text:
                return text
    except Exception:
        pass
    return None


def _detect_editing_software_name(software: str | None) -> str | None:
    if not software:
        return None
    lower = software.lower()
    for keyword in EDIT_SOFTWARE_KEYWORDS:
        if keyword in lower:
            return software
    return None


def _clone_stamp_score(cv_img_bgr: np.ndarray) -> tuple[float, bool]:
    gray = cv2.cvtColor(cv_img_bgr, cv2.COLOR_BGR2GRAY)
    work = cv2.resize(gray, (384, 384), interpolation=cv2.INTER_AREA)
    patch = 16
    best_corr = 0.0
    for y in range(0, work.shape[0] - patch, patch):
        for x in range(0, work.shape[1] - patch, patch):
            p1 = work[y : y + patch, x : x + patch].astype(np.float32).flatten()
            if np.std(p1) < 5:
                continue
            for y2 in range(y + patch, work.shape[0] - patch, patch):
                for x2 in range(0, work.shape[1] - patch, patch):
                    if abs(x - x2) < patch and abs(y - y2) < patch:
                        continue
                    p2 = work[y2 : y2 + patch, x2 : x2 + patch].astype(np.float32).flatten()
                    if np.std(p2) < 5:
                        continue
                    corr = float(np.corrcoef(p1, p2)[0, 1])
                    if corr > best_corr:
                        best_corr = corr
    edited = best_corr > 0.92
    score = min(max((best_corr - 0.75) * 250, 0.0), 85.0)
    return round(score, 2), edited


def detect_software_edits(
    pil_img: Image.Image,
    cv_img_bgr: np.ndarray,
    image_bytes: bytes,
) -> dict[str, Any]:
    ela_image = compute_ela(pil_img)
    ela_score, ela_edited = _ela_score(ela_image)
    software_raw = check_editing_software(image_bytes)
    software = _detect_editing_software_name(software_raw)
    clone_score, clone_edited = _clone_stamp_score(cv_img_bgr)

    edit_score = max(ela_score, clone_score)
    if software:
        edit_score = max(edit_score, 55.0)
    edited = ela_edited or clone_edited or software is not None

    return {
        "edited": edited,
        "edit_score": round(min(edit_score / 100.0, 1.0), 4),
        "edit_score_percent": round(edit_score, 2),
        "software": software or software_raw,
        "ela_heatmap": ela_image,
        "signals": sorted(
            s
            for s, cond in (
                ("ela_localized_variance", ela_edited),
                ("clone_stamp_pattern", clone_edited),
                ("editing_software_exif", software is not None),
            )
            if cond
        ),
    }
