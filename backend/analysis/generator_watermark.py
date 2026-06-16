"""
Detect visible generator watermarks (Google Gemini sparkle, etc.) after social re-sharing.

Telegram/WhatsApp strip C2PA but often leave a small semi-transparent logo in a corner.
Uses (1) upscaled micro-corner blob analysis for faint post-compression sparkles,
(2) brighter-corner star heuristics for direct exports, (3) residual byte hints.
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np


def _corner_rois(h: int, w: int) -> list[tuple[str, slice, slice]]:
    """Named corner regions for direct-export / light-compression sparkles."""
    fh, fw = max(20, int(h * 0.06)), max(20, int(w * 0.06))
    return [
        ("bottom_right", slice(h - fh, h), slice(w - fw, w)),
        ("bottom_left", slice(h - fh, h), slice(0, fw)),
        ("top_right", slice(0, fh), slice(w - fw, w)),
        ("top_left", slice(0, fh), slice(0, fw)),
    ]


def _score_bright_sparkle_patch(bgr: np.ndarray) -> tuple[float, list[str], bool]:
    """Bright Gemini sparkle on typical exports (val > ~170)."""
    findings: list[str] = []
    if bgr.size == 0:
        return 0.0, findings, False

    h, w = bgr.shape[:2]
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    val = hsv[:, :, 2].astype(np.float32)
    sat = hsv[:, :, 1].astype(np.float32)

    blur = cv2.GaussianBlur(gray, (0, 0), 2.2)
    local_contrast = np.abs(gray - blur)
    lc_mean = float(np.mean(local_contrast))

    bright = ((val > 168) & (local_contrast > lc_mean * 1.45)).astype(np.uint8) * 255
    bright = cv2.morphologyEx(bright, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    contours, _ = cv2.findContours(bright, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    patch_area = h * w
    star_like = 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < patch_area * 0.0002 or area > patch_area * 0.05:
            continue
        peri = cv2.arcLength(cnt, True)
        if peri < 1e-3:
            continue
        approx = cv2.approxPolyDP(cnt, 0.20 * peri, True)
        if 4 <= len(approx) <= 12:
            x, y, bw, bh = cv2.boundingRect(cnt)
            ar = bw / (bh + 1e-6)
            compactness = 4 * np.pi * area / (peri * peri + 1e-6)
            if 0.5 <= ar <= 1.8 and compactness > 0.35:
                star_like += 1

    has_star = star_like >= 1
    score = 0.0
    if has_star:
        score += 58 + min(star_like * 10, 25)
        findings.append(f"compact star-like bright mark (candidates={star_like})")

    gray_frac = float(
        np.mean(
            ((val > 145) & (val < 240) & (sat < 85) & (local_contrast > lc_mean * 1.25)).astype(
                np.uint8
            )
        )
    )
    if has_star and 0.003 < gray_frac < 0.10:
        score += 22
        findings.append("semi-transparent gray mark overlapping sparkle")

    return min(score, 100.0), findings, has_star


def _detect_faint_gemini_sparkle(cv_bgr: np.ndarray) -> tuple[float, list[str], bool]:
    """
    Detect ultra-faint Gemini sparkle after Telegram/WhatsApp JPEG (gray values ~50–60).
    Upscales a micro bottom-right strip and finds a small compact bright blob.
    """
    findings: list[str] = []
    h, w = cv_bgr.shape[:2]
    if h < 32 or w < 32:
        return 0.0, findings, False

    fh = max(8, int(h * 0.035))
    fw = max(14, int(w * 0.055))
    strip = cv_bgr[h - fh : h, w - fw : w]
    if strip.size == 0:
        return 0.0, findings, False

    up = cv2.resize(
        strip,
        (max(200, strip.shape[1] * 22), max(120, strip.shape[0] * 22)),
        interpolation=cv2.INTER_LANCZOS4,
    )
    gray = cv2.cvtColor(up, cv2.COLOR_BGR2GRAY)
    patch_area = float(gray.shape[0] * gray.shape[1])
    thr = float(np.percentile(gray, 97.0))
    mask = (gray >= thr).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best = 0.0
    has_mark = False
    gh, gw = gray.shape[:2]

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 100 or area > patch_area * 0.22:
            continue
        peri = cv2.arcLength(cnt, True)
        if peri < 1e-3:
            continue
        compactness = 4 * np.pi * area / (peri * peri + 1e-6)
        if compactness < 0.32:
            continue

        x, y, bw, bh = cv2.boundingRect(cnt)
        cx, cy = x + bw / 2.0, y + bh / 2.0
        corner_bias = 1.0
        if cx >= gw * 0.52 and cy >= gh * 0.52:
            corner_bias = 1.18

        sc = (44.0 + min(compactness * 42.0, 36.0) + min(area / 22.0, 20.0)) * corner_bias
        if sc > best:
            best = sc
            has_mark = True
            findings.append(
                f"faint Gemini-style sparkle (area={area:.0f}, compactness={compactness:.2f})"
            )

    return min(best, 100.0), findings, has_mark


def _byte_hint_provider(image_bytes: bytes) -> tuple[float, str | None, list[str]]:
    sample = image_bytes[: min(len(image_bytes), 384_000)]
    text = sample.decode("latin-1", errors="ignore").lower()
    for hint, label in (
        ("gemini", "Google Gemini"),
        ("googleai", "Google Gemini"),
        ("imagen", "Google Imagen"),
        ("made with google", "Google"),
    ):
        if hint in text:
            return 88.0, label, [f"byte hint: '{hint}'"]
    return 0.0, None, []


def detect_generator_watermark(
    cv_img_bgr: np.ndarray,
    image_bytes: bytes | None = None,
) -> dict[str, Any]:
    """
    Scan for GenAI watermarks. ``detected`` requires a star-like mark or byte hint —
    not generic corner JPEG noise.
    """
    findings: list[str] = []
    provider: str | None = None
    best_score = 0.0
    best_corner = ""
    has_star_mark = False
    method = ""

    if cv_img_bgr is None or cv_img_bgr.size == 0:
        return {
            "score": 0.0,
            "detected": False,
            "confidence": "none",
            "provider": None,
            "findings": [],
            "corner": None,
            "has_star_mark": False,
            "method": None,
        }

    faint_score, faint_findings, faint_mark = _detect_faint_gemini_sparkle(cv_img_bgr)
    if faint_score > best_score:
        best_score = faint_score
        best_corner = "bottom_right"
        has_star_mark = faint_mark
        findings = [f"[bottom_right] {f}" for f in faint_findings]
        method = "faint_sparkle"

    h, w = cv_img_bgr.shape[:2]
    work = cv_img_bgr
    if max(h, w) > 1280:
        scale = 1280 / max(h, w)
        work = cv2.resize(
            cv_img_bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA
        )
        h, w = work.shape[:2]

    for name, rs, cs in _corner_rois(h, w):
        patch = work[rs, cs]
        sc, patch_findings, star = _score_bright_sparkle_patch(patch)
        if sc > best_score:
            best_score = sc
            best_corner = name
            has_star_mark = star
            findings = [f"[{name}] {f}" for f in patch_findings]
            method = "bright_sparkle"

    byte_score, byte_provider, byte_findings = 0.0, None, []
    if image_bytes:
        byte_score, byte_provider, byte_findings = _byte_hint_provider(image_bytes)
        if byte_score > best_score:
            best_score = byte_score
            provider = byte_provider
            findings = byte_findings
            has_star_mark = True
            method = "byte_hint"

    if best_score >= 55 and provider is None and has_star_mark:
        provider = "Google Gemini"

    has_byte_hint = bool(byte_findings)
    suspect = has_byte_hint or (has_star_mark and best_score >= 50)
    definitive = has_byte_hint or (
        method == "bright_sparkle" and has_star_mark and best_score >= 58
    )
    # Faint sparkle: internal score only — pipeline uses messenger consensus, not solo override.
    detected = suspect
    confidence = "high" if definitive else (
        "medium" if method == "faint_sparkle" and best_score >= 55 else (
            "low" if suspect else "none"
        )
    )

    return {
        "score": round(best_score, 1),
        "detected": detected,
        "definitive": definitive,
        "confidence": confidence,
        "provider": provider if definitive else None,
        "findings": findings,
        "corner": best_corner if suspect else None,
        "has_star_mark": has_star_mark or has_byte_hint,
        "method": method if suspect else None,
    }
