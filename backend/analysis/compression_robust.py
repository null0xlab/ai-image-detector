"""
Compression-robust forensic signals for messenger / social re-shared images.

Sightengine-style systems rely on pixel-only deep models trained on recompressed
data. This module adds classical + spectral cues that survive JPEG re-quantization,
metadata stripping, and resizing (Telegram, WhatsApp, Instagram, screenshots).
"""

from __future__ import annotations

import io
from typing import Any

import cv2
import numpy as np
from PIL import Image


def build_inference_variants(pil_img: Image.Image, max_variants: int = 6) -> list[Image.Image]:
    """
    Test-time augmentation branches for neural detectors.
    Includes center crops and synthetic recompression similar to messaging apps.
    """
    rgb = pil_img.convert("RGB")
    w, h = rgb.size
    variants: list[Image.Image] = [rgb]

    for scale in (0.88, 0.76):
        cw = max(96, int(w * scale))
        ch = max(96, int(h * scale))
        x0 = (w - cw) // 2
        y0 = (h - ch) // 2
        variants.append(rgb.crop((x0, y0, x0 + cw, y0 + ch)))

    # Messenger-style JPEG passes (Telegram often ~80–88 quality)
    for q in (82, 72):
        buf = io.BytesIO()
        rgb.save(buf, format="JPEG", quality=q, optimize=True)
        buf.seek(0)
        variants.append(Image.open(buf).convert("RGB"))

    return variants[:max_variants]


def _radial_spectrum(gray: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Log radial average of 2D FFT magnitude."""
    f = np.fft.fft2(gray.astype(np.float64))
    fshift = np.fft.fftshift(f)
    mag = np.log1p(np.abs(fshift))
    h, w = mag.shape
    cy, cx = h // 2, w // 2
    y, x = np.ogrid[:h, :w]
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2).astype(np.int32)
    max_r = min(cx, cy) - 2
    radial = np.zeros(max_r, dtype=np.float64)
    counts = np.zeros(max_r, dtype=np.float64)
    for radius in range(max_r):
        mask = r == radius
        if np.any(mask):
            radial[radius] = float(np.mean(mag[mask]))
            counts[radius] = float(np.sum(mask))
    return radial, counts


def _spectral_slope(radial: np.ndarray) -> float:
    """Negative slope in log-log domain → diffusion-like smooth tail."""
    n = min(len(radial), 64)
    if n < 12:
        return 0.0
    seg = radial[4:n]
    seg = seg[seg > 1e-6]
    if len(seg) < 8:
        return 0.0
    x = np.log(np.arange(1, len(seg) + 1, dtype=np.float64))
    y = np.log(seg)
    slope = float(np.polyfit(x, y, 1)[0])
    return slope


def _grid_energy_ratio(gray: np.ndarray) -> float:
    """Energy at 8×8 JPEG lattice frequencies vs neighbors."""
    f = np.fft.fft2(gray.astype(np.float64))
    mag = np.abs(np.fft.fftshift(f))
    h, w = mag.shape
    cy, cx = h // 2, w // 2
    peaks = []
    bg = []
    for k in range(1, min(cx, cy) // 8):
        fx = cx + k * 8
        fy = cy + k * 8
        if fx < w and fy < h:
            peaks.append(mag[fy, fx])
            bg.append(mag[fy, min(fx + 2, w - 1)])
    if not peaks:
        return 0.0
    return float(np.mean(peaks) / (np.mean(bg) + 1e-8))


def _multiscale_gradient_entropy(gray: np.ndarray) -> float:
    entropies = []
    for size in (256, 384, 512):
        g = cv2.resize(gray, (size, size), interpolation=cv2.INTER_AREA).astype(np.float32)
        gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
        mag = np.sqrt(gx * gx + gy * gy)
        hist, _ = np.histogram(mag.ravel(), bins=32, range=(0, float(np.percentile(mag, 99)) + 1e-6))
        hist = hist.astype(np.float64) + 1e-8
        hist /= hist.sum()
        ent = float(-np.sum(hist * np.log(hist)))
        entropies.append(ent)
    return float(np.std(entropies))


def analyze_diffusion_spectrum(cv_img_bgr: np.ndarray) -> dict[str, Any]:
    """
    Frequency-domain fingerprint for diffusion / GAN outputs.
    Survives moderate JPEG because spectral shape persists in radial averages.
    """
    gray = cv2.cvtColor(cv_img_bgr, cv2.COLOR_BGR2GRAY)
    work = cv2.resize(gray, (512, 512), interpolation=cv2.INTER_AREA)
    work = work.astype(np.float64)
    work -= work.mean()

    radial, _ = _radial_spectrum(work)
    slope = _spectral_slope(radial)
    grid_ratio = _grid_energy_ratio(work)
    grad_entropy_std = _multiscale_gradient_entropy(work)

    score = 0.0
    findings: list[str] = []

    # Diffusion: smoother high-frequency roll-off (less negative slope magnitude)
    if slope > -0.85:
        bump = min((slope + 0.85) * 55, 45)
        score += bump
        findings.append(f"diffusion-like spectral roll-off (slope={slope:.2f})")

    mid_band = float(np.mean(radial[8:32])) if len(radial) > 32 else 0.0
    high_band = float(np.mean(radial[32:64])) if len(radial) > 64 else mid_band
    if mid_band > 1e-6 and high_band / (mid_band + 1e-8) < 0.42:
        score += 22
        findings.append("suppressed high-frequency energy vs mid-band (generative)")

    if grid_ratio > 1.35:
        score += min((grid_ratio - 1.35) * 40, 35)
        findings.append(f"JPEG lattice periodicity elevated (ratio={grid_ratio:.2f})")

    # AI images often have unusually stable gradient distribution across scales
    if grad_entropy_std < 0.28:
        score += min((0.28 - grad_entropy_std) * 120, 30)
        findings.append("scale-invariant gradient statistics (synthetic)")

    return {
        "score": round(min(max(score, 0.0), 100.0), 1),
        "findings": findings,
        "details": {
            "spectral_slope": round(slope, 3),
            "grid_ratio": round(grid_ratio, 3),
            "grad_entropy_std": round(grad_entropy_std, 3),
        },
    }


def analyze_multiscale_texture(cv_img_bgr: np.ndarray) -> dict[str, Any]:
    """LBP-style micro-texture spread across scales (compression-invariant)."""
    gray = cv2.cvtColor(cv_img_bgr, cv2.COLOR_BGR2GRAY)
    scores = []
    for size in (224, 320, 448):
        g = cv2.resize(gray, (size, size), interpolation=cv2.INTER_AREA)
        lap = cv2.Laplacian(g.astype(np.float32), cv2.CV_32F)
        block_vars = []
        bs = 32
        for r in range(0, size - bs, bs):
            for c in range(0, size - bs, bs):
                block_vars.append(float(np.var(lap[r : r + bs, c : c + bs])))
        arr = np.array(block_vars, dtype=np.float64)
        cv_ratio = float(np.std(arr) / (np.mean(arr) + 1e-8))
        scores.append(cv_ratio)

    mean_cv = float(np.mean(scores))
    cross_scale_std = float(np.std(scores))
    score = 0.0
    findings: list[str] = []

    if mean_cv < 0.48:
        score += min((0.48 - mean_cv) * 130, 50)
        findings.append(f"uniform micro-texture across frame (CV={mean_cv:.2f})")
    if cross_scale_std < 0.06:
        score += 18
        findings.append("texture statistics stable across scales")

    return {
        "score": round(min(max(score, 0.0), 100.0), 1),
        "findings": findings,
        "details": {"mean_cv": round(mean_cv, 3), "cross_scale_std": round(cross_scale_std, 3)},
    }


def run_compression_robust_analysis(
    cv_img_bgr: np.ndarray,
    *,
    is_messaging_compressed: bool = False,
    hf_ai_score: float = 50.0,
) -> dict[str, Any]:
    """Combined compression-robust score (0–100). Capped when HF strongly says real."""
    spec = analyze_diffusion_spectrum(cv_img_bgr)
    tex = analyze_multiscale_texture(cv_img_bgr)

    if is_messaging_compressed:
        w_spec, w_tex = 0.58, 0.42
    else:
        w_spec, w_tex = 0.50, 0.50

    combined = spec["score"] * w_spec + tex["score"] * w_tex
    findings = spec["findings"] + tex["findings"]
    n_findings = len(findings)

    # JPEG recompression triggers spectral findings on real photos — require HF corroboration
    if hf_ai_score >= 20:
        if n_findings >= 2:
            combined = max(combined, 56.0 + (n_findings - 2) * 7)
        if n_findings >= 3 and hf_ai_score >= 30:
            combined = max(combined, 65.0)

    n_strong = sum(1 for s in (spec["score"], tex["score"]) if s >= 45)
    if n_strong >= 2 and hf_ai_score >= 25:
        combined = max(combined, 58.0 + (n_strong - 2) * 12)

    return {
        "score": round(min(combined, 100.0), 1),
        "diffusion_spectrum_score": spec["score"],
        "multiscale_texture_score": tex["score"],
        "findings": findings,
        "details": {"spectrum": spec["details"], "texture": tex["details"]},
    }
