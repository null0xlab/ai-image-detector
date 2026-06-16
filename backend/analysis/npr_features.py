"""
Lightweight Neighboring Pixel Relationship (NPR) inspired features.

Based on: Tan et al., CVPR 2024 — NPR captures local pixel correlations from
upsampling in GAN/diffusion pipelines. Full detector uses trained ResNet weights;
here we use NPR statistics as a fast, metadata-free heuristic.

Reference: https://github.com/chuangchuangtan/NPR-DeepfakeDetection
"""

from __future__ import annotations

import cv2
import numpy as np


def analyze_npr_artifacts(cv_img_bgr: np.ndarray, is_messaging_compressed: bool = False) -> dict:
    """
    NPR-style neighbor difference statistics.

    Real camera photos typically show higher variance in horizontal/vertical
    neighbor residuals (sensor noise + natural texture). Heavy AI upsampling
    can produce more regular local correlations (lower NPR variance), but
    strong JPEG recompression also lowers variance — dampen penalties when
    is_messaging_compressed.
    """
    score = 0.0
    findings: list[str] = []

    try:
        gray = cv2.cvtColor(cv_img_bgr, cv2.COLOR_BGR2GRAY)
        size = min(384, gray.shape[0], gray.shape[1])
        g = cv2.resize(gray, (size, size), interpolation=cv2.INTER_AREA).astype(np.float32)

        # Horizontal and vertical neighbor differences (NPR at stride 1)
        dh = g[:, 1:] - g[:, :-1]
        dv = g[1:, :] - g[:-1, :]
        combined = np.concatenate([dh.flatten(), dv.flatten()])

        npr_std = float(np.std(combined))
        npr_kurt = float(np.mean((combined - np.mean(combined)) ** 4) / (np.var(combined) + 1e-8) - 3.0)

        # 2x2 block NPR (coarser grid, closer to paper's patch idea)
        block_nprs = []
        for r in range(0, size - 2, 2):
            for c in range(0, size - 2, 2):
                patch = g[r : r + 2, c : c + 2]
                if patch.shape == (2, 2):
                    block_nprs.append(float(patch[0, 1] - patch[0, 0]))
                    block_nprs.append(float(patch[1, 0] - patch[0, 0]))
        block_std = float(np.std(block_nprs)) if block_nprs else npr_std

        dampen = 0.55 if is_messaging_compressed else 1.0

        # Very low neighbor residual std → overly regular (AI-like)
        if npr_std < 9.5:
            penalty = (9.5 - npr_std) * 4.5 * dampen
            score += min(penalty, 35)
            findings.append(
                f"low neighbor-pixel residual variance (std={npr_std:.2f}; typical real photos: >11)"
            )

        if block_std < 7.0:
            penalty = (7.0 - block_std) * 3.5 * dampen
            score += min(penalty, 25)
            findings.append(
                f"regular 2x2 neighbor relationships (block_std={block_std:.2f})"
            )

        # Near-Gaussian NPR (low excess kurtosis) — common in synthetic smooth regions
        if npr_kurt < 0.5 and npr_std < 12.0:
            score += min((0.5 - npr_kurt) * 20 * dampen, 18)
            findings.append(f"Gaussian-like NPR distribution (kurtosis={npr_kurt:.2f})")

        return {
            "score": round(min(score, 100.0), 1),
            "npr_std": round(npr_std, 3),
            "npr_kurtosis": round(npr_kurt, 3),
            "block_npr_std": round(block_std, 3),
            "findings": findings,
        }
    except Exception as e:
        return {"score": 20.0, "error": str(e), "findings": []}
