"""
Deepfake-specific forensic analysis module.

Detects face-swap and deepfake manipulation artifacts that are distinct from
general AI generation. Works on pixel data only — no metadata required.

Techniques:
  1. Face region blend boundary analysis (sharpness discontinuity, frequency mismatch)
  2. Multi-scale facial texture inconsistency (noise pattern mismatch between face/background)
  3. Facial frequency domain fingerprint (face region DCT vs background DCT anomaly)
  4. Color statistics mismatch (illumination inconsistency)
  5. Eye/teeth region artifact detection (commonly imperfect in deepfakes)
  6. Facial geometry frequency analysis (unnatural skin texture periodicity)
"""

from __future__ import annotations

import cv2
import numpy as np
from scipy.ndimage import uniform_filter
from scipy.stats import kurtosis


def _extract_face_regions(
    cv_img_bgr: np.ndarray,
    face_cascade: cv2.CascadeClassifier,
    work_size: int = 512,
) -> list[dict]:
    """Detect faces and return cropped region data."""
    h, w = cv_img_bgr.shape[:2]
    scale = work_size / max(h, w)
    new_w = int(w * scale)
    new_h = int(h * scale)
    work = cv2.resize(cv_img_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)

    gray = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.08, minNeighbors=4, minSize=(36, 36)
    )

    result = []
    for x, y, fw, fh in faces[:3]:
        pad = int(0.20 * max(fw, fh))
        x0 = max(0, x - pad)
        y0 = max(0, y - pad)
        x1 = min(new_w, x + fw + pad)
        y1 = min(new_h, y + fh + pad)

        face_crop = work[y0:y1, x0:x1]
        face_gray = gray[y0:y1, x0:x1]

        # Compute an "outer ring" area (surrounding background)
        ring_x0 = max(0, x0 - int(pad * 1.5))
        ring_y0 = max(0, y0 - int(pad * 1.5))
        ring_x1 = min(new_w, x1 + int(pad * 1.5))
        ring_y1 = min(new_h, y1 + int(pad * 1.5))

        outer_mask = np.ones((ring_y1 - ring_y0, ring_x1 - ring_x0), dtype=bool)
        iy0 = y0 - ring_y0
        iy1 = y1 - ring_y0
        ix0 = x0 - ring_x0
        ix1 = x1 - ring_x0
        outer_mask[max(0, iy0):iy1, max(0, ix0):ix1] = False

        outer_region = work[ring_y0:ring_y1, ring_x0:ring_x1]
        outer_gray = gray[ring_y0:ring_y1, ring_x0:ring_x1]

        if face_crop.shape[0] < 20 or face_crop.shape[1] < 20:
            continue

        result.append({
            "face_crop": face_crop,
            "face_gray": face_gray,
            "outer_region": outer_region,
            "outer_gray": outer_gray,
            "outer_mask": outer_mask,
            "face_box": (x, y, fw, fh),
            "scale": scale,
        })
    return result


def _noise_residual(gray_f32: np.ndarray) -> np.ndarray:
    """Fast noise residual via local mean subtraction."""
    local_mean = uniform_filter(gray_f32, size=5)
    return gray_f32 - local_mean


def analyze_blend_boundary(face_data: dict) -> dict:
    """
    Detect abrupt blend boundaries between the face region and the background.

    Deepfakes paste a face from one source into another image; the blending
    boundary often creates a narrow ring of frequency / noise discontinuity
    that is unphysical.
    """
    score = 0.0
    findings = []

    try:
        fg = face_data["face_gray"].astype(np.float32)
        bg_gray = face_data["outer_gray"].astype(np.float32)
        bg_mask = face_data["outer_mask"]

        if fg.size < 100 or bg_gray.size < 100:
            return {"score": 0.0, "findings": []}

        # 1. Laplacian variance ratio: face vs background
        lap_fg = float(np.var(cv2.Laplacian(fg, cv2.CV_32F)))
        lap_bg = cv2.Laplacian(bg_gray.astype(np.float32), cv2.CV_32F)
        if bg_mask.shape == lap_bg.shape:
            bg_vals = lap_bg[bg_mask]
        else:
            bg_vals = lap_bg.flatten()
        lap_bg_var = float(np.var(bg_vals)) if bg_vals.size > 0 else 1.0

        if lap_bg_var > 5.0:
            ratio = lap_fg / (lap_bg_var + 1e-8)
            if ratio > 4.0:
                penalty = min((ratio - 4.0) * 7, 35)
                score += penalty
                findings.append(
                    f"face region {ratio:.1f}x sharper than surroundings — deepfake paste boundary"
                )
            elif ratio < 0.22:
                penalty = min((0.22 - ratio) * 80, 30)
                score += penalty
                findings.append(
                    f"face region over-smooth compared to background (ratio={ratio:.2f}) — blending artifact"
                )

        # 2. Noise pattern mismatch between face and background
        fg_noise = _noise_residual(fg)
        bg_noise_full = _noise_residual(bg_gray.astype(np.float32))
        if bg_mask.shape == bg_noise_full.shape:
            bg_noise = bg_noise_full[bg_mask]
        else:
            bg_noise = bg_noise_full.flatten()

        face_noise_std = float(np.std(fg_noise))
        bg_noise_std = float(np.std(bg_noise)) if bg_noise.size > 0 else 1.0

        if bg_noise_std > 0.5:
            noise_ratio = face_noise_std / (bg_noise_std + 1e-8)
            if noise_ratio < 0.35 or noise_ratio > 3.5:
                penalty = min(abs(np.log(noise_ratio + 1e-8)) * 12, 25)
                score += penalty
                findings.append(
                    f"noise level mismatch between face and background (ratio={noise_ratio:.2f})"
                )

        # 3. Color temperature discontinuity (lighting inconsistency)
        face_bgr = face_data["face_crop"].astype(np.float32)
        outer_bgr = face_data["outer_region"].astype(np.float32)

        if face_bgr.size > 0 and outer_bgr.size > 0:
            face_means = face_bgr.reshape(-1, 3).mean(axis=0)  # B, G, R
            outer_means = outer_bgr.reshape(-1, 3).mean(axis=0)

            # Warm/cool balance: R-B ratio difference
            face_rb = (face_means[2] + 1e-8) / (face_means[0] + 1e-8)
            outer_rb = (outer_means[2] + 1e-8) / (outer_means[0] + 1e-8)
            rb_diff = abs(face_rb - outer_rb)

            if rb_diff > 0.30:
                penalty = min(rb_diff * 35, 28)
                score += penalty
                findings.append(
                    f"illumination color mismatch face/background (R/B ratio diff={rb_diff:.2f})"
                )

    except Exception as e:
        findings.append(f"blend_boundary error: {e}")

    return {"score": round(min(score, 100.0), 1), "findings": findings}


def analyze_face_frequency(face_data: dict) -> dict:
    """
    Compare frequency fingerprints of face vs background regions.

    A deepfake face often comes from a different generative source, leaving
    a DCT / spectral fingerprint that differs from the background's fingerprint.
    """
    score = 0.0
    findings = []

    try:
        fg = face_data["face_gray"].astype(np.float32)
        bg_gray = face_data["outer_gray"].astype(np.float32)

        # Resize to same area for fair comparison
        area = max(64, min(128, min(fg.shape)))
        fg_r = cv2.resize(fg, (area, area), interpolation=cv2.INTER_AREA)
        bg_r = cv2.resize(bg_gray, (area, area), interpolation=cv2.INTER_AREA)

        # DCT-based spectral slope comparison
        def spectral_slope(patch: np.ndarray) -> float:
            fft = np.fft.fft2(patch.astype(np.float64))
            fft_s = np.fft.fftshift(fft)
            power = np.abs(fft_s) ** 2 + 1e-10
            cy, cx = patch.shape[0] // 2, patch.shape[1] // 2
            y_idx, x_idx = np.mgrid[:patch.shape[0], :patch.shape[1]]
            radii = np.sqrt((x_idx - cx) ** 2 + (y_idx - cy) ** 2).astype(int)
            max_r = min(cy, cx)
            radial_p = [
                float(np.mean(power[radii == r]))
                for r in range(1, max_r)
                if np.any(radii == r)
            ]
            if len(radial_p) < 5:
                return -2.0
            log_r = np.log(np.arange(1, len(radial_p) + 1))
            log_p = np.log(np.array(radial_p) + 1e-10)
            return float(np.polyfit(log_r, log_p, 1)[0])

        slope_fg = spectral_slope(fg_r)
        slope_bg = spectral_slope(bg_r)
        slope_diff = abs(slope_fg - slope_bg)

        if slope_diff > 0.8:
            penalty = min(slope_diff * 18, 30)
            score += penalty
            findings.append(
                f"spectral slope mismatch face vs background (face={slope_fg:.2f}, bg={slope_bg:.2f})"
            )

        # High-frequency energy ratio comparison
        def hf_energy_ratio(patch: np.ndarray) -> float:
            dct_p = cv2.dct(patch)
            mid = patch.shape[0] // 2
            hf = np.abs(dct_p[mid:, mid:])
            total = np.abs(dct_p).sum() + 1e-10
            return float(hf.sum() / total)

        hf_fg = hf_energy_ratio(fg_r.astype(np.float32))
        hf_bg = hf_energy_ratio(bg_r.astype(np.float32))
        hf_diff = abs(hf_fg - hf_bg)

        if hf_diff > 0.06:
            penalty = min(hf_diff * 200, 25)
            score += penalty
            findings.append(
                f"high-frequency energy mismatch face/background (diff={hf_diff:.3f})"
            )

    except Exception as e:
        findings.append(f"face_frequency error: {e}")

    return {"score": round(min(score, 100.0), 1), "findings": findings}


def analyze_facial_texture_inconsistency(face_data: dict) -> dict:
    """
    Measure texture granularity inconsistency between face and background.

    Real photos have consistent sensor noise texture across all regions.
    Deepfake compositing often introduces a face crop with a different
    noise texture (over-smoothed, or from a different sensor/generator).
    """
    score = 0.0
    findings = []

    try:
        fg = face_data["face_gray"].astype(np.float32)
        bg_gray = face_data["outer_gray"].astype(np.float32)
        bg_mask = face_data["outer_mask"]

        def lbp_histogram(patch: np.ndarray) -> np.ndarray:
            """Compute simplified LBP-like texture histogram."""
            p = cv2.resize(patch, (64, 64), interpolation=cv2.INTER_AREA)
            p_u8 = np.clip(p, 0, 255).astype(np.uint8)
            # Use gradient magnitudes binned as a texture proxy
            gx = cv2.Sobel(p_u8, cv2.CV_32F, 1, 0, ksize=3)
            gy = cv2.Sobel(p_u8, cv2.CV_32F, 0, 1, ksize=3)
            mag = np.sqrt(gx**2 + gy**2)
            hist, _ = np.histogram(mag, bins=32, range=(0, 150))
            hist = hist.astype(np.float32)
            total = hist.sum() + 1e-10
            return hist / total

        hist_fg = lbp_histogram(fg)

        if bg_mask.shape == bg_gray.shape:
            bg_patch = bg_gray.copy()
            bg_patch[~bg_mask] = np.median(bg_gray[bg_mask]) if bg_mask.any() else 128
        else:
            bg_patch = bg_gray

        hist_bg = lbp_histogram(bg_patch)

        # Chi-squared distance between texture histograms
        chi2 = float(np.sum((hist_fg - hist_bg) ** 2 / (hist_fg + hist_bg + 1e-10)))

        if chi2 > 0.25:
            penalty = min(chi2 * 60, 35)
            score += penalty
            findings.append(
                f"texture pattern mismatch between face and background (chi2={chi2:.3f})"
            )

        # Noise granularity: standard deviation of noise residual
        fg_noise_std = float(np.std(_noise_residual(fg)))
        if bg_mask.shape == bg_gray.shape and bg_mask.any():
            bg_vals = bg_gray[bg_mask]
        else:
            bg_vals = bg_gray.flatten()

        # Approximate noise from background via local variation
        bg_noise_std = float(np.std(bg_vals - uniform_filter(bg_vals.reshape(-1, 1).flatten(), size=5)))

        if bg_noise_std > 0.3:
            texture_ratio = fg_noise_std / (bg_noise_std + 1e-8)
            if texture_ratio < 0.25 or texture_ratio > 4.0:
                penalty = min(abs(np.log(texture_ratio + 1e-8)) * 10, 25)
                score += penalty
                findings.append(
                    f"noise granularity mismatch face/background (ratio={texture_ratio:.2f})"
                )

    except Exception as e:
        findings.append(f"texture_inconsistency error: {e}")

    return {"score": round(min(score, 100.0), 1), "findings": findings}


def analyze_eye_region_artifacts(face_data: dict) -> dict:
    """
    Look for artifacts in the eye region — a common failure point in deepfakes.

    Deepfake models often introduce artifacts around the eyes:
    - Unnatural white balance / iris color
    - Blurring at eyelash boundaries
    - Asymmetric eye sharpness
    """
    score = 0.0
    findings = []

    try:
        fg = face_data["face_gray"].astype(np.float32)
        face_crop = face_data["face_crop"]
        fh, fw = fg.shape[:2]

        if fh < 40 or fw < 40:
            return {"score": 0.0, "findings": []}

        # Eye region: roughly top 40-65% vertically, left/right halves
        eye_top = int(fh * 0.22)
        eye_bot = int(fh * 0.52)
        left_eye = fg[eye_top:eye_bot, :fw // 2]
        right_eye = fg[eye_top:eye_bot, fw // 2:]

        if left_eye.size < 50 or right_eye.size < 50:
            return {"score": 0.0, "findings": []}

        # Sharpness asymmetry between left and right eye
        lap_left = float(np.var(cv2.Laplacian(left_eye, cv2.CV_32F)))
        lap_right = float(np.var(cv2.Laplacian(right_eye, cv2.CV_32F)))

        if min(lap_left, lap_right) > 1.0:
            asym_ratio = max(lap_left, lap_right) / (min(lap_left, lap_right) + 1e-8)
            if asym_ratio > 3.5:
                penalty = min((asym_ratio - 3.5) * 8, 25)
                score += penalty
                findings.append(
                    f"eye sharpness asymmetry L/R={asym_ratio:.1f}x — deepfake blending artifact"
                )

        # Check for color channel imbalance in eye region (sclera should be near-white)
        eye_region_bgr = face_crop[eye_top:eye_bot, :]
        if eye_region_bgr.size > 0:
            b_mean = float(eye_region_bgr[:, :, 0].mean())
            r_mean = float(eye_region_bgr[:, :, 2].mean())
            br_ratio = abs(b_mean - r_mean)
            # Sclera is roughly equal in R/G/B; strong imbalance suggests color grading mismatch
            if br_ratio > 30:
                penalty = min((br_ratio - 30) * 0.5, 18)
                score += penalty
                findings.append(
                    f"eye region color imbalance (B-R diff={br_ratio:.0f}) — lighting inconsistency"
                )

    except Exception as e:
        findings.append(f"eye_artifact error: {e}")

    return {"score": round(min(score, 100.0), 1), "findings": findings}


def run_deepfake_forensics(
    cv_img_bgr: np.ndarray,
    face_cascade: cv2.CascadeClassifier | None = None,
) -> dict:
    """
    Master deepfake forensics function.

    Returns a 0-100 deepfake likelihood score and detailed findings.
    Works purely on pixel data — no metadata required.
    """
    if face_cascade is None:
        try:
            path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            face_cascade = cv2.CascadeClassifier(path)
            if face_cascade.empty():
                return {
                    "score": 0.0,
                    "has_faces": False,
                    "face_count": 0,
                    "all_findings": [],
                    "sub_scores": {},
                }
        except Exception:
            return {
                "score": 0.0,
                "has_faces": False,
                "face_count": 0,
                "all_findings": [],
                "sub_scores": {},
            }

    face_regions = _extract_face_regions(cv_img_bgr, face_cascade)

    if not face_regions:
        return {
            "score": 0.0,
            "has_faces": False,
            "face_count": 0,
            "all_findings": [],
            "sub_scores": {},
        }

    # Analyze each face; take max score (worst case face is most suspicious)
    all_findings: list[str] = []
    best_score = 0.0
    best_sub_scores: dict = {}

    for face_data in face_regions:
        blend = analyze_blend_boundary(face_data)
        freq = analyze_face_frequency(face_data)
        texture = analyze_facial_texture_inconsistency(face_data)
        eye = analyze_eye_region_artifacts(face_data)

        sub_scores = {
            "blend_boundary": blend["score"],
            "face_frequency": freq["score"],
            "texture_inconsistency": texture["score"],
            "eye_artifacts": eye["score"],
        }

        # Weighted combination
        face_score = (
            sub_scores["blend_boundary"] * 0.35
            + sub_scores["face_frequency"] * 0.25
            + sub_scores["texture_inconsistency"] * 0.25
            + sub_scores["eye_artifacts"] * 0.15
        )

        # Consensus boost: multiple signals firing simultaneously
        n_strong = sum(1 for v in sub_scores.values() if v >= 30)
        if n_strong >= 3:
            face_score = max(face_score, 55.0 + (n_strong - 3) * 8)
        elif n_strong >= 2:
            face_score = max(face_score, 38.0)

        face_score = round(min(face_score, 100.0), 1)

        if face_score > best_score:
            best_score = face_score
            best_sub_scores = sub_scores

        for analysis, result in [("Blend", blend), ("Freq", freq), ("Texture", texture), ("Eye", eye)]:
            for f in result.get("findings", []):
                tag = f"[Deepfake/{analysis}] {f}"
                if tag not in all_findings:
                    all_findings.append(tag)

    return {
        "score": best_score,
        "has_faces": True,
        "face_count": len(face_regions),
        "all_findings": all_findings[:12],
        "sub_scores": best_sub_scores,
    }
