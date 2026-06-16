"""
Heuristics tuned for modern generative exports (DALL-E 3, ChatGPT, Midjourney, Flux).
These models produce photorealistic images that often pass legacy forensic checks.
"""
import io
import cv2
import numpy as np
from PIL import Image
from scipy.ndimage import uniform_filter

AI_METADATA_HINTS = [
    "openai", "dall-e", "dalle", "chatgpt", "midjourney", "stable diffusion",
    "adobe firefly", "flux", "bing image", "generative", "ai generated",
    "gemini", "google imagen", "imagen", "nano banana", "vertex",
]


def _block_laplacian_variances(gray: np.ndarray, block_size: int = 32) -> np.ndarray:
    gray64 = np.asarray(gray, dtype=np.float64)
    lap = cv2.Laplacian(gray64, cv2.CV_64F)
    h, w = gray.shape
    vars_ = []
    for r in range(0, h - block_size + 1, block_size):
        for c in range(0, w - block_size + 1, block_size):
            block = lap[r : r + block_size, c : c + block_size]
            vars_.append(np.var(block))
    return np.array(vars_, dtype=np.float64)


def _scan_embedded_metadata(pil_img: Image.Image, image_bytes: bytes) -> str | None:
    """Look for generator hints in PNG/WebP text chunks or EXIF software fields."""
    try:
        for key, val in (pil_img.info or {}).items():
            blob = f"{key} {val}".lower()
            for hint in AI_METADATA_HINTS:
                if hint in blob:
                    return f"{key}: {str(val)[:120]}"
    except Exception:
        pass

    try:
        # Scan raw file bytes (PNG tEXt/iTXt, JPEG COM markers, WebP metadata, etc.)
        sample = image_bytes[: min(len(image_bytes), 256_000)]
        text = sample.decode("latin-1", errors="ignore").lower()
        for hint in AI_METADATA_HINTS:
            if hint in text:
                fmt = "PNG" if image_bytes[:8] == b"\x89PNG\r\n\x1a\n" else "file"
                return f"{fmt} embedded hint: '{hint}'"
    except Exception:
        pass
    return None


def analyze_generative(
    cv_img_bgr: np.ndarray,
    pil_img: Image.Image,
    image_bytes: bytes,
    is_messaging_compressed: bool = False,
) -> dict:
    """
    Detects statistical signatures common in diffusion-model exports:
    uniform global sharpness, hyper-saturated grading, missing lens chromatic aberration,
    and dual-depth sharpness (subject + distant landmarks both crisp).
    """
    findings = []
    score = 0.0

    try:
        meta_hit = _scan_embedded_metadata(pil_img, image_bytes)
        if meta_hit:
            return {
                "score": 100.0,
                "details": f"Embedded generator metadata detected ({meta_hit}).",
            }

        gray_full = cv2.cvtColor(cv_img_bgr, cv2.COLOR_BGR2GRAY)
        work = cv2.resize(cv_img_bgr, (512, 512), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(work, cv2.COLOR_BGR2HSV)
        h, w = gray.shape

        block_vars = _block_laplacian_variances(gray, 32)
        mean_var = float(np.mean(block_vars))
        std_var = float(np.std(block_vars))
        sharpness_cv = std_var / (mean_var + 1e-8)

        # Real photos: depth-of-field creates strong sharpness variation (CV > 0.42).
        # Diffusion renders: globally even micro-contrast (CV < 0.40).
        if sharpness_cv < 0.50:
            penalty = min((0.50 - sharpness_cv) * 200, 80)
            score += penalty * 0.28
            findings.append(
                f"unnaturally uniform global sharpness (micro-contrast CV: {sharpness_cv:.2f})"
            )

        # Saturation & luminance grading typical of DALL-E / Midjourney exports
        sat = hsv[:, :, 1].astype(np.float32)
        val = hsv[:, :, 2].astype(np.float32)
        sat_mean = float(np.mean(sat))
        sat_std = float(np.std(sat))
        if sat_mean > 95 and sat_std < 52:
            sat_penalty = min((sat_mean - 95) * 0.35 + (52 - sat_std) * 0.6, 55)
            score += sat_penalty * 0.18
            findings.append(
                f"hyper-saturated synthetic color grading (mean saturation {sat_mean:.0f})"
            )

        highlight_frac = float(np.mean(val > 248))
        if highlight_frac > 0.008 and highlight_frac < 0.18:
            score += min(highlight_frac * 280, 30) * 0.10
            findings.append("controlled highlight clipping typical of generative tone-mapping")

        # Smooth-surface fraction: large areas with very low local variance but visible structure
        blur_var = cv2.GaussianBlur(gray, (7, 7), 0).astype(np.float64)
        diff_sq = (gray.astype(np.float64) - blur_var) ** 2
        local_var = uniform_filter(diff_sq, size=15)
        smooth_mask = local_var < 12.0
        smooth_frac = float(np.mean(smooth_mask))
        lap_on_smooth = float(np.mean(np.abs(cv2.Laplacian(gray, cv2.CV_32F))[smooth_mask])) if np.any(smooth_mask) else 0.0
        if smooth_frac > 0.22 and lap_on_smooth < 8.5:
            smooth_penalty = min((smooth_frac - 0.22) * 180 + (8.5 - lap_on_smooth) * 4, 65)
            score += smooth_penalty * 0.22
            findings.append(
                f"over-smoothed surface regions covering {smooth_frac * 100:.0f}% of the frame"
            )

        # Chromatic aberration proxy: real lenses misalign R/B edges at frame periphery
        b, g, r = cv2.split(work)
        edges_g = cv2.Canny(gray, 40, 120)
        margin = max(8, min(h, w) // 16)
        border_mask = np.zeros_like(edges_g)
        border_mask[:margin, :] = 1
        border_mask[-margin:, :] = 1
        border_mask[:, :margin] = 1
        border_mask[:, -margin:] = 1
        border_edges = edges_g & border_mask
        if np.sum(border_edges) > 80:
            er = cv2.Canny(r, 40, 120) & border_mask
            eb = cv2.Canny(b, 40, 120) & border_mask
            ca_shift = abs(int(np.sum(er)) - int(np.sum(eb))) / (np.sum(border_edges) + 1e-8)
            if ca_shift < 0.12:
                score += 38 * 0.14
                findings.append(
                    "missing lens chromatic aberration at periphery (optically perfect render)"
                )

        # Dual-depth sharpness: subject AND distant background both unusually sharp
        center = gray[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4]
        peri_mask = np.zeros_like(gray, dtype=np.uint8)
        peri_mask[: h // 5, :] = 1
        peri_mask[-h // 5 :, :] = 1
        peri_mask[:, : w // 5] = 1
        peri_mask[:, -w // 5 :] = 1
        lap_full = cv2.Laplacian(gray, cv2.CV_64F)
        center_sharp = float(np.var(lap_full[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4]))
        peri_vals = lap_full[peri_mask > 0]
        peri_sharp = float(np.var(peri_vals)) if peri_vals.size else 0.0
        if center_sharp > 60 and peri_sharp > center_sharp * 0.45:
            dual_penalty = min((peri_sharp / (center_sharp + 1e-8)) * 55, 50)
            score += dual_penalty * 0.18
            findings.append(
                "foreground and distant background both hyper-sharp (non-physical depth of field)"
            )

        # Full-resolution noise residual: generative images lack sensor PRNU fingerprint
        hi_w = min(1024, gray_full.shape[1])
        hi_h = min(1024, gray_full.shape[0])
        gray_hi = cv2.resize(gray_full, (hi_w, hi_h), interpolation=cv2.INTER_AREA)
        gray_u8 = gray_hi.astype(np.uint8)
        denoised = cv2.bilateralFilter(gray_u8, 5, 50, 50)
        residual = gray_hi.astype(np.float32) - denoised.astype(np.float32)
        residual_std = float(np.std(residual))
        residual_blocks = _block_laplacian_variances(np.abs(residual), 48)
        residual_cv = float(np.std(residual_blocks) / (np.mean(np.abs(residual_blocks)) + 1e-8))
        if residual_std < 2.8 or residual_cv < 0.35:
            prnu_penalty = min((2.8 - residual_std) * 12 + (0.35 - residual_cv) * 90, 55)
            score += max(prnu_penalty, 0) * 0.20
            findings.append(
                "absent camera sensor PRNU noise fingerprint (statistically sterile pixels)"
            )

        # Face boundary consistency check: deepfakes often have subtle blending artifacts
        # at the boundary between the swapped face and the original photo background
        face_cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        face_cascade = cv2.CascadeClassifier(face_cascade_path)
        if not face_cascade.empty():
            gray_detect = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(
                gray_detect, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40)
            )
            if len(faces) > 0:
                for (fx, fy, fw, fh) in faces[:2]:
                    # Analyze the face region vs a ring around it (boundary check)
                    pad = max(8, int(0.12 * max(fw, fh)))
                    face_roi = gray[fy:fy + fh, fx:fx + fw]
                    
                    # Outer ring (just outside face boundary)
                    oy0 = max(0, fy - pad)
                    oy1 = min(h, fy + fh + pad)
                    ox0 = max(0, fx - pad)
                    ox1 = min(w, fx + fw + pad)
                    outer = gray[oy0:oy1, ox0:ox1]
                    
                    if face_roi.size > 100 and outer.size > 100:
                        face_lap = float(np.var(cv2.Laplacian(face_roi, cv2.CV_64F)))
                        outer_lap = float(np.var(cv2.Laplacian(outer, cv2.CV_64F)))
                        
                        # Deepfakes: face region has VERY different sharpness from surroundings
                        # Either too sharp (face pasted in) or too smooth (face blended in)
                        if outer_lap > 10:
                            sharpness_ratio = face_lap / (outer_lap + 1e-8)
                            if sharpness_ratio > 3.5:
                                penalty = min((sharpness_ratio - 3.5) * 8, 35)
                                score += penalty * 0.20
                                findings.append(
                                    f"face region {sharpness_ratio:.1f}x sharper than surroundings "
                                    "(deepfake paste boundary)"
                                )
                            elif sharpness_ratio < 0.28:
                                penalty = min((0.28 - sharpness_ratio) * 100, 30)
                                score += penalty * 0.18
                                findings.append(
                                    f"face region {sharpness_ratio:.2f}x smoother than surroundings "
                                    "(deepfake over-smooth boundary)"
                                )
                        
                        # Color temperature inconsistency: face vs background
                        face_color = work[fy:fy + fh, fx:fx + fw]
                        outer_color = work[oy0:oy1, ox0:ox1]
                        if face_color.size > 0 and outer_color.size > 0:
                            face_mean = face_color.astype(np.float32).mean(axis=(0, 1))
                            outer_mean = outer_color.astype(np.float32).mean(axis=(0, 1))
                            # B/R ratio difference between face and background
                            face_br = face_mean[0] / (face_mean[2] + 1e-8)
                            outer_br = outer_mean[0] / (outer_mean[2] + 1e-8)
                            br_diff = abs(face_br - outer_br)
                            if br_diff > 0.25:
                                score += min(br_diff * 25, 20) * 0.15
                                findings.append(
                                    f"face/background color temperature mismatch "
                                    f"(B/R ratio diff={br_diff:.2f})"
                                )

        if not findings:
            details = (
                "No strong generative-export signatures detected in color grading, sharpness "
                "uniformity, sensor noise structure, or face boundary consistency."
            )
        else:
            details = "Generative indicators: " + ", ".join(findings) + "."

        final_score = max(min(score * 1.55, 100.0), 0.0)
        n_findings = len(findings)
        # Heuristic findings alone are unreliable on messenger JPEG — no consensus boost here
        # Chat-app JPEG dulls some cues; mild dampen only for weak scores (keep strong AI signals)
        if is_messaging_compressed and final_score < 85:
            if final_score >= 58:
                final_score *= 0.90
            elif final_score >= 45:
                final_score *= 0.82
            else:
                final_score *= 0.72
        return {"score": round(final_score, 1), "details": details}

    except Exception as e:
        return {"score": 50.0, "details": f"Error running generative analysis: {str(e)}"}
