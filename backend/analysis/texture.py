import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern


def _co_occurrence_entropy(gray_16: np.ndarray) -> float:
    """Fast single-direction co-occurrence entropy (horizontal)."""
    shifted = np.roll(gray_16, -1, axis=1)
    pairs = gray_16[:, :-1].flatten() * 16 + shifted[:, :-1].flatten()
    hist = np.bincount(pairs.astype(np.int32), minlength=256).astype(np.float64)
    hist /= hist.sum() + 1e-10
    entropy = -np.sum(hist[hist > 0] * np.log2(hist[hist > 0]))
    return float(entropy)


def _multi_scale_lbp_entropy(gray: np.ndarray) -> float:
    """Average LBP entropy across two scales for robustness."""
    entropies = []
    for r, p in [(1, 8), (2, 16)]:
        try:
            lbp = local_binary_pattern(gray, P=p, R=r, method="uniform")
            n_bins = p + 2
            hist, _ = np.histogram(lbp, bins=n_bins, range=(0, n_bins), density=True)
            ent = -float(np.sum(hist * np.log2(hist + 1e-12)))
            entropies.append(ent)
        except Exception:
            pass
    return float(np.mean(entropies)) if entropies else 2.5


def _diffusion_periodicity_score(gray: np.ndarray) -> float:
    """
    Detect subtle periodic patterns in texture typical of diffusion upsampling.
    Diffusion models upsample features at fixed strides, leaving faint grid patterns
    at 8px / 16px that are detectable in the autocorrelation of the texture.
    """
    size = min(256, gray.shape[0], gray.shape[1])
    patch = cv2.resize(gray, (size, size), interpolation=cv2.INTER_AREA).astype(np.float32)
    patch -= patch.mean()

    # 2D autocorrelation via FFT
    fft = np.fft.fft2(patch)
    ac = np.real(np.fft.ifft2(fft * np.conj(fft)))
    ac = np.fft.fftshift(ac)
    ac /= ac.max() + 1e-10

    cy, cx = size // 2, size // 2
    score = 0.0
    for stride in (8, 16, 32):
        if stride >= cy:
            continue
        # Check autocorrelation peaks at multiples of stride (1–3 periods)
        peak_vals = []
        for mult in (1, 2, 3):
            d = stride * mult
            if d < cy:
                # Sample 4 directions
                for dy, dx in [(d, 0), (-d, 0), (0, d), (0, -d)]:
                    py, px = cy + dy, cx + dx
                    if 0 <= py < size and 0 <= px < size:
                        peak_vals.append(float(ac[py, px]))
        if peak_vals:
            max_peak = max(peak_vals)
            if max_peak > 0.15:
                score += min((max_peak - 0.15) * 80, 30)

    return round(min(score, 60.0), 2)


def analyze_texture(
    cv_img_bgr: np.ndarray,
    is_web_downloaded: bool = False,
    is_messaging_compressed: bool = False,
) -> dict:
    """
    Applies statistical texture and noise analysis to detect synthetic surfaces.
    Uses:
        - GLCM (Gray-Level Co-occurrence Matrix) for homogeneity & contrast.
        - Multi-scale LBP for surface microstructure entropy.
        - Spatial noise variance analysis for camera sensor organic noise detection.
        - Diffusion periodicity detection (autocorrelation grid peak analysis).
        - Co-occurrence entropy for diffusion vs GAN discrimination.
    Returns:
        score: float (0 - 100)
        details: str
    """
    try:
        gray = cv2.cvtColor(cv_img_bgr, cv2.COLOR_BGR2GRAY)
        gray_resized = cv2.resize(gray, (384, 384), interpolation=cv2.INTER_AREA)

        # --- 1. GLCM Feature Extraction ---
        gray_16 = (gray_resized // 16).astype(np.uint8)
        glcm = graycomatrix(
            gray_16,
            distances=[1],
            angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
            levels=16,
            symmetric=True,
            normed=True,
        )
        homogeneity = float(np.mean(graycoprops(glcm, "homogeneity")))
        contrast = float(np.mean(graycoprops(glcm, "contrast")))
        energy = float(np.mean(graycoprops(glcm, "energy")))

        # Co-occurrence entropy — AI images have lower co-occurrence entropy
        cooc_entropy = _co_occurrence_entropy(gray_16)

        # --- 2. Multi-scale LBP Texture Microstructure ---
        lbp_entropy = _multi_scale_lbp_entropy(gray_resized)

        # --- 3. Sensor Noise Residual Analysis ---
        laplacian = cv2.Laplacian(gray_resized, cv2.CV_64F)
        block_size = 48
        blocks_x = gray_resized.shape[1] // block_size
        blocks_y = gray_resized.shape[0] // block_size
        block_variances = []
        for i in range(blocks_y):
            for j in range(blocks_x):
                block = laplacian[
                    i * block_size : (i + 1) * block_size,
                    j * block_size : (j + 1) * block_size,
                ]
                block_variances.append(float(np.var(block)))

        block_variances_arr = np.array(block_variances)
        mean_var = float(np.mean(block_variances_arr))
        std_var = float(np.std(block_variances_arr))
        noise_cv = std_var / (mean_var + 1e-8)

        # --- 4. Diffusion Periodicity ---
        periodicity_score = _diffusion_periodicity_score(gray_resized)

        # --- 5. Scoring ---
        findings = []
        texture_score = 0.0

        # Over-smoothness — tuned for DALL-E 3 / ChatGPT polished surfaces
        if homogeneity > 0.60 and contrast < 1.8:
            smoothness_penalty = min(
                (homogeneity - 0.58) * 160 + max(1.8 - contrast, 0) * 22, 85
            )
            texture_score += smoothness_penalty * 0.35
            findings.append("unnatural spatial smoothness (hyper-homogeneous surface textures)")

        # High GLCM energy with low contrast = diffusion-model skin/sky
        if energy > 0.42 and contrast < 2.2:
            texture_score += min((energy - 0.42) * 120, 40) * 0.16
            findings.append(
                f"low-contrast high-energy texture tiling (GLCM energy: {energy:.2f})"
            )

        # Low co-occurrence entropy: AI images have unnaturally predictable adjacent pixel patterns
        if cooc_entropy < 5.5:
            penalty = min((5.5 - cooc_entropy) * 15, 30)
            texture_score += penalty * 0.12
            findings.append(
                f"low co-occurrence entropy ({cooc_entropy:.2f}) — predictable texture repetition"
            )

        # Multi-scale LBP Entropy — modern AI still shows reduced microstructure entropy
        if lbp_entropy < 2.55:
            lbp_penalty = min((2.55 - lbp_entropy) * 130, 75)
            texture_score += lbp_penalty * 0.28
            findings.append(
                f"highly repetitive microstructures (multi-scale LBP entropy: {lbp_entropy:.2f})"
            )

        # Noise profile check
        if mean_var < 8.0:
            sterile_penalty = min((8.0 - mean_var) * 5 + 20, 50)
            texture_score += sterile_penalty * 0.32
            findings.append(
                "sterile noise profile (complete lack of organic camera sensor noise)"
            )
        elif noise_cv < 0.38:
            uniform_penalty = min((0.38 - noise_cv) * 180, 65)
            texture_score += uniform_penalty * 0.28
            findings.append(
                f"synthetically uniform noise distribution (noise variance CV: {noise_cv:.2f})"
            )

        # Diffusion periodicity artifacts
        if periodicity_score > 12:
            texture_score += min(periodicity_score * 0.5, 20)
            findings.append(
                f"periodic autocorrelation peaks consistent with diffusion upsampling (score={periodicity_score:.1f})"
            )

        if not findings:
            details = (
                "Textures and noise profiles are organic. Microstructure shows natural variation "
                "and typical camera sensor noise."
            )
        else:
            details = "Anomalies detected: " + ", ".join(findings) + "."

        final_score = max(min(texture_score * 1.65, 100.0), 0.0)
        if is_web_downloaded:
            final_score *= 0.50
        if is_messaging_compressed:
            final_score *= 0.58

        return {
            "score": round(final_score, 1),
            "details": details,
            "glcm": {
                "homogeneity": round(homogeneity, 3),
                "contrast": round(contrast, 3),
                "energy": round(energy, 3),
            },
            "lbp_entropy": round(lbp_entropy, 3),
            "cooc_entropy": round(cooc_entropy, 3),
            "noise_cv": round(noise_cv, 3),
            "periodicity_score": round(periodicity_score, 2),
        }

    except Exception as e:
        return {
            "score": 50.0,
            "details": f"Error running texture analysis: {str(e)}",
        }
