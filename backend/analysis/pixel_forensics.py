"""
Advanced pixel-level forensics for detecting AI-generated and deepfake images
even when metadata (EXIF) has been stripped by messaging apps.

These signals work purely on pixel data and survive JPEG re-encoding, resizing,
and metadata stripping by Telegram, Messenger, WhatsApp, etc.

Techniques:
  1. PRNU (Photo Response Non-Uniformity) noise analysis via Wiener filter
  2. Chromatic aberration measurement (real lenses have this; AI does not)
  3. DCT coefficient distribution fitting (Benford-like natural image statistics)
  4. Wavelet subband uniformity (AI images have unnaturally smooth HF subbands)
  5. Noise floor spectral analysis (sensor noise vs generative noise)
  6. Local contrast entropy map (real photos have depth-of-field entropy variance)
  7. Banding/grid artifact detection (GAN upsampling checkerboard at pixel level)
"""

from __future__ import annotations

import cv2
import numpy as np
from PIL import Image
from scipy import ndimage
from scipy.stats import kurtosis, skew

from .npr_features import analyze_npr_artifacts


# ---------------------------------------------------------------------------
# 1. PRNU / Sensor Noise Analysis
# ---------------------------------------------------------------------------

def _wiener_denoise(gray: np.ndarray, size: int = 5) -> np.ndarray:
    """Fast Wiener-like filter using local mean/variance — CPU efficient."""
    gray_f = gray.astype(np.float32)
    local_mean = ndimage.uniform_filter(gray_f, size)
    local_sq_mean = ndimage.uniform_filter(gray_f ** 2, size)
    local_var = np.maximum(local_sq_mean - local_mean ** 2, 0.0)
    noise_var = float(np.mean(local_var))
    denoised = local_mean + np.maximum(local_var - noise_var, 0.0) / (local_var + 1e-6) * (gray_f - local_mean)
    return denoised.astype(np.float32)


def analyze_prnu_noise(cv_img_bgr: np.ndarray) -> dict:
    """
    Analyze sensor noise residual using Wiener filter denoising.
    
    Real cameras: PRNU creates spatially correlated noise with:
      - std in range 1.5–5.0
      - Non-Gaussian distribution (positive kurtosis)
      - Spatially non-uniform (high CV across blocks)
    
    AI-generated images:
      - Noise residual std < 1.2 (too clean) OR > 6.0 (compression artifacts)
      - Near-Gaussian or flat noise (kurtosis near 0)
      - Spatially uniform noise (low CV)
    """
    score = 0.0
    findings = []

    try:
        h, w = cv_img_bgr.shape[:2]
        work_size = min(512, min(h, w))
        img_small = cv2.resize(cv_img_bgr, (work_size, work_size), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(img_small, cv2.COLOR_BGR2GRAY).astype(np.float32)

        # Wiener filter denoising — more accurate than bilateral for PRNU
        denoised = _wiener_denoise(gray, size=7)
        residual = gray - denoised

        residual_std = float(np.std(residual))
        residual_flat = residual.flatten()

        # Kurtosis: camera noise has slight positive kurtosis (leptokurtic)
        # AI images: near 0 (normal) or negative (platykurtic, over-smooth)
        res_kurtosis = float(kurtosis(residual_flat))
        res_skewness = abs(float(skew(residual_flat)))

        # Block-level CV: real camera noise is spatially non-uniform
        block_size = work_size // 8
        block_stds = []
        for r in range(0, work_size - block_size + 1, block_size):
            for c in range(0, work_size - block_size + 1, block_size):
                block = residual[r:r + block_size, c:c + block_size]
                block_stds.append(float(np.std(block)))
        
        block_stds_arr = np.array(block_stds)
        noise_cv = float(np.std(block_stds_arr) / (np.mean(block_stds_arr) + 1e-6))

        # Spectral flatness of noise (camera sensor has non-flat noise spectrum)
        noise_fft = np.abs(np.fft.fft2(residual))
        noise_spectrum = noise_fft.flatten()
        spectral_flatness = float(
            np.exp(np.mean(np.log(noise_spectrum + 1e-10))) / (np.mean(noise_spectrum) + 1e-10)
        )

        # Score contributions
        # 1. Too-clean residual (AI image with no real sensor noise)
        if residual_std < 1.5:
            penalty = (1.5 - residual_std) * 30
            score += min(penalty, 40)
            findings.append(f"near-zero sensor noise residual (std={residual_std:.2f}; real cameras: >1.5)")

        # 2. Near-Gaussian kurtosis (AI noise is too Gaussian, camera noise is not)
        if res_kurtosis < 0.3:
            penalty = (0.3 - res_kurtosis) * 60
            score += min(penalty, 35)
            findings.append(f"Gaussian-flat noise distribution (kurtosis={res_kurtosis:.2f}; cameras: >0.5)")

        # 3. Spatially uniform noise (AI images lack depth-of-field noise variation)
        if noise_cv < 0.25:
            penalty = (0.25 - noise_cv) * 120
            score += min(penalty, 30)
            findings.append(f"spatially uniform noise pattern (CV={noise_cv:.2f}; cameras: >0.30)")

        # 4. Very high spectral flatness = white noise = AI compression artifact
        if spectral_flatness > 0.55:
            penalty = (spectral_flatness - 0.55) * 80
            score += min(penalty, 20)
            findings.append(f"spectrally flat noise (flatness={spectral_flatness:.2f}; typical of generative compression)")

        final_score = round(min(score, 100.0), 1)
        return {
            "score": final_score,
            "residual_std": round(residual_std, 3),
            "noise_kurtosis": round(res_kurtosis, 3),
            "noise_cv": round(noise_cv, 3),
            "spectral_flatness": round(spectral_flatness, 4),
            "findings": findings,
        }

    except Exception as e:
        return {"score": 30.0, "error": str(e), "findings": []}


# ---------------------------------------------------------------------------
# 2. Chromatic Aberration Analysis
# ---------------------------------------------------------------------------

def analyze_chromatic_aberration(cv_img_bgr: np.ndarray) -> dict:
    """
    Measure lens chromatic aberration in edge regions.
    
    Real camera lenses cause R and B channels to misalign slightly at edges
    (lateral chromatic aberration), especially at frame periphery. This is
    a physical optics property that AI models cannot replicate without explicit
    training on it — and most don't.
    
    Low CA shift score = suspicious (AI or CGI render).
    High CA shift score = more likely a real camera photo.
    """
    score = 0.0
    findings = []

    try:
        h, w = cv_img_bgr.shape[:2]
        work = cv2.resize(cv_img_bgr, (512, 512), interpolation=cv2.INTER_AREA)
        b, g, r = cv2.split(work)

        # Edge detection on green channel (highest resolution in Bayer sensors)
        gray = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 30, 100)

        # Focus on periphery (outer 20% of frame) where CA is strongest
        margin = 102  # ~20% of 512
        peri_mask = np.zeros_like(edges, dtype=np.uint8)
        peri_mask[:margin, :] = 1
        peri_mask[-margin:, :] = 1
        peri_mask[:, :margin] = 1
        peri_mask[:, -margin:] = 1
        peri_edges = (edges > 0) & (peri_mask > 0)

        if np.sum(peri_edges) < 50:
            return {"score": 25.0, "ca_shift": 0.0, "findings": ["insufficient edge pixels for CA analysis"]}

        # Per-edge-pixel: measure color channel gradient alignment
        peri_coords = np.argwhere(peri_edges)

        # Sample up to 500 edge pixels for speed
        if len(peri_coords) > 500:
            idx = np.random.choice(len(peri_coords), 500, replace=False)
            peri_coords = peri_coords[idx]

        ca_magnitudes = []
        for (py, px) in peri_coords:
            # Extract 5x5 neighborhood for each channel
            y0, y1 = max(0, py - 2), min(511, py + 3)
            x0, x1 = max(0, px - 2), min(511, px + 3)
            
            patch_r = r[y0:y1, x0:x1].astype(np.float32)
            patch_b = b[y0:y1, x0:x1].astype(np.float32)
            
            if patch_r.size < 4:
                continue
            
            # Centroid shift between R and B channels at this edge
            def centroid(p):
                total = p.sum() + 1e-8
                cy = (np.arange(p.shape[0])[:, None] * p).sum() / total
                cx = (np.arange(p.shape[1])[None, :] * p).sum() / total
                return cy, cx
            
            ry, rx = centroid(patch_r)
            by_, bx = centroid(patch_b)
            shift = np.sqrt((ry - by_) ** 2 + (rx - bx) ** 2)
            ca_magnitudes.append(float(shift))

        if not ca_magnitudes:
            return {"score": 25.0, "ca_shift": 0.0, "findings": ["CA computation failed"]}

        mean_ca = float(np.mean(ca_magnitudes))
        median_ca = float(np.median(ca_magnitudes))

        # Real camera lenses typically show CA shift of 0.15–0.80 pixels
        # AI images: near-zero CA (perfect synthetic optics)
        if mean_ca < 0.08:
            penalty = (0.08 - mean_ca) * 500
            score += min(penalty, 50)
            findings.append(f"virtually no chromatic aberration (mean_shift={mean_ca:.3f}px; real lenses: >0.10px)")
        elif mean_ca < 0.15:
            penalty = (0.15 - mean_ca) * 200
            score += min(penalty, 25)
            findings.append(f"minimal chromatic aberration (mean_shift={mean_ca:.3f}px)")

        final_score = round(min(score, 100.0), 1)
        return {
            "score": final_score,
            "ca_shift": round(mean_ca, 4),
            "ca_median": round(median_ca, 4),
            "findings": findings,
        }

    except Exception as e:
        return {"score": 20.0, "error": str(e), "findings": []}


# ---------------------------------------------------------------------------
# 3. DCT Natural Image Statistics
# ---------------------------------------------------------------------------

def analyze_dct_statistics(cv_img_bgr: np.ndarray) -> dict:
    """
    Analyze DCT coefficient distributions against natural image statistics.
    
    Natural photographs follow predictable statistical laws in DCT space:
    - AC coefficients follow a Laplacian (double-exponential) distribution
    - Coefficient histograms follow Benford's law
    - High-frequency to low-frequency energy ratio follows ~1/f² power law
    
    AI-generated images subtly deviate from these natural image statistics.
    """
    score = 0.0
    findings = []

    try:
        gray = cv2.cvtColor(cv_img_bgr, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape

        # Work on up to 512x512
        size = min(512, h, w)
        gray_small = cv2.resize(gray, (size, size), interpolation=cv2.INTER_AREA)

        # Collect DCT coefficients from 8x8 blocks
        ac_coeffs = []
        dc_coeffs = []
        for r in range(0, size - 7, 8):
            for c in range(0, size - 7, 8):
                block = gray_small[r:r + 8, c:c + 8].astype(np.float64)
                dct_block = cv2.dct(block)
                dc_coeffs.append(float(dct_block[0, 0]))
                # AC coefficients (exclude DC at [0,0])
                ac_flat = dct_block.flatten()[1:]
                ac_coeffs.extend(ac_flat.tolist())

        ac_arr = np.array(ac_coeffs, dtype=np.float64)
        if len(ac_arr) < 500:
            return {"score": 25.0, "findings": ["insufficient blocks for DCT analysis"]}

        # Test 1: Laplacian fit for AC coefficients
        # Natural images: AC coefficients fit Laplacian very well (low MSE)
        # AI images: can be over-quantized (too few unique values) or too smooth
        ac_abs = np.abs(ac_arr)
        laplacian_scale = float(np.mean(ac_abs))  # MLE estimate for Laplacian scale
        
        # Generate expected Laplacian histogram
        hist, bin_edges = np.histogram(ac_arr, bins=64, density=True)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        laplacian_pdf = (1.0 / (2 * laplacian_scale + 1e-10)) * np.exp(
            -np.abs(bin_centers) / (laplacian_scale + 1e-10)
        )
        
        # Normalize both to sum to 1
        hist_norm = hist / (hist.sum() + 1e-10)
        lap_norm = laplacian_pdf / (laplacian_pdf.sum() + 1e-10)
        
        # Chi-squared-like goodness of fit
        laplacian_fit_error = float(np.mean((hist_norm - lap_norm) ** 2)) * 1000

        # Test 2: Unique value ratio for quantization analysis
        # Heavy quantization (too few unique AC values) = AI export artifact
        unique_ratio = len(np.unique(np.round(ac_arr, 1))) / max(len(ac_arr), 1)

        # Test 3: Power spectral density slope
        # 2D FFT of the full image
        fft2 = np.fft.fft2(gray_small.astype(np.float64))
        fft_shifted = np.fft.fftshift(fft2)
        power = np.abs(fft_shifted) ** 2 + 1e-10

        # Radially averaged power spectrum
        cy, cx = size // 2, size // 2
        y_idx, x_idx = np.mgrid[:size, :size]
        radii = np.sqrt((x_idx - cx) ** 2 + (y_idx - cy) ** 2).astype(int)
        max_r = min(cy, cx)
        
        radial_power = []
        for r_val in range(1, max_r):
            mask = radii == r_val
            if np.any(mask):
                radial_power.append(float(np.mean(power[mask])))

        if len(radial_power) > 10:
            log_r = np.log(np.arange(1, len(radial_power) + 1))
            log_p = np.log(np.array(radial_power) + 1e-10)
            
            # Linear fit in log-log space: slope should be approximately -2 for natural images
            coeffs = np.polyfit(log_r, log_p, 1)
            psd_slope = float(coeffs[0])
            
            # Natural images: slope around -1.8 to -2.5
            # AI images: can be too flat (-1.0 to -1.5) or too steep (< -3.0)
            if psd_slope > -1.5:
                penalty = (-1.5 - psd_slope) * -40  # flatter than expected
                score += min(penalty, 30)
                findings.append(f"anomalous power spectral slope ({psd_slope:.2f}; natural images: -1.8 to -2.5)")
            elif psd_slope < -3.0:
                penalty = (psd_slope + 3.0) * -30
                score += min(penalty, 20)
                findings.append(f"over-steep power spectral slope ({psd_slope:.2f}; possible AI over-sharpening)")
        else:
            psd_slope = -2.0

        # Test 4: DC coefficient variance (natural images have high DC variance)
        dc_var = float(np.var(dc_coeffs))
        if dc_var < 200:
            penalty = (200 - dc_var) * 0.08
            score += min(penalty, 20)
            findings.append(f"low DC coefficient variance ({dc_var:.1f}; unnaturally uniform brightness blocks)")

        # Combine findings
        if laplacian_fit_error > 8.0:
            penalty = (laplacian_fit_error - 8.0) * 1.5
            score += min(penalty, 25)
            findings.append(f"AC coefficients deviate from natural Laplacian distribution (error={laplacian_fit_error:.2f})")

        final_score = round(min(score, 100.0), 1)
        return {
            "score": final_score,
            "psd_slope": round(psd_slope, 3),
            "laplacian_fit_error": round(laplacian_fit_error, 3),
            "dc_variance": round(dc_var, 1),
            "findings": findings,
        }

    except Exception as e:
        return {"score": 25.0, "error": str(e), "findings": []}


# ---------------------------------------------------------------------------
# 4. Wavelet Subband Uniformity Analysis
# ---------------------------------------------------------------------------

def analyze_wavelet_statistics(cv_img_bgr: np.ndarray) -> dict:
    """
    Analyze wavelet subband statistics for AI image detection.
    
    Uses a simple Haar wavelet decomposition (implementable with pure NumPy/SciPy).
    
    AI-generated images have unnaturally uniform high-frequency wavelet subbands
    across different spatial regions because generative models apply spatially
    uniform operations during upsampling/synthesis.
    
    Real photos: strong spatial variation in HF subbands due to:
    - Depth-of-field (sharp foreground, blurry background)
    - Sensor noise variation with local brightness
    - Texture variation across different materials
    """
    score = 0.0
    findings = []

    try:
        gray = cv2.cvtColor(cv_img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
        h, w = gray.shape
        size = min(256, h, w)
        # Ensure even dimensions for Haar wavelet
        size = size - (size % 2)
        gray_small = cv2.resize(gray, (size, size), interpolation=cv2.INTER_AREA)

        def haar_2d(img):
            """Single-level 2D Haar wavelet decomposition."""
            h_rows = (img[::2, :] + img[1::2, :]) / 2.0  # Low-pass rows
            d_rows = (img[::2, :] - img[1::2, :]) / 2.0  # High-pass rows
            LL = (h_rows[:, ::2] + h_rows[:, 1::2]) / 2.0
            LH = (h_rows[:, ::2] - h_rows[:, 1::2]) / 2.0  # Horizontal edges
            HL = (d_rows[:, ::2] + d_rows[:, 1::2]) / 2.0  # Vertical edges
            HH = (d_rows[:, ::2] - d_rows[:, 1::2]) / 2.0  # Diagonal edges
            return LL, LH, HL, HH

        # Two-level Haar decomposition
        LL1, LH1, HL1, HH1 = haar_2d(gray_small)
        LL2, LH2, HL2, HH2 = haar_2d(LL1)

        # Analyze spatial uniformity of high-frequency subbands
        def spatial_uniformity(subband: np.ndarray, n_blocks: int = 4) -> float:
            """CV of block-level energy in the subband."""
            sh, sw = subband.shape
            bh, bw = max(1, sh // n_blocks), max(1, sw // n_blocks)
            block_energies = []
            for r in range(0, sh - bh + 1, bh):
                for c in range(0, sw - bw + 1, bw):
                    block = subband[r:r + bh, c:c + bw]
                    block_energies.append(float(np.mean(block ** 2)))
            arr = np.array(block_energies)
            return float(np.std(arr) / (np.mean(arr) + 1e-8))

        # Level-1 subbands
        lh1_cv = spatial_uniformity(LH1)
        hl1_cv = spatial_uniformity(HL1)
        hh1_cv = spatial_uniformity(HH1)

        # Level-2 subbands
        lh2_cv = spatial_uniformity(LH2)
        hl2_cv = spatial_uniformity(HL2)
        hh2_cv = spatial_uniformity(HH2)

        mean_cv_l1 = (lh1_cv + hl1_cv + hh1_cv) / 3.0
        mean_cv_l2 = (lh2_cv + hl2_cv + hh2_cv) / 3.0

        # Real photos: spatially non-uniform HF content (CV > 0.45 typically)
        # AI images: spatially uniform HF content (CV < 0.30)
        if mean_cv_l1 < 0.35:
            penalty = (0.35 - mean_cv_l1) * 150
            score += min(penalty, 40)
            findings.append(f"uniform level-1 HF wavelet subbands (CV={mean_cv_l1:.2f}; real photos: >0.40)")

        if mean_cv_l2 < 0.40:
            penalty = (0.40 - mean_cv_l2) * 120
            score += min(penalty, 35)
            findings.append(f"uniform level-2 HF wavelet subbands (CV={mean_cv_l2:.2f}; real photos: >0.45)")

        # HH subband kurtosis: AI images have platykurtic HH (too smooth high-freq diagonals)
        hh1_flat = HH1.flatten()
        hh_kurt = float(kurtosis(hh1_flat))
        if hh_kurt < 1.5:
            penalty = (1.5 - hh_kurt) * 15
            score += min(penalty, 20)
            findings.append(f"low HH diagonal subband kurtosis ({hh_kurt:.2f}; real camera: >2.0)")

        final_score = round(min(score, 100.0), 1)
        return {
            "score": final_score,
            "wavelet_cv_l1": round(mean_cv_l1, 3),
            "wavelet_cv_l2": round(mean_cv_l2, 3),
            "hh_kurtosis": round(hh_kurt, 3),
            "findings": findings,
        }

    except Exception as e:
        return {"score": 25.0, "error": str(e), "findings": []}


# ---------------------------------------------------------------------------
# 5. GAN Grid / Upsampling Artifact Detection
# ---------------------------------------------------------------------------

def analyze_grid_artifacts(cv_img_bgr: np.ndarray) -> dict:
    """
    Detect GAN/diffusion upsampling grid artifacts at pixel level.
    
    Many GAN architectures (PGGAN, StyleGAN, etc.) and older diffusion upsampling
    networks leave periodic grid artifacts at 8x8, 16x16, or 32x32 intervals.
    These are invisible to the naked eye but detectable in the residual after
    denoising and in the FFT power spectrum.
    """
    score = 0.0
    findings = []

    try:
        gray = cv2.cvtColor(cv_img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
        h, w = gray.shape
        size = min(512, h, w)
        gray_small = cv2.resize(gray, (size, size), interpolation=cv2.INTER_AREA)

        # Compute noise residual
        blurred = cv2.GaussianBlur(gray_small, (5, 5), 1.0)
        residual = gray_small - blurred

        # FFT of residual
        fft2 = np.fft.fft2(residual)
        fft_shifted = np.fft.fftshift(fft2)
        power = np.abs(fft_shifted) ** 2

        cy, cx = size // 2, size // 2

        # Check for periodic peaks at frequencies corresponding to grid artifacts
        # GAN grid artifacts at period P appear at frequency size/P in FFT
        total_power = np.mean(power)
        grid_scores = []

        for period in [8, 16, 32]:
            freq = size // period
            if freq <= 0 or freq >= cy:
                continue

            # Check horizontal, vertical, and diagonal grid frequencies
            h_freq_power = float(np.mean(power[cy - 2:cy + 3, cx + freq - 2:cx + freq + 3]))
            v_freq_power = float(np.mean(power[cy + freq - 2:cy + freq + 3, cx - 2:cx + 3]))

            h_ratio = h_freq_power / (total_power + 1e-10)
            v_ratio = v_freq_power / (total_power + 1e-10)
            peak_ratio = max(h_ratio, v_ratio)
            grid_scores.append((period, peak_ratio))

            if peak_ratio > 5.0:
                penalty = (peak_ratio - 5.0) * 8
                score += min(penalty, 30)
                findings.append(f"periodic grid artifact at {period}px spacing (power ratio={peak_ratio:.1f}x)")

        # Also check for checkerboard pattern (strong at Nyquist frequency)
        nyquist_region = power[cy - 3:cy + 4, cx - 3:cx + 4]
        nyquist_power = float(np.mean(np.abs(fft2[size // 2 - 3:size // 2 + 4, size // 2 - 3:size // 2 + 4]) ** 2))
        nyquist_ratio = nyquist_power / (total_power + 1e-10)
        if nyquist_ratio > 8.0:
            score += min((nyquist_ratio - 8.0) * 5, 25)
            findings.append(f"checkerboard artifact at Nyquist frequency (ratio={nyquist_ratio:.1f}x)")

        final_score = round(min(score, 100.0), 1)
        return {
            "score": final_score,
            "grid_analysis": {str(p): round(r, 3) for p, r in grid_scores},
            "nyquist_ratio": round(nyquist_ratio, 3),
            "findings": findings,
        }

    except Exception as e:
        return {"score": 0.0, "error": str(e), "findings": []}


# ---------------------------------------------------------------------------
# 6. Local Contrast Entropy Analysis
# ---------------------------------------------------------------------------

def analyze_local_contrast_entropy(cv_img_bgr: np.ndarray) -> dict:
    """
    Measure spatial entropy of local contrast variations.
    
    Real photos have significant spatial variation in contrast entropy due to:
    - Depth of field (sharp vs blurry regions)
    - Motion blur
    - Uneven lighting
    - Different material textures
    
    AI images apply contrast/sharpness uniformly — every region ends up with
    similar local contrast entropy.
    """
    score = 0.0
    findings = []

    try:
        gray = cv2.cvtColor(cv_img_bgr, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        size = min(256, h, w)
        gray_small = cv2.resize(gray, (size, size), interpolation=cv2.INTER_AREA)

        # Compute local Laplacian variance map (sharpness map)
        lap = cv2.Laplacian(gray_small.astype(np.float32), cv2.CV_32F)
        
        # Block-wise local variance of Laplacian
        block_size = size // 8
        local_vars = []
        for r in range(0, size - block_size + 1, block_size):
            for c in range(0, size - block_size + 1, block_size):
                block = lap[r:r + block_size, c:c + block_size]
                local_vars.append(float(np.var(block)))

        local_vars_arr = np.array(local_vars)
        
        # CV of local variance: high = spatially non-uniform (real photo)
        local_contrast_cv = float(np.std(local_vars_arr) / (np.mean(local_vars_arr) + 1e-6))

        # Log-range of local variances: real photos span multiple orders of magnitude
        if np.any(local_vars_arr > 0):
            log_range = float(np.log10(np.max(local_vars_arr) + 1) - np.log10(np.min(local_vars_arr) + 1))
        else:
            log_range = 0.0

        # Entropy of the local variance distribution
        hist, _ = np.histogram(local_vars_arr, bins=16)
        hist_norm = hist / (hist.sum() + 1e-10)
        hist_norm = hist_norm[hist_norm > 0]
        entropy = float(-np.sum(hist_norm * np.log2(hist_norm + 1e-10)))

        # Real photos: high CV (>0.8), high log-range (>2.0), moderate entropy
        # AI images: low CV (<0.5), low log-range (<1.2), concentrated entropy
        if local_contrast_cv < 0.6:
            penalty = (0.6 - local_contrast_cv) * 70
            score += min(penalty, 35)
            findings.append(f"uniformly distributed local contrast (CV={local_contrast_cv:.2f}; real photos: >0.70)")

        if log_range < 1.5:
            penalty = (1.5 - log_range) * 20
            score += min(penalty, 25)
            findings.append(f"narrow local sharpness range ({log_range:.2f} log-decades; real photos: >2.0)")

        final_score = round(min(score, 100.0), 1)
        return {
            "score": final_score,
            "local_contrast_cv": round(local_contrast_cv, 3),
            "sharpness_log_range": round(log_range, 3),
            "local_contrast_entropy": round(entropy, 3),
            "findings": findings,
        }

    except Exception as e:
        return {"score": 20.0, "error": str(e), "findings": []}


# ---------------------------------------------------------------------------
# 7. SRM (Steganalysis Rich Model) Noise Residual Analysis
# ---------------------------------------------------------------------------

# 30-kernel SRM filter bank subset — captures diverse noise residual directions
_SRM_KERNELS = [
    # 1st-order horizontal / vertical
    np.array([[0, 0, 0], [0, 1, -1], [0, 0, 0]], dtype=np.float32),
    np.array([[0, 0, 0], [0, 1, 0], [0, -1, 0]], dtype=np.float32),
    np.array([[0, 0, 0], [-1, 1, 0], [0, 0, 0]], dtype=np.float32),
    np.array([[0, 0, 0], [0, 1, 0], [0, 0, -1]], dtype=np.float32),
    # 2nd-order (Laplacian-family)
    np.array([[-1, 2, -1], [2, -4, 2], [-1, 2, -1]], dtype=np.float32) / 4.0,
    np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], dtype=np.float32) / 4.0,
    np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], dtype=np.float32) / 4.0,
    # Diagonal
    np.array([[1, 0, -1], [0, 0, 0], [-1, 0, 1]], dtype=np.float32) / 2.0,
    np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32) / 4.0,
    # Edge emphasis
    np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype=np.float32) / 8.0,
]


def analyze_srm_noise_residual(cv_img_bgr: np.ndarray) -> dict:
    """
    SRM (Steganalysis Rich Model) filter bank noise analysis.

    Generative models produce characteristic noise residual patterns in SRM space:
    - AI images: low SRM residual STD, low cross-filter variability
    - Deepfakes: high SRM anomaly in face region vs surroundings
    - Real cameras: moderate SRM residual with high spatial variability

    This is one of the most robust signals for metadata-stripped images as it
    operates entirely on pixel statistics that survive JPEG recompression.
    """
    score = 0.0
    findings = []

    try:
        h, w = cv_img_bgr.shape[:2]
        size = min(384, h, w)
        work = cv2.resize(cv_img_bgr, (size, size), interpolation=cv2.INTER_AREA)
        gray_f = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY).astype(np.float32)

        # Apply all SRM kernels and collect residual statistics
        residual_stds: list[float] = []
        residual_skews: list[float] = []
        for kernel in _SRM_KERNELS:
            res = cv2.filter2D(gray_f, -1, kernel)
            residual_stds.append(float(np.std(res)))
            flat = res.flatten()
            if len(flat) > 10:
                from scipy.stats import skew as _skew
                residual_skews.append(abs(float(_skew(flat))))

        srm_stds = np.array(residual_stds, dtype=np.float64)
        srm_mean = float(np.mean(srm_stds))
        srm_std_of_stds = float(np.std(srm_stds))
        srm_max = float(np.max(srm_stds))

        # Real cameras: moderate SRM mean (3-10), high variability between filters
        # AI generated: very low SRM mean (<2.5) OR very uniform across filters (low std_of_stds)
        if srm_mean < 2.5:
            penalty = (2.5 - srm_mean) * 20
            score += min(penalty, 40)
            findings.append(
                f"very low SRM noise residual (mean={srm_mean:.2f}; real cameras: >3.0) — AI over-smoothing"
            )

        # AI-generated images have unnaturally uniform noise across filter types
        if srm_std_of_stds < 0.8 and srm_mean < 6.0:
            penalty = (0.8 - srm_std_of_stds) * 25
            score += min(penalty, 30)
            findings.append(
                f"uniform SRM response across filter types (std={srm_std_of_stds:.2f}; real: >1.0)"
            )

        # Spatial non-uniformity of SRM residual (key for deepfakes: face≠background)
        # Use the Laplacian SRM kernel for spatial analysis
        lap_kernel = _SRM_KERNELS[5]  # 2nd-order Laplacian-family
        lap_res = cv2.filter2D(gray_f, -1, lap_kernel)

        block_size = size // 8
        block_stds = []
        for r in range(0, size - block_size + 1, block_size):
            for c in range(0, size - block_size + 1, block_size):
                block_stds.append(float(np.std(lap_res[r:r + block_size, c:c + block_size])))

        block_arr = np.array(block_stds)
        srm_spatial_cv = float(np.std(block_arr) / (np.mean(block_arr) + 1e-8))

        # Real photos: high spatial non-uniformity of SRM residual (CV > 0.45)
        # AI images: spatially uniform SRM residual (CV < 0.30)
        if srm_spatial_cv < 0.30:
            penalty = (0.30 - srm_spatial_cv) * 80
            score += min(penalty, 30)
            findings.append(
                f"spatially uniform SRM residual (CV={srm_spatial_cv:.2f}; real photos: >0.40)"
            )

        final_score = round(min(score, 100.0), 1)
        return {
            "score": final_score,
            "srm_mean": round(srm_mean, 3),
            "srm_std_of_stds": round(srm_std_of_stds, 3),
            "srm_spatial_cv": round(srm_spatial_cv, 3),
            "findings": findings,
        }

    except Exception as e:
        return {"score": 20.0, "error": str(e), "findings": []}


# ---------------------------------------------------------------------------
# Master function: run all pixel forensics
# ---------------------------------------------------------------------------

def _pixel_forensics_weights(is_messaging_compressed: bool) -> dict[str, float]:
    # PRNU, CA, and SRM are the most reliable for stripped-metadata images.
    if is_messaging_compressed:
        return {
            "prnu": 0.22,
            "chromatic_aberration": 0.16,
            "dct_statistics": 0.14,
            "wavelet": 0.11,
            "grid_artifacts": 0.07,
            "local_contrast": 0.06,
            "srm_noise": 0.14,
            "npr": 0.10,
        }
    return {
        "prnu": 0.20,
        "chromatic_aberration": 0.17,
        "dct_statistics": 0.15,
        "wavelet": 0.12,
        "grid_artifacts": 0.10,
        "local_contrast": 0.08,
        "srm_noise": 0.10,
        "npr": 0.08,
    }


def _run_pixel_forensics_single(cv_img_bgr: np.ndarray, is_messaging_compressed: bool) -> dict:
    """
    Run all pixel-level forensic analyses on a single crop.
    Kept separate so we can average multiple crops for robustness.
    """
    prnu = analyze_prnu_noise(cv_img_bgr)
    ca = analyze_chromatic_aberration(cv_img_bgr)
    dct = analyze_dct_statistics(cv_img_bgr)
    wavelet = analyze_wavelet_statistics(cv_img_bgr)
    grid = analyze_grid_artifacts(cv_img_bgr)
    contrast = analyze_local_contrast_entropy(cv_img_bgr)
    srm = analyze_srm_noise_residual(cv_img_bgr)
    npr = analyze_npr_artifacts(cv_img_bgr, is_messaging_compressed=is_messaging_compressed)

    scores = {
        "prnu": prnu["score"],
        "chromatic_aberration": ca["score"],
        "dct_statistics": dct["score"],
        "wavelet": wavelet["score"],
        "grid_artifacts": grid["score"],
        "local_contrast": contrast["score"],
        "srm_noise": srm["score"],
        "npr": npr["score"],
    }

    # Messenger JPEG recompression causes false PRNU/SRM/NPR positives — dampen ensemble
    if is_messaging_compressed:
        for key in ("prnu", "srm_noise", "npr", "wavelet"):
            scores[key] = round(scores[key] * 0.72, 1)

    weights = _pixel_forensics_weights(is_messaging_compressed)

    combined_score = sum(scores[k] * weights[k] for k in scores)

    strong_signals = [k for k, v in scores.items() if v >= 40]
    n_strong = len(strong_signals)

    # Consensus boost: multiple signals agreeing increases confidence
    if n_strong >= 4:
        combined_score = max(combined_score, 62.0 + (n_strong - 4) * 6)
    elif n_strong >= 3:
        combined_score = max(combined_score, 45.0 + (n_strong - 3) * 8)

    all_findings = []
    for module, result in [
        ("PRNU", prnu),
        ("ChromaticAberration", ca),
        ("DCT", dct),
        ("Wavelet", wavelet),
        ("GridArtifacts", grid),
        ("LocalContrast", contrast),
        ("SRM", srm),
        ("NPR", npr),
    ]:
        for finding in result.get("findings", []):
            all_findings.append(f"[{module}] {finding}")

    final_score = round(min(combined_score, 100.0), 1)

    return {
        "score": final_score,
        "signal_scores": scores,
        "strong_signals": strong_signals,
        "strong_signal_count": n_strong,
        "all_findings": all_findings,
        "details": {
            "prnu": prnu,
            "chromatic_aberration": ca,
            "dct_statistics": dct,
            "wavelet": wavelet,
            "grid_artifacts": grid,
            "local_contrast": contrast,
            "srm_noise": srm,
            "npr": npr,
        },
    }


def run_pixel_forensics(
    cv_img_bgr: np.ndarray,
    is_messaging_compressed: bool = False,
    n_crops: int = 2,
) -> dict:
    """
    Run all pixel-level forensics and return a combined score.

    Uses small test-time augmentation (a couple of crops) so the verdict
    doesn't depend too heavily on a single region after re-encoding.
    """
    try:
        h, w = cv_img_bgr.shape[:2]
        if n_crops <= 1 or min(h, w) < 180:
            return _run_pixel_forensics_single(cv_img_bgr, is_messaging_compressed)

        crops: list[np.ndarray] = []
        # Pass 1: full frame.
        crops.append(cv_img_bgr)

        # Pass 2: center crop (keeps optics/noise evidence in the most stable region).
        crop_scale = 0.82 if is_messaging_compressed else 0.88
        cw = int(w * crop_scale)
        ch = int(h * crop_scale)
        cw = max(140, min(cw, w))
        ch = max(140, min(ch, h))
        x0 = (w - cw) // 2
        y0 = (h - ch) // 2
        x1 = x0 + cw
        y1 = y0 + ch
        crops.append(cv_img_bgr[y0:y1, x0:x1])

        # Pass 3 (optional): horizontal flip of full frame.
        # This often stabilizes PRNU/noise-residual metrics after re-encoding.
        if n_crops >= 3:
            crops.append(cv2.flip(cv_img_bgr, 1))

        # Average signal scores across crops.
        per_pass = [
            _run_pixel_forensics_single(c, is_messaging_compressed)
            for c in crops[:n_crops]
        ]

        avg_signal_scores: dict[str, float] = {}
        signal_keys = per_pass[0]["signal_scores"].keys()
        for k in signal_keys:
            avg_signal_scores[k] = float(np.mean([p["signal_scores"][k] for p in per_pass]))

        weights = _pixel_forensics_weights(is_messaging_compressed)
        combined_score = sum(avg_signal_scores[k] * weights[k] for k in avg_signal_scores)

        strong_signals = [k for k, v in avg_signal_scores.items() if v >= 40]
        n_strong = len(strong_signals)

        if n_strong >= 3:
            combined_score = max(combined_score, 45.0 + (n_strong - 3) * 8)
        elif n_strong >= 4:
            combined_score = max(combined_score, 60.0)

        # Findings: union/cap to keep response small.
        all_findings = []
        for p in per_pass:
            all_findings.extend(p.get("all_findings", []))
        # Deduplicate while preserving order.
        seen = set()
        dedup = []
        for f in all_findings:
            if f not in seen:
                seen.add(f)
                dedup.append(f)
        all_findings = dedup[:14]

        final_score = round(min(combined_score, 100.0), 1)

        # Provide averaged "details" as the first pass (fine for UI) to avoid payload bloat.
        # Pixel-level details are mainly for debugging; the core numbers are returned above.
        return {
            "score": final_score,
            "signal_scores": avg_signal_scores,
            "strong_signals": strong_signals,
            "strong_signal_count": n_strong,
            "all_findings": all_findings,
            "details": per_pass[0].get("details", {}),
        }
    except Exception as e:
        fallback = _run_pixel_forensics_single(cv_img_bgr, is_messaging_compressed)
        fallback["error"] = str(e)
        return fallback
