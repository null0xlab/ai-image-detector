import cv2
import numpy as np
import base64
from scipy.ndimage import maximum_filter

def analyze_frequency(cv_img_bgr: np.ndarray) -> dict:
    """
    Applies Fast Fourier Transform (FFT) to analyze frequency domain characteristics.
    Detects upsampling/grid artifacts (typical of GAN/Diffusion generators) 
    and checks if the spectral slope deviates from natural image laws.
    Returns:
        score: float (0 - 100)
        details: str
        fft_base64: str (base64 PNG of FFT magnitude spectrum for UI visualization)
    """
    try:
        # Convert to grayscale
        gray = cv2.cvtColor(cv_img_bgr, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        
        # Resize to a standard square size (e.g., 512x512) for consistent frequency bin sizes
        size = 512
        gray_resized = cv2.resize(gray, (size, size), interpolation=cv2.INTER_AREA)
        
        # Apply Hanning window to reduce boundary effects/leakage
        window = np.hanning(size)[:, None] * np.hanning(size)[None, :]
        gray_windowed = (gray_resized.astype(float) - 127.5) * window
        
        # Compute 2D Fast Fourier Transform
        f_transform = np.fft.fft2(gray_windowed)
        f_shift = np.fft.fftshift(f_transform)
        
        # Calculate Magnitude Spectrum (Log scale for visualization)
        magnitude = np.abs(f_shift)
        magnitude_log = 20 * np.log(magnitude + 1)
        
        # Normalize magnitude to 0-255 for export
        mag_min, mag_max = magnitude_log.min(), magnitude_log.max()
        if mag_max > mag_min:
            mag_img = ((magnitude_log - mag_min) / (mag_max - mag_min) * 255).astype(np.uint8)
        else:
            mag_img = np.zeros((size, size), dtype=np.uint8)
            
        # Encode magnitude spectrum image to Base64
        _, buffer = cv2.imencode(".png", mag_img)
        fft_base64 = f"data:image/png;base64,{base64.b64encode(buffer).decode('utf-8')}"
        
        # --- 1. Grid Peak Detection ---
        # AI images leave periodic "stars" or grid peaks in the high-frequency spectrum.
        # We mask the central low-frequency region (the natural high-energy hub) and search for high-frequency peaks.
        center = size // 2
        mask_radius = 40  # Mask out central 40 pixels radius
        
        y_grid, x_grid = np.ogrid[-center:size-center, -center:size-center]
        high_freq_mask = x_grid**2 + y_grid**2 > mask_radius**2
        
        high_freq_mag = magnitude * high_freq_mask
        
        # Detect local peaks using a maximum filter
        local_max = maximum_filter(high_freq_mag, size=15) == high_freq_mag
        # Keep only peaks above a relative threshold
        threshold = np.mean(high_freq_mag[high_freq_mask]) + 3.2 * np.std(high_freq_mag[high_freq_mask])
        peaks = local_max & (high_freq_mag > threshold) & high_freq_mask
        
        num_peaks = np.sum(peaks)
        
        # --- 2. Radially Averaged Power Spectrum (RAPS) ---
        # Natural images exhibit a spectral decay following 1 / f^alpha where alpha ≈ 2
        # Synthetic images often show elevated high frequencies or structural anomalies (flattening)
        r_indices = np.round(np.sqrt(x_grid**2 + y_grid**2)).astype(int)
        r_max = center
        
        # Calculate the average energy at each radial frequency bin
        radial_averages = []
        frequencies = []
        for r in range(1, r_max):
            radial_mask = r_indices == r
            if np.any(radial_mask):
                avg_energy = np.mean(magnitude[radial_mask])
                radial_averages.append(avg_energy)
                frequencies.append(r)
                
        radial_averages = np.array(radial_averages)
        frequencies = np.array(frequencies)
        
        # Fit linear regression to log-log RAPS (excluding very low and very high bins to avoid windowing/leakage artifacts)
        fit_start = 5
        fit_end = int(r_max * 0.75)
        
        log_freq = np.log(frequencies[fit_start:fit_end])
        log_energy = np.log(radial_averages[fit_start:fit_end] + 1e-8)
        
        # Compute slope (alpha) and R-squared error
        slope, intercept = np.polyfit(log_freq, log_energy, 1)
        
        # Calculate fitting error (RMSE)
        predictions = slope * log_freq + intercept
        rmse = np.sqrt(np.mean((log_energy - predictions) ** 2))
        
        # --- 3. Scoring Synthesis ---
        # Penalize if:
        # A. Symmetrical grid peak count is high (strong indicator of generator upsampling checkerboard artifacts)
        # B. Power spectrum slope is too flat (slope > -1.2, meaning excess high-frequency noise)
        # C. High regression fitting error (RMSE > 0.40, indicating non-power-law synthetic frequency clusters)
        
        peak_score = min(float(num_peaks) * 12.0, 100.0)

        # Slope scoring: optimal is between -1.8 and -2.4. Flattening (towards 0) is synthetic.
        if slope > -1.55:
            slope_score = min((slope - (-1.55)) * 90 + 35, 100)
        elif slope < -3.0:
            slope_score = min(((-3.0) - slope) * 50 + 20, 100)
        else:
            slope_score = 0.0

        rmse_score = min(rmse * 200, 100.0)

        # Mid/high frequency excess (diffusion models elevate non power-law bands)
        mid_mask = (r_indices > 25) & (r_indices < int(r_max * 0.55))
        high_mask = r_indices > int(r_max * 0.55)
        low_mask = (r_indices > 5) & (r_indices < 25)
        mid_energy = float(np.mean(magnitude[mid_mask])) if np.any(mid_mask) else 0.0
        high_energy = float(np.mean(magnitude[high_mask])) if np.any(high_mask) else 0.0
        low_energy = float(np.mean(magnitude[low_mask])) if np.any(low_mask) else 1e-8
        mid_high_ratio = (mid_energy + high_energy) / (low_energy + 1e-8)
        ratio_score = min(max((mid_high_ratio - 0.85) * 55, 0), 70) if mid_high_ratio > 0.85 else 0.0

        frequency_score = (
            peak_score * 0.38
            + slope_score * 0.22
            + rmse_score * 0.22
            + ratio_score * 0.18
        )
        
        # Frame details based on findings
        findings = []
        if num_peaks > 2:
            findings.append(f"detected {num_peaks} high-frequency periodic grid spikes (indicative of generator upsampling checkers)")
        if slope > -1.4:
            findings.append(f"elevated high-frequency energy profile (slope {slope:.2f} deviates from natural 1/f^2 spectrum)")
        if rmse > 0.30:
            findings.append(f"atypical spectral fluctuations (Fourier fit error: {rmse:.3f})")
        if mid_high_ratio > 0.95:
            findings.append(
                f"elevated mid/high-frequency energy vs low band (ratio {mid_high_ratio:.2f})"
            )
            
        if not findings:
            details = "Frequency distribution is organic. Follows standard natural power-law decay with no periodic upsampling artifacts."
        else:
            details = "Anomalies detected: " + ", ".join(findings) + "."
            
        return {
            "score": round(min(frequency_score, 100.0), 1),
            "details": details,
            "fft_base64": fft_base64
        }
        
    except Exception as e:
        return {
            "score": 50.0,
            "details": f"Error running frequency analysis: {str(e)}",
            "fft_base64": ""
        }
