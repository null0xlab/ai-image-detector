import cv2
import numpy as np
import io
from PIL import Image

def analyze_compression(
    pil_img: Image.Image, cv_img_bgr: np.ndarray, is_lossless_origin: bool = False
) -> dict:
    """
    Applies Error Level Analysis (ELA) and DCT coefficient Benford's Law testing
    to identify compression and synthesis inconsistencies.
    Returns:
        score: float (0 - 100)
        details: str
    """
    try:
        # --- 1. Error Level Analysis (ELA) ---
        # Resave at quality = 95
        out = io.BytesIO()
        # Convert PIL image to RGB to handle PNG/RGBA transparency
        ela_temp = pil_img.convert("RGB")
        ela_temp.save(out, format="JPEG", quality=95)
        out.seek(0)
        resaved = Image.open(out)
        
        # Calculate pixel difference
        original_np = np.array(ela_temp).astype(float)
        resaved_np = np.array(resaved).astype(float)
        
        ela_diff = np.abs(original_np - resaved_np)
        mean_diff = np.mean(ela_diff)
        
        # Analyze ELA variance across flat vs textured regions
        # Convert BGR to grayscale for edge mask
        gray = cv2.cvtColor(cv_img_bgr, cv2.COLOR_BGR2GRAY)
        gray_resized = cv2.resize(gray, (ela_diff.shape[1], ela_diff.shape[0]))
        
        # Compute edge mask using Sobel filters
        sobelx = cv2.Sobel(gray_resized, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray_resized, cv2.CV_64F, 0, 1, ksize=3)
        edges = np.sqrt(sobelx**2 + sobely**2)
        
        # Binary mask for flat regions (low edge energy) vs edge regions
        flat_mask = edges < np.percentile(edges, 40)
        edge_mask = edges > np.percentile(edges, 80)
        
        ela_gray = np.mean(ela_diff, axis=2)
        
        # ELA properties:
        # In a real photo, ELA is very active on edges, but extremely quiet on flat areas.
        # Ratio of ELA on edges vs flat regions is high in natural images.
        # In AI images, the ELA error is highly uniform: the ratio is low.
        ela_edge_mean = np.mean(ela_gray[edge_mask]) if np.any(edge_mask) else 1e-8
        ela_flat_mean = np.mean(ela_gray[flat_mask]) if np.any(flat_mask) else 1e-8
        
        ela_ratio = ela_edge_mean / (ela_flat_mean + 1e-8)
        
        # --- 2. DCT coefficient analysis (Benford's Law) ---
        # Benford's Law first-digit probability distribution
        benford_probs = np.array([np.log10(1 + 1/d) for d in range(1, 10)])
        
        # Resize to multiple of 8
        h, w = gray.shape
        h_new, w_new = (h // 8) * 8, (w // 8) * 8
        gray_8 = cv2.resize(gray, (w_new, h_new), interpolation=cv2.INTER_AREA)
        
        first_digits = []
        
        # Compute DCT on 8x8 blocks
        for r in range(0, h_new, 8):
            for c in range(0, w_new, 8):
                block = gray_8[r:r+8, c:c+8].astype(float)
                dct_block = cv2.dct(block)
                
                # Extract AC coefficients (excluding DC at [0,0])
                # We look at the first few high-power AC coefficients
                ac_coeffs = [dct_block[0, 1], dct_block[1, 0], dct_block[1, 1]]
                for coef in ac_coeffs:
                    val = abs(coef)
                    if val > 0.1:
                        # Extract first significant digit
                        str_val = f"{val:.10f}".replace("0.", "").replace(".", "")
                        for char in str_val:
                            if char != '0':
                                digit = int(char)
                                if 1 <= digit <= 9:
                                    first_digits.append(digit)
                                break
                                
        # Calculate digit frequencies
        first_digits = np.array(first_digits)
        digit_counts = np.zeros(9)
        
        if len(first_digits) > 50:
            for d in range(1, 10):
                digit_counts[d-1] = np.sum(first_digits == d)
            digit_probs = digit_counts / len(first_digits)
            
            # Chi-square divergence between actual digit probs and Benford's Law
            chi_square = np.sum((digit_probs - benford_probs)**2 / benford_probs)
        else:
            chi_square = 0.0
            
        # --- 3. Scoring ---
        findings = []
        compression_score = 0.0
        
        # ELA Uniformity check:
        # In typical real photos, ELA ratio is > 3.5 (edges are at least 3.5x more active than flat areas).
        # In synthetically created images, or uniform ELA structures, ratio is lower (< 2.2).
        ela_threshold = 3.2 if is_lossless_origin else 2.8
        if ela_ratio < ela_threshold:
            ela_penalty = min((ela_threshold - ela_ratio) * 40, 80)
            compression_score += ela_penalty * 0.50
            findings.append(f"spatially uniform Error Level Analysis profile (edge-to-flat compression ratio: {ela_ratio:.2f})")
            
        # DCT Chi-square divergence:
        # High quality natural JPEGs align perfectly with Benford's Law (chi-square < 0.04)
        # Synthetic images, AI generation with unique convolutional blocks, or double JPEGs deviate (chi-square > 0.08)
        chi_threshold = 0.06 if is_lossless_origin else 0.07
        if chi_square > chi_threshold:
            dct_penalty = min((chi_square - chi_threshold) * 220 + 12, 80)
            compression_score += dct_penalty * 0.50
            findings.append(f"DCT AC coefficient distribution deviates from natural photographic laws (Benford divergence: {chi_square:.4f})")
            
        if not findings:
            details = "Compression grid and DCT statistics align with authentic, single-compression photographic profiles."
        else:
            details = "Anomalies detected: " + ", ".join(findings) + "."
            
        final_score = max(min(compression_score, 100.0), 0.0)
        
        return {
            "score": round(final_score, 1),
            "details": details
        }
        
    except Exception as e:
        return {
            "score": 50.0,
            "details": f"Error running compression analysis: {str(e)}"
        }
